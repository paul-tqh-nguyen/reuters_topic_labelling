#!/usr/bin/python3
"#!/usr/bin/python3 -OO"

"""
"""

# @todo fill in the top-level doc string
# @todo add type declarations 
# @todo verify that all the imported stuff is used

###########
# Imports #
###########

import random
import json
import os
from functools import reduce
from typing import List, Tuple, Set
from collections import OrderedDict

import preprocess_data
from misc_utilites import eager_map, eager_filter, timer, tqdm_with_message, debug_on_error, dpn, dpf # @todo get rid of debug_on_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchtext
from torchtext import data

################################################
# Misc. Globals & Global State Initializations #
################################################

SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

NUMBER_OF_RELEVANT_RECENT_EPOCHS = 5

##################
# Helper Classes #
##################

def SoftF1Loss(y_hat:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    batch_size, output_size = tuple(y.shape)
    assert tuple(y.shape) == (batch_size, output_size)
    assert tuple(y_hat.shape) == (batch_size, output_size)
    true_positive_sum = (y_hat * y.float()).sum(dim=1)
    false_positive_sum = (y_hat * (1-y.float())).sum(dim=1)
    false_negative_sum = ((1-y_hat) * y.float()).sum(dim=1)
    soft_recall = true_positive_sum / (true_positive_sum + false_positive_sum + 1e-16)
    soft_precision = true_positive_sum / (true_positive_sum + false_negative_sum + 1e-16)
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall + 1e-16)
    mean_soft_f1 = torch.mean(soft_f1)
    loss = 1-mean_soft_f1
    assert not 'nan' in str(loss)
    return loss

class NumericalizedBatchIterator:
    def __init__(self, non_numericalized_iterator, x_attribute_name, y_attribute_names):
        self.non_numericalized_iterator = non_numericalized_iterator
        self.x_attribute_name: str = x_attribute_name
        self.y_attribute_names: List[str] = y_attribute_names
        
    def __iter__(self):
        for non_numericalized_batch in self.non_numericalized_iterator:
            x = getattr(non_numericalized_batch, self.x_attribute_name)
            y = torch.cat([getattr(non_numericalized_batch, y_attribute_name).unsqueeze(1) for y_attribute_name in self.y_attribute_names], dim=1).float()
            yield (x, y)
            
    def __len__(self):
        return len(self.non_numericalized_iterator)

##########
# Models #
##########

class AttentionLayers(nn.Module):
    def __init__(self, encoding_hidden_size: int, attention_intermediate_size: int, number_of_attention_heads: int, dropout_probability: float) -> None:
        super().__init__()
        self.encoding_hidden_size = encoding_hidden_size
        self.number_of_attention_heads = number_of_attention_heads
        self.attention_layers = nn.Sequential(OrderedDict([
            ("intermediate_attention_layer", nn.Linear(encoding_hidden_size*2, attention_intermediate_size)),
            ("intermediate_attention_dropout_layer", nn.Dropout(dropout_probability)),
            ("attention_activation", nn.ReLU(True)),
            ("final_attention_layer", nn.Linear(attention_intermediate_size, number_of_attention_heads)),
            ("final_attention_dropout_layer", nn.Dropout(dropout_probability)),
            ("softmax_layer", nn.Softmax(dim=0)),
        ]))

    def forward(self, encoded_batch: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        batch_size = text_lengths.shape[0]
        max_sentence_length = encoded_batch.shape[1]
        assert tuple(encoded_batch.shape) == (batch_size, max_sentence_length, self.encoding_hidden_size*2)

        attended_batch = Variable(torch.empty(batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads).to(encoded_batch.device))

        for batch_index in range(batch_size):
            sentence_length = text_lengths[batch_index]
            sentence_matrix = encoded_batch[batch_index, :sentence_length, :]
            assert encoded_batch[batch_index ,sentence_length:, :].data.sum() == 0
            assert tuple(sentence_matrix.shape) == (sentence_length, self.encoding_hidden_size*2)

            sentence_weights = self.attention_layers(sentence_matrix)
            assert tuple(sentence_weights.shape) == (sentence_length, self.number_of_attention_heads)
            assert (sentence_weights.data.sum(dim=0)-1).abs().mean() < 1e-4

            weight_adjusted_sentence_matrix = torch.mm(sentence_matrix.t(), sentence_weights)
            assert tuple(weight_adjusted_sentence_matrix.shape) == (self.encoding_hidden_size*2, self.number_of_attention_heads,)

            concatenated_attention_vectors = weight_adjusted_sentence_matrix.view(-1)
            assert tuple(concatenated_attention_vectors.shape) == (self.encoding_hidden_size*2*self.number_of_attention_heads,)

            attended_batch[batch_index, :] = concatenated_attention_vectors

        assert tuple(attended_batch.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)
        return attended_batch

class EEAPNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, output_size, dropout_probability, pad_idx):
        super().__init__()
        if __debug__:
            self.embedding_size = embedding_size
            self.encoding_hidden_size = encoding_hidden_size
            self.number_of_encoding_layers = number_of_encoding_layers
            self.number_of_attention_heads = number_of_attention_heads
            self.output_size = output_size
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("embedding_layer", nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx, max_norm=1.0)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
        ]))
        self.encoding_layers = nn.LSTM(embedding_size,
                                       encoding_hidden_size,
                                       num_layers=number_of_encoding_layers,
                                       bidirectional=True,
                                       dropout=dropout_probability)
        self.attention_layers = AttentionLayers(encoding_hidden_size, attention_intermediate_size, number_of_attention_heads, dropout_probability)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layer", nn.Linear(encoding_hidden_size*2*number_of_attention_heads, output_size)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("sigmoid_layer", nn.Sigmoid()),
        ]))

    def forward(self, text_batch, text_lengths):
        if __debug__:
            max_sentence_length = max(text_lengths)
            batch_size = text_batch.shape[0]
        assert tuple(text_batch.shape) == (batch_size, max_sentence_length)
        assert tuple(text_lengths.shape) == (batch_size,)

        embedded_batch = self.embedding_layers(text_batch)
        assert tuple(embedded_batch.shape) == (batch_size, max_sentence_length, self.embedding_size)

        embedded_batch_packed = nn.utils.rnn.pack_padded_sequence(embedded_batch, text_lengths, batch_first=True)
        if __debug__:
            encoded_batch_packed, (encoding_hidden_state, encoding_cell_state) = self.encoding_layers(embedded_batch_packed)
            encoded_batch, encoded_batch_lengths = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        else:
            encoded_batch_packed, _ = self.encoding_layers(embedded_batch_packed)
            encoded_batch, _ = nn.utils.rnn.pad_packed_sequence(encoded_batch_packed, batch_first=True)
        assert tuple(encoded_batch.shape) == (batch_size, max_sentence_length, self.encoding_hidden_size*2)
        assert tuple(encoding_hidden_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoding_cell_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoded_batch_lengths.shape) == (batch_size,)
        assert (encoded_batch_lengths.to(DEVICE) == text_lengths).all()

        attended_batch = self.attention_layers(encoded_batch, text_lengths)
        assert tuple(attended_batch.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)
        
        prediction = self.prediction_layers(attended_batch)
        assert tuple(prediction.shape) == (batch_size, self.output_size)
        
        return prediction

###############
# Classifiers #
###############

class EEAPClassifier():
    def __init__(self, number_of_epochs: int, batch_size: int, train_portion: float, validation_portion: float, testing_portion: float, max_vocab_size: int, pre_trained_embedding_specification: str, encoding_hidden_size: int, number_of_encoding_layers: int, attention_intermediate_size: int, number_of_attention_heads: int, dropout_probability: float, output_directory: str):
        super().__init__()
        self.best_valid_loss = float('inf')

        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        self.pre_trained_embedding_specification = pre_trained_embedding_specification
        self.encoding_hidden_size = encoding_hidden_size
        self.number_of_encoding_layers = number_of_encoding_layers
        self.attention_intermediate_size = attention_intermediate_size
        self.number_of_attention_heads = number_of_attention_heads
        self.dropout_probability = dropout_probability
        self.train_portion = train_portion
        self.validation_portion = validation_portion
        self.testing_portion = testing_portion
        
        self.load_data()
        self.initialize_model()
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    @debug_on_error # @todo remove this decorator
    def load_data(self):
        self.text_field = data.Field(tokenize = 'spacy', include_lengths = True, batch_first = True)
        self.label_field = data.LabelField(dtype = torch.long)
        with open(preprocess_data.TOPICS_DATA_OUTPUT_CSV_FILE, 'r') as topics_csv_file:
            column_names = eager_map(str.strip, topics_csv_file.readline().split(','))
            column_name_to_field_map = [(column_name, self.text_field if column_name=='text' else
                                         None if column_name in preprocess_data.COLUMNS_RELEVANT_TO_TOPICS_DATA else
                                         self.label_field) for column_name in column_names]
        self.topics: List[str] = list(set(column_names)-preprocess_data.COLUMNS_RELEVANT_TO_TOPICS_DATA)
        self.output_size = len(self.topics)
        self.all_data = data.dataset.TabularDataset(
            path=preprocess_data.TOPICS_DATA_OUTPUT_CSV_FILE,
            format='csv',
            skip_header=True,
            fields=column_name_to_field_map)
        self.training_data, self.validation_data, self.testing_data = self.all_data.split(split_ratio=[self.train_portion, self.validation_portion, self.testing_portion], random_state = random.seed(SEED))
        self.embedding_size = torchtext.vocab.pretrained_aliases[self.pre_trained_embedding_specification]().dim
        self.text_field.build_vocab(self.training_data, max_size = self.max_vocab_size, vectors = self.pre_trained_embedding_specification, unk_init = torch.Tensor.normal_)
        self.label_field.build_vocab(self.training_data)
        assert self.text_field.vocab.vectors.shape[0] <= self.max_vocab_size+2
        assert self.text_field.vocab.vectors.shape[1] == self.embedding_size
        self.training_iterator, self.validation_iterator, self.testing_iterator = data.BucketIterator.splits(
            (self.training_data, self.validation_data, self.testing_data),
            batch_size = self.batch_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch = True,
            repeat=False,
            device = DEVICE)
        self.training_iterator = NumericalizedBatchIterator(self.training_iterator, 'text', self.topics)
        self.validation_iterator = NumericalizedBatchIterator(self.validation_iterator, 'text', self.topics)
        self.testing_iterator = NumericalizedBatchIterator(self.testing_iterator, 'text', self.topics)
        self.pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.unk_idx = self.text_field.vocab.stoi[self.text_field.unk_token]
        
    def determine_training_unknown_words(self) -> None:
        pretrained_embedding_vectors = torchtext.vocab.pretrained_aliases[self.pre_trained_embedding_specification]()
        pretrained_embedding_vectors_unk_default_tensor = pretrained_embedding_vectors.unk_init(torch.Tensor(pretrained_embedding_vectors.dim))
        is_unk_token = lambda token: torch.all(pretrained_embedding_vectors[token] == pretrained_embedding_vectors_unk_default_tensor)
        tokens = reduce(set.union, (set(map(str,example.text)) for example in self.training_data))
        self.training_unk_words = set(eager_filter(is_unk_token, tokens))
        return
        
    def initialize_model(self) -> None:
        vocab_size = len(self.text_field.vocab)
        self.model = EEAPNetwork(vocab_size, self.embedding_size, self.encoding_hidden_size, self.number_of_encoding_layers, self.attention_intermediate_size, self.number_of_attention_heads, self.output_size, self.dropout_probability, self.pad_idx)
        self.model.embedding_layers.embedding_layer.weight.data.copy_(self.text_field.vocab.vectors)
        self.model.embedding_layers.embedding_layer.weight.data[self.unk_idx] = torch.zeros(self.embedding_size)
        self.model.embedding_layers.embedding_layer.weight.data[self.pad_idx] = torch.zeros(self.embedding_size)
        self.model = self.model.to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.BCELoss().to(DEVICE)
        # self.loss_function = SoftF1Loss # @todo see if we can get this working.
        return

    def train_one_epoch(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_f1 = 0
        number_of_training_batches = len(self.training_iterator)
        self.model.train()
        for (texts, text_lengths), multiclass_labels in tqdm_with_message(self.training_iterator, post_yield_message_func = lambda index: f'Training F1 {epoch_f1/(index+1):.8f}', total=number_of_training_batches, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            self.optimizer.zero_grad()
            predictions = self.model(texts, text_lengths)
            loss = self.loss_function(predictions, multiclass_labels)
            f1 = self.f1_score_of_discretized_values(predictions, multiclass_labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_f1 += f1
        return epoch_loss / number_of_training_batches, epoch_f1 / number_of_training_batches

    def evaluate(self, iterator, iterator_is_test_set) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_f1 = 0
        self.model.eval()
        with torch.no_grad():
            for (texts, text_lengths), multiclass_labels in tqdm_with_message(iterator, post_yield_message_func = lambda index: f'{"Testing" if iterator_is_test_set else "Validation"} F1 {epoch_f1/(index+1):.8f}', total=len(iterator), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                predictions = self.model(texts, text_lengths).squeeze(1)
                loss = self.loss_function(predictions, multiclass_labels)
                f1 = self.f1_score_of_discretized_values(predictions, multiclass_labels)
                epoch_loss += loss.item()
                epoch_f1 += f1
        return epoch_loss / len(iterator), epoch_f1 / len(iterator)

    def validate(self) -> Tuple[float, float]:
        return self.evaluate(self.validation_iterator, False)

    def test(self, epoch_index: int, result_is_from_final_run: bool) -> None:
        test_loss, test_f1 = self.evaluate(self.testing_iterator, True)
        print(f'\t  Test F1: {test_f1:.8f} |  Test Loss: {test_loss:.8f}')
        if not os.path.isfile('global_best_model_score.json'):
            log_current_model_as_best = True
        else:
            with open('global_best_model_score.json', 'r') as current_global_best_model_score_json_file:
                current_global_best_model_score_dict = json.load(current_global_best_model_score_json_file)
                current_global_best_model_f1: float = current_global_best_model_score_dict['test_f1']
                log_current_model_as_best = current_global_best_model_f1 < test_f1
        self_score_dict = {
            'best_valid_loss': self.best_valid_loss,
            'number_of_epochs': self.number_of_epochs,
            'most_recently_completed_epoch_index': epoch_index,
            'batch_size': self.batch_size,
            'max_vocab_size': self.max_vocab_size,
            'vocab_size': len(self.text_field.vocab), 
            'pre_trained_embedding_specification': self.pre_trained_embedding_specification,
            'encoding_hidden_size': self.encoding_hidden_size,
            'number_of_encoding_layers': self.number_of_encoding_layers,
            'attention_intermediate_size': self.attention_intermediate_size,
            'number_of_attention_heads': self.number_of_attention_heads,
            'dropout_probability': self.dropout_probability,
            'output_size': self.output_size,
            'train_portion': self.train_portion,
            'validation_portion': self.validation_portion,
            'testing_portion': self.testing_portion,
            'number_of_parameters': self.count_parameters(),
            'test_f1': test_f1,
            'test_loss': test_loss,
        }
        if log_current_model_as_best:
            with open('global_best_model_score.json', 'w') as outfile:
                json.dump(self_score_dict, outfile)
        latest_model_score_location = os.path.join(self.output_directory, 'latest_model_score.json')
        with open(latest_model_score_location, 'w') as outfile:
            json.dump(self_score_dict, outfile)
        if result_is_from_final_run:
            os.remove(latest_model_score_location)
            with open(os.path.join(self.output_directory, 'final_model_score.json'), 'w') as outfile:
                json.dump(self_score_dict, outfile)
        return
    
    def train(self) -> None:
        self.print_hyperparameters()
        best_saved_model_location = os.path.join(self.output_directory, 'best-model.pt')
        most_recent_validation_f1_scores = [0]*NUMBER_OF_RELEVANT_RECENT_EPOCHS
        print(f'Starting training')
        for epoch_index in range(self.number_of_epochs):
            print("\n\n\n")
            print(f"Epoch {epoch_index}")
            with timer(section_name=f"Epoch {epoch_index}"):
                train_loss, train_f1 = self.train_one_epoch()
                valid_loss, valid_f1 = self.validate()
                print(f'\t Train F1: {train_f1:.8f} | Train Loss: {train_loss:.8f}')
                print(f'\t  Val. F1: {valid_f1:.8f} |  Val. Loss: {valid_loss:.8f}')
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_parameters(best_saved_model_location)
                    self.test(epoch_index, False)
            if reduce(bool.__or__, (valid_f1 > previous_f1 for previous_f1 in most_recent_validation_f1_scores)):
                most_recent_validation_f1_scores.pop(0)
                most_recent_validation_f1_scores.append(valid_f1)
            else:
                print(f"Validation is not better than any of the {NUMBER_OF_RELEVANT_RECENT_EPOCHS} recent epochs, so training is ending early due to apparent convergence.")
                print()
                break
        self.load_parameters(best_saved_model_location)
        self.test(epoch_index, True)
        os.remove(best_saved_model_location)
        return

    def print_hyperparameters(self) -> None:
        print()
        print(f"Model hyperparameters are:")
        print(f'        number_of_epochs: {self.number_of_epochs}')
        print(f'        batch_size: {self.batch_size}')
        print(f'        max_vocab_size: {self.max_vocab_size}')
        print(f'        vocab_size: {len(self.text_field.vocab)}')
        print(f'        pre_trained_embedding_specification: {self.pre_trained_embedding_specification}')
        print(f'        encoding_hidden_size: {self.encoding_hidden_size}')
        print(f'        number_of_encoding_layers: {self.number_of_encoding_layers}')
        print(f'        attention_intermediate_size: {self.attention_intermediate_size}')
        print(f'        number_of_attention_heads: {self.number_of_attention_heads}')
        print(f'        output_size: {self.output_size}')
        print(f'        dropout_probability: {self.dropout_probability}')
        print(f'        output_directory: {self.output_directory}')
        print()
        print(f'The model has {self.count_parameters()} trainable parameters.')
        print(f"This processes's PID is {os.getpid()}.")
        print()
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def f1_score_of_discretized_values(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        batch_size = y.shape[0]
        assert batch_size <= self.batch_size
        assert tuple(y.shape) == (batch_size, self.output_size)
        assert tuple(y_hat.shape) == (batch_size, self.output_size)
        y_hat_discretized = torch.round(y_hat)
        true_positive_count = ((y_hat_discretized == y) & y.bool()).float().sum(dim=1)
        false_positive_count = ((y_hat_discretized != y) & y.bool()).float().sum(dim=1)
        false_negative_count = ((y_hat_discretized != y) & ~y.bool()).float().sum(dim=1)
        _make_safe_divisor = lambda divisor: divisor + (~(divisor.bool())).float()
        recall = true_positive_count / _make_safe_divisor(true_positive_count + false_positive_count)
        precision = true_positive_count / _make_safe_divisor(true_positive_count + false_negative_count)
        f1 = 2 * precision * recall / _make_safe_divisor(precision + recall)
        mean_f1 = torch.mean(f1).item()
        assert isinstance(mean_f1, float)
        return mean_f1

    def save_parameters(self, parameter_file_location: str) -> None:
        torch.save(self.model.state_dict(), parameter_file_location)
        return
    
    def load_parameters(self, parameter_file_location: str) -> None:
        self.model.load_state_dict(torch.load(parameter_file_location))
        return

    @debug_on_error # @todo get rid of this debug_on_error
    def classify_string(self, input_string: str) -> Set[str]:
        self.model.eval()
        tokenized = [token.text for token in self.text_field.tokenize(input_string)]
        indexed = [self.text_field.vocab.stoi[t] for t in tokenized]
        lengths = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(DEVICE)
        tensor = tensor.view(1,-1)
        length_tensor = torch.LongTensor(lengths).to(DEVICE)
        prediction_as_index = self.model(tensor, length_tensor).argmax(dim=1).item() # @todo Is this correct? 
        prediction = self.label_field.vocab.itos[prediction_as_index]
        return prediction

###############
# Main Driver #
###############

if __name__ == '__main__':
    print() # @todo fill this in
