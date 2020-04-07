#!/usr/bin/python3
"#!/usr/bin/python3 -OO"

"""
"""

# @todo fill in the top-level doc string
# @todo add type declarations 

###########
# Imports #
###########

import random
from typing import List, Tuple, Set
from collections import OrderedDict

import preprocess_data
from misc_utilites import eager_map, timer, tqdm_with_message, debug_on_error, dpn, dpf # @todo get rid of debug_on_error

import spacy
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

##########
# Models #
##########

class EEPNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoding_hidden_size, number_of_encoding_layers, output_size, dropout_probability, pad_idx):
        super().__init__()
        if __debug__:
            self.embedding_size = embedding_size
            self.encoding_hidden_size = encoding_hidden_size
            self.number_of_encoding_layers = number_of_encoding_layers
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
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layer", nn.Linear(encoding_hidden_size*2, output_size)),
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

        mean_batch = Variable(torch.empty(batch_size, self.encoding_hidden_size*2).to(DEVICE))
        for batch_index in range(batch_size):
            batch_sequence_length = text_lengths[batch_index]
            last_word_index = batch_sequence_length-1
            mean_batch[batch_index, :] = encoded_batch[batch_index,:batch_sequence_length,:].mean(dim=0)
            assert encoded_batch[batch_index,batch_sequence_length:,:].sum() == 0
        assert tuple(mean_batch.shape) == (batch_size, self.encoding_hidden_size*2)
        
        prediction = self.prediction_layers(mean_batch)
        assert tuple(prediction.shape) == (batch_size, self.output_size)
        
        return prediction

###############
# Classifiers #
###############

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

class EEPClassifier(nn.Module):
    def __init__(self, number_of_epochs: int, batch_size: int, train_portion: float, validation_portion: float, testing_portion: float, max_vocab_size: int, pre_trained_embedding_specification: str, encoding_hidden_size: int, number_of_encoding_layers: int, dropout_probability: float):
        super().__init__()
        self.best_valid_loss = float('inf')

        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        self.pre_trained_embedding_specification = pre_trained_embedding_specification
        self.encoding_hidden_size = encoding_hidden_size
        self.number_of_encoding_layers = number_of_encoding_layers
        self.dropout_probability = dropout_probability
        self.train_portion = train_portion
        self.validation_portion = validation_portion
        self.testing_portion = testing_portion
        
        self.load_data()
        self.initialize_model()
        self.nlp = spacy.load('en')

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
        all_data = data.dataset.TabularDataset(
            path=preprocess_data.TOPICS_DATA_OUTPUT_CSV_FILE,
            format='csv',
            skip_header=True,
            fields=column_name_to_field_map)
        training_data, validation_data, testing_data = all_data.split(split_ratio=[self.train_portion, self.validation_portion, self.testing_portion], random_state = random.seed(SEED))
        self.text_field.build_vocab(training_data, max_size = self.max_vocab_size, vectors = self.pre_trained_embedding_specification, unk_init = torch.Tensor.normal_)
        self.label_field.build_vocab(training_data)
        assert self.text_field.vocab.vectors.shape[0] <= self.max_vocab_size+2
        assert self.text_field.vocab.vectors.shape[1] == self.dimensionality_from_pre_trained_embedding_specification()
        self.training_iterator, self.validation_iterator, self.testing_iterator = data.BucketIterator.splits(
            (training_data, validation_data, testing_data),
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

    def initialize_model(self) -> None:
        vocab_size = len(self.text_field.vocab)
        embedding_size = self.dimensionality_from_pre_trained_embedding_specification()
        self.model = EEPNetwork(vocab_size, embedding_size, self.encoding_hidden_size, self.number_of_encoding_layers, self.output_size, self.dropout_probability, self.pad_idx)
        self.model.embedding_layers.embedding_layer.weight.data.copy_(self.text_field.vocab.vectors)
        self.model.embedding_layers.embedding_layer.weight.data[self.unk_idx] = torch.zeros(embedding_size)
        self.model.embedding_layers.embedding_layer.weight.data[self.pad_idx] = torch.zeros(embedding_size)
        self.model = self.model.to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.BCELoss()
        self.loss_function = self.loss_function.to(DEVICE)
        return
    
    def dimensionality_from_pre_trained_embedding_specification(self) -> int:
        if 'dim' in torchtext.vocab.pretrained_aliases[self.pre_trained_embedding_specification].keywords:
            return int(torchtext.vocab.pretrained_aliases[self.pre_trained_embedding_specification].keywords['dim'])
        else:
            return int(self.pre_trained_embedding_specification.split('.')[-1][:-1])

    def train_one_epoch(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_f1 = 0
        number_of_training_batches = len(self.training_iterator)
        self.model.train()
        for (texts, text_lengths), multiclass_labels in tqdm_with_message(self.training_iterator, post_yield_message_func = lambda index: f'Training F1 {epoch_f1/(index+1):.8f}%', total=number_of_training_batches, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
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
            for (texts, text_lengths), multiclass_labels in tqdm_with_message(iterator, post_yield_message_func = lambda index: f'{"Testing" if iterator_is_test_set else "Validation"} F1 {epoch_f1/(index+1):.8f}%', total=len(iterator), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                predictions = self.model(texts, text_lengths).squeeze(1)
                loss = self.loss_function(predictions, multiclass_labels)
                f1 = self.f1_score_of_discretized_values(predictions, multiclass_labels)
                epoch_loss += loss.item()
                epoch_f1 += f1
        return epoch_loss / len(iterator), epoch_f1 / len(iterator)

    def validate(self) -> Tuple[float, float]:
        return self.evaluate(self.validation_iterator, False)

    def test(self) -> None:
        test_loss, test_f1 = self.evaluate(self.testing_iterator, True)
        print(f'Test Loss: {test_loss:.8f} | Test F1: {test_f1:.8f}%')
        return
    
    def train(self) -> None:
        self.print_hyperparameters()
        print(f'Starting training')
        for epoch_index in range(self.number_of_epochs):
            print("\n\n\n")
            print(f"Epoch {epoch_index}")
            with timer(section_name=f"Epoch {epoch_index}"):
                train_loss, train_f1 = self.train_one_epoch()
                valid_loss, valid_f1 = self.validate()
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_parameters('best-model.pt')
                print(f'\tTrain Loss: {train_loss:.8f} | Train F1: {train_f1:.8f}%')
                print(f'\t Val. Loss: {valid_loss:.8f} |  Val. F1: {valid_f1:.8f}%')
        self.load_parameters('best-model.pt')
        self.test()
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
        print(f'        output_size: {self.output_size}')
        print(f'        dropout_probability: {self.dropout_probability}')
        print()
        print(f'The model has {self.count_parameters()} trainable parameters')
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
        recall = true_positive_count / (true_positive_count + false_positive_count)
        precision = true_positive_count / (true_positive_count + false_negative_count)
        mean_f1 = torch.mean(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return mean_f1

    def save_parameters(self, parameter_file_location: str) -> None:
        torch.save(self.model.state_dict(), parameter_file_location)
        return
    
    def load_parameters(self, parameter_file_location: str) -> None:
        self.model.load_state_dict(torch.load(parameter_file_location))
        return
    
    @debug_on_error
    def classify_string(self, input_string: str) -> Set[str]:
        self.model.eval()
        tokenized = [token.text for token in self.nlp.tokenizer(input_string)]
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
