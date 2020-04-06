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
from typing import List, Tuple
from collections import OrderedDict

import preprocess_data
from misc_utilites import eager_map, timer, tqdm_with_message
from misc_utilites import debug_on_error, dpn # @todo remove this

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

###########################
# Domain Specific Helpers #
###########################

def dimensionality_from_pre_trained_embedding_specification(pre_trained_embedding_specification: str) -> int:
    if 'dim' in torchtext.vocab.pretrained_aliases[pre_trained_embedding_specification].keywords:
        return int(torchtext.vocab.pretrained_aliases[pre_trained_embedding_specification].keywords['dim'])
    else:
        return int(pre_trained_embedding_specification.split('.')[-1][:-1])

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
        self.load_data(batch_size, train_portion, validation_portion, testing_portion, max_vocab_size, pre_trained_embedding_specification)
        self.initialize_model(pre_trained_embedding_specification, encoding_hidden_size, number_of_encoding_layers, dropout_probability)
        self.best_valid_loss = float('inf')
        self.number_of_epochs = number_of_epochs

    def load_data(self, batch_size: int, train_portion: float, validation_portion: float, testing_portion: float, max_vocab_size: int, pre_trained_embedding_specification: str):
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
        training_data, validation_data, testing_data = all_data.split(split_ratio=[train_portion, validation_portion, testing_portion], random_state = random.seed(SEED))
        self.text_field.build_vocab(training_data, max_size = max_vocab_size, vectors = pre_trained_embedding_specification, unk_init = torch.Tensor.normal_)
        self.label_field.build_vocab(training_data)
        assert self.text_field.vocab.vectors.shape[0] <= max_vocab_size+2
        assert self.text_field.vocab.vectors.shape[1] == dimensionality_from_pre_trained_embedding_specification(pre_trained_embedding_specification)
        self.training_iterator, self.validation_iterator, self.testing_iterator = data.BucketIterator.splits(
            (training_data, validation_data, testing_data),
            batch_size = batch_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch = True,
            repeat=False,
            device = DEVICE)
        self.training_iterator = NumericalizedBatchIterator(self.training_iterator, 'text', self.topics)
        self.validation_iterator = NumericalizedBatchIterator(self.validation_iterator, 'text', self.topics)
        self.testing_iterator = NumericalizedBatchIterator(self.testing_iterator, 'text', self.topics)
        self.pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.unk_idx = self.text_field.vocab.stoi[self.text_field.unk_token]

    def initialize_model(self, pre_trained_embedding_specification: str, encoding_hidden_size: int, number_of_encoding_layers: int, dropout_probability: int) -> None:
        vocab_size = len(self.text_field.vocab)
        embedding_size = dimensionality_from_pre_trained_embedding_specification(pre_trained_embedding_specification)
        self.model = EEPNetwork(vocab_size, embedding_size, encoding_hidden_size, number_of_encoding_layers, self.output_size, dropout_probability, self.pad_idx)
        self.model.embedding_layers.embedding_layer.weight.data.copy_(self.text_field.vocab.vectors)
        self.model.embedding_layers.embedding_layer.weight.data[self.unk_idx] = torch.zeros(embedding_size)
        self.model.embedding_layers.embedding_layer.weight.data[self.pad_idx] = torch.zeros(embedding_size)
        self.model = self.model.to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.BCELoss()
        self.loss_function = self.loss_function.to(DEVICE)
        return

    def train_one_epoch(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_acc = 0
        number_of_training_batches = len(self.training_iterator)
        self.model.train()
        for (texts, text_lengths), multiclass_labels in tqdm_with_message(self.training_iterator, post_yield_message_func = lambda index: f'Training Accuracy {epoch_acc/(index+1)*100:.8f}%', total=number_of_training_batches, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            self.optimizer.zero_grad()
            predictions = self.model(texts, text_lengths)
            loss = self.loss_function(predictions, multiclass_labels) # @todo verify that these dimensions are correct
            acc = self.discrete_accuracy(predictions, multiclass_labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / number_of_training_batches, epoch_acc / number_of_training_batches

    def evaluate(self, iterator, iterator_is_test_set) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            for (texts, text_lengths), multiclass_labels in tqdm_with_message(iterator, post_yield_message_func = lambda index: f'{"Testing" if iterator_is_test_set else "Validation"} Accuracy {epoch_acc/(index+1)*100:.8f}%', total=len(iterator), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                predictions = self.model(texts, text_lengths).squeeze(1)
                loss = self.loss_function(predictions, multiclass_labels) # @todo verify that these dimensions are correct
                acc = self.discrete_accuracy(predictions, multiclass_labels)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def validate(self) -> Tuple[float, float]:
        return self.evaluate(self.validation_iterator, False)

    def test(self) -> Tuple[float, float]:
        return self.evaluate(self.testing_iterator, True)
    
    @debug_on_error # @todo remove this debugging decorator
    def train(self) -> None:
        print(f'The model has {self.count_parameters()}, trainable parameters')
        print(f'Starting training')
        for epoch_index in range(self.number_of_epochs):
            with timer(section_name=f"Epoch {epoch_index}"):
                train_loss, train_acc = self.train_one_epoch()
                valid_loss, valid_acc = self.validate()
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), 'best-model.pt')
                print(f'\tTrain Loss: {train_loss:.8f} | Train Acc: {train_acc*100:.8f}%')
                print(f'\t Val. Loss: {valid_loss:.8f} |  Val. Acc: {valid_acc*100:.8f}%')
        self.model.load_state_dict(torch.load('best-model.pt'))
        test_loss, test_acc = self.test()
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
        return
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def discrete_accuracy(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = y.shape[0]
        y_hat_discretized = torch.round(y_hat)
        number_of_correct_answers = (y_hat_discretized == y).sum(dim=1)
        accuracy = number_of_correct_answers.float() / len(self.topics)
        mean_accuracy = torch.mean(accuracy)
        return mean_accuracy
    
###############
# Main Driver #
###############

if __name__ == '__main__':
    print() # @todo fill this in
    # @todo get rid of this
    number_of_epochs = 1
    batch_size = 32
    max_vocab_size = 25_000
    train_portion, validation_portion, testing_portion = (0.50, 0.20, 0.3)
    
    pre_trained_embedding_specification = "glove.6B.100d"
    encoding_hidden_size = 256
    number_of_encoding_layers = 2
    dropout_probability = 0.5

    classifier = EEPClassifier(number_of_epochs, batch_size, train_portion, validation_portion, testing_portion, max_vocab_size, pre_trained_embedding_specification, encoding_hidden_size, number_of_encoding_layers, dropout_probability)
    classifier.train()
