#!/usr/bin/python3
"#!/usr/bin/python3 -OO"

"""
"""

# @todo abstract the contents of this file out
# @todo fill in the top-level doc string
# @todo add type declarations 
# @todo verify that all the imported stuff is used

###########
# Imports #
###########

import random
import json
import os
import math # @todo verify that this is used
from statistics import mean
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Tuple, Set, Callable, Iterable
from collections import OrderedDict

import preprocess_data
from misc_utilites import eager_map, eager_filter, timer, histogram, tqdm_with_message, debug_on_error, current_tensors, dpn, dpf # @todo get rid of debug_on_error

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext
from torchtext import data

################################################
# Misc. Globals & Global State Initializations #
################################################

SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = __debug__
torch.backends.cudnn.benchmark = not __debug__

NUMBER_OF_RELEVANT_RECENT_EPOCHS = 5
GOAL_NUMBER_OF_OVERSAMPLED_DATAPOINTS = 0
PORTION_OF_WORDS_TO_CROP_TO_UNK_FOR_DATA_AUGMENTATION = 0.30

####################
# Helper Utilities #
####################

def tensor_has_nan(tensor: torch.Tensor) -> bool:
    return (tensor != tensor).any().item()

def _safe_count_tensor_division(dividend: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    safe_divisor = divisor + (~(divisor.bool())).float()
    answer = dividend / safe_divisor
    assert not tensor_has_nan(answer)
    return answer

def soft_f1_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch_size, output_size = tuple(y.shape)
    assert tuple(y.shape) == (batch_size, output_size)
    assert tuple(y_hat.shape) == (batch_size, output_size)
    true_positive_sum = (y_hat * y.float()).sum(dim=0)
    false_positive_sum = (y_hat * (1-y.float())).sum(dim=0)
    false_negative_sum = ((1-y_hat) * y.float()).sum(dim=0)
    soft_recall = true_positive_sum / (true_positive_sum + false_negative_sum + 1e-16)
    soft_precision = true_positive_sum / (true_positive_sum + false_positive_sum + 1e-16)
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall + 1e-16)
    mean_soft_f1 = torch.mean(soft_f1)
    loss = 1-mean_soft_f1
    assert not tensor_has_nan(loss)
    return loss

def soft_recall_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batch_size, output_size = tuple(y.shape)
    assert tuple(y.shape) == (batch_size, output_size)
    assert tuple(y_hat.shape) == (batch_size, output_size)
    true_positive_sum = (y_hat * y.float()).sum(dim=0)
    false_negative_sum = ((1-y_hat) * y.float()).sum(dim=0)
    soft_recall = true_positive_sum / (true_positive_sum + false_negative_sum + 1e-16)
    mean_soft_recall = torch.mean(soft_recall)
    loss = 1-mean_soft_recall
    assert not tensor_has_nan(loss)
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

############################
# Data Balancing Utilities #
############################

def chi_squared_loss(histogram: torch.Tensor) -> torch.Tensor:
    expected_value = torch.mean(histogram)
    loss = torch.sum((histogram-expected_value)**2)/expected_value
    return loss

def normalize_occurences(unnormalized_occurences: torch.Tensor) -> torch.Tensor:
    positive_occurences = torch.exp(unnormalized_occurences)
    greater_than_one_occurences = positive_occurences + 1
    return greater_than_one_occurences

def _sanity_check_discrete_occurences(discrete_occurences: torch.Tensor, original_labels_matrix: torch.Tensor, example_numericalized_labels: Callable[[torchtext.data.example.Example], torch.Tensor], dataset: torchtext.data.dataset.Dataset) -> None:
    if __debug__:
        label_to_index_and_example_pairs_map = dict()
        for example_index, (label_tensor, example) in enumerate(zip(original_labels_matrix, dataset)):
            label = tuple(eager_map(torch.Tensor.item, label_tensor))
            assert label == tuple(eager_map(torch.Tensor.item, example_numericalized_labels(example)))
            if label in label_to_index_and_example_pairs_map:
                label_to_index_and_example_pairs_map[label].append((example_index, example))
            else:
                label_to_index_and_example_pairs_map[label] = [(example_index, example)]
        for label, index_and_example_pairs in label_to_index_and_example_pairs_map.items():
            indices = eager_map(lambda x: x[0], index_and_example_pairs)
            mean_discrete_occurences_for_label = mean([discrete_occurences[index].item() for index in indices])
            for index, example in index_and_example_pairs:
                assert discrete_occurences[index].item() - mean_discrete_occurences_for_label < 1
    return

def balance_dataset_wrt_chi_squared_test(dataset: torchtext.data.dataset.Dataset, example_numericalized_labels: Callable[[torchtext.data.example.Example], torch.Tensor]) -> torchtext.data.dataset.Dataset:
    if GOAL_NUMBER_OF_OVERSAMPLED_DATAPOINTS == 0:
        oversampled_dataset = dataset
    else: # @todo this causes a GPU memory leak
        original_labels_matrix = torch.stack([example_numericalized_labels(example) for example in dataset]).to(DEVICE)
        occurences = nn.Parameter(math.log(GOAL_NUMBER_OF_OVERSAMPLED_DATAPOINTS / len(dataset)) * torch.ones([len(dataset),1], dtype=float).to(DEVICE))
        optimizer = optim.Adam([occurences])
        count = 0
        number_of_recent_losses_relevant_to_convergence = 10
        most_recent_losses_via_chi_squared = [1e99]*number_of_recent_losses_relevant_to_convergence
        chi_squared_threshold = 1
        loss_via_chi_squared_is_sufficiently_low = False
        minimum_distance_for_convergence = 1e-6
        convergence_reached = False
        while not convergence_reached and not loss_via_chi_squared_is_sufficiently_low:
            optimizer.zero_grad()
            normalized_occurences = normalize_occurences(occurences)
            total_histogram = (original_labels_matrix * normalized_occurences).sum(dim=0)
            loss_via_chi_squared = chi_squared_loss(total_histogram)

            number_of_oversampled_datapoints = normalized_occurences.sum(dim=0) - len(dataset)
            loss_via_number_of_samples_from_goal = (number_of_oversampled_datapoints - GOAL_NUMBER_OF_OVERSAMPLED_DATAPOINTS) ** 2
            
            loss = loss_via_chi_squared + loss_via_number_of_samples_from_goal
            loss.backward()
            optimizer.step()
            
            discrete_occurences = normalized_occurences.detach().round().int()
            discrete_number_of_oversampled_datapoints = discrete_occurences.sum(dim=0).item() - len(dataset)

            loss_via_chi_squared_is_sufficiently_low = loss_via_chi_squared.item() < chi_squared_threshold
            convergence_reached = abs(loss_via_chi_squared.item() - mean(most_recent_losses_via_chi_squared)) < minimum_distance_for_convergence
            most_recent_losses_via_chi_squared.pop(0)
            most_recent_losses_via_chi_squared.append(loss_via_chi_squared.item())
        _sanity_check_discrete_occurences(discrete_occurences, original_labels_matrix, example_numericalized_labels, dataset)
        oversampled_examples = []
        for example, example_number_of_occurences in zip(dataset, discrete_occurences):
            for _ in range(example_number_of_occurences.item()):
                oversampled_examples.append(example)
        oversampled_dataset = torchtext.data.dataset.Dataset(oversampled_examples, dataset.fields)
        assert torch.all(torch.stack(eager_map(example_numericalized_labels, oversampled_dataset)).to(DEVICE).sum(dim=0) == (original_labels_matrix * discrete_occurences).sum(dim=0))
    return oversampled_dataset

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
            assert encoded_batch[batch_index, sentence_length:, :].data.sum() == 0
            assert tuple(sentence_matrix.shape) == (sentence_length, self.encoding_hidden_size*2)

            sentence_weights = self.attention_layers(sentence_matrix)
            assert tuple(sentence_weights.shape) == (sentence_length, self.number_of_attention_heads)
            assert (sentence_weights.data.sum(dim=0)-1).abs().mean() < 1e-4 # @todo can we use math.isclose here?

            weight_adjusted_sentence_matrix = torch.mm(sentence_matrix.t(), sentence_weights)
            assert tuple(weight_adjusted_sentence_matrix.shape) == (self.encoding_hidden_size*2, self.number_of_attention_heads,)

            concatenated_attention_vectors = weight_adjusted_sentence_matrix.view(-1)
            assert tuple(concatenated_attention_vectors.shape) == (self.encoding_hidden_size*2*self.number_of_attention_heads,)

            attended_batch[batch_index, :] = concatenated_attention_vectors

        assert tuple(attended_batch.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)
        return attended_batch

class EEAPNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, output_size, dropout_probability, pad_idx, unk_idx, initial_embedding_vectors):
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
        self.embedding_layers.embedding_layer.weight.data.copy_(initial_embedding_vectors)
        self.embedding_layers.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)
        self.embedding_layers.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)
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
        self.to(DEVICE)

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
        assert (encoded_batch_lengths.to(text_lengths.device) == text_lengths).all()

        attended_batch = self.attention_layers(encoded_batch, text_lengths)
        assert tuple(attended_batch.shape) == (batch_size, self.encoding_hidden_size*2*self.number_of_attention_heads)
        
        prediction = self.prediction_layers(attended_batch)
        assert tuple(prediction.shape) == (batch_size, self.output_size)
        
        return prediction

class ConvNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, convolution_hidden_size: int, kernel_sizes: List[int], pooling_method: Callable[[torch.Tensor], torch.Tensor], output_size: int, dropout_probability: float, pad_idx: int, unk_idx: int, initial_embedding_vectors): # @todo declare type of initial_embedding_vectors
        super().__init__()
        if __debug__:
            self.embedding_size = embedding_size
            self.convolution_hidden_size = convolution_hidden_size
            self.kernel_sizes = kernel_sizes
            self.output_size = output_size
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("embedding_layer", nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx, max_norm=1.0)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
        ]))
        self.embedding_layers.embedding_layer.weight.data.copy_(initial_embedding_vectors)
        self.embedding_layers.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)
        self.embedding_layers.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)
        self.convolutional_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_size, out_channels=convolution_hidden_size, kernel_size=kernel_size)
            for kernel_size in kernel_sizes # @todo add drop out here.
        ])
        self.pooling_method = pooling_method
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layers", nn.Linear(len(kernel_sizes) * convolution_hidden_size, output_size)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("sigmoid_layer", nn.Sigmoid()),
        ]))
        self.dropout_layers = nn.Dropout(dropout_probability)
        self.to(DEVICE)
        
    def forward(self, text_batch: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        batch_size = text_lengths.shape[0]
        max_sentence_length = text_batch.shape[1]
        assert tuple(text_batch.shape) == (batch_size, max_sentence_length)
        assert tuple(text_lengths.shape) == (batch_size,)
        
        embedded_batch = self.embedding_layers(text_batch)
        assert tuple(embedded_batch.shape) == (batch_size, max_sentence_length, self.embedding_size)
        embedded_batch = embedded_batch.permute(0, 2, 1)
        assert tuple(embedded_batch.shape) == (batch_size, self.embedding_size, max_sentence_length)
        
        convolved_batches = [F.relu(conv(embedded_batch)) for conv in self.convolutional_layers]
        assert reduce(bool.__and__, [tuple(convolved_batch.shape) == (batch_size, self.convolution_hidden_size, max_sentence_length-kernel_size+1)
                                     for convolved_batch, kernel_size in zip(convolved_batches, self.kernel_sizes)])
        
        pooled_batches = [self.pooling_method(convolved_batch, kernel_size=convolved_batch.shape[2]).squeeze(2) for convolved_batch in convolved_batches]
        assert reduce(bool.__and__, [tuple(pooled_batch.shape) == (batch_size, self.convolution_hidden_size) for pooled_batch in pooled_batches])
        
        concatenated_pooled_batches = torch.cat(pooled_batches, dim = 1)
        concatenated_pooled_batches = self.dropout_layers(concatenated_pooled_batches)
        assert tuple(concatenated_pooled_batches.shape) == (batch_size, self.convolution_hidden_size * len(self.kernel_sizes))

        prediction = self.prediction_layers(concatenated_pooled_batches)
        assert tuple(prediction.shape) == (batch_size, self.output_size)
        
        return prediction

class DenseNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, dense_hidden_sizes: Iterable[int], output_size: int, dropout_probability: float, pad_idx: int, unk_idx: int, initial_embedding_vectors): # @todo declare type of initial_embedding_vectors
        super().__init__()
        self.max_sequence_length = dense_hidden_sizes[0]
        if __debug__:
            self.embedding_size = embedding_size
            self.dense_hidden_sizes = dense_hidden_sizes
            self.final_hidden_size = dense_hidden_sizes[-1]
            self.output_size = output_size
        self.embedding_layers = nn.Sequential(OrderedDict([
            ("embedding_layer", nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx, max_norm=1.0)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
        ]))
        self.embedding_layers.embedding_layer.weight.data.copy_(initial_embedding_vectors)
        self.embedding_layers.embedding_layer.weight.data[unk_idx] = torch.zeros(embedding_size)
        self.embedding_layers.embedding_layer.weight.data[pad_idx] = torch.zeros(embedding_size)
        
        previous_dense_hidden_size = self.max_sequence_length
        dense_layers_elements = OrderedDict()
        for dense_hidden_size_index, dense_hidden_size in enumerate(dense_hidden_sizes[1:]):
            linear_layer = nn.Linear(previous_dense_hidden_size*embedding_size, dense_hidden_size*embedding_size)
            dropout_layer = nn.Dropout(dropout_probability)
            activation_layer = nn.ReLU(True)
            dense_layers_elements[f'linear_layer_{dense_hidden_size_index}'] = linear_layer
            dense_layers_elements[f'dropout_layer_{dense_hidden_size_index}'] = dropout_layer
            dense_layers_elements[f'relu_{dense_hidden_size_index}'] = activation_layer
            previous_dense_hidden_size = dense_hidden_size
        self.dense_layers = nn.Sequential(dense_layers_elements)
        
        self.prediction_layers = nn.Sequential(OrderedDict([
            ("fully_connected_layers", nn.Linear(self.final_hidden_size*embedding_size, output_size)),
            ("dropout_layer", nn.Dropout(dropout_probability)),
            ("sigmoid_layer", nn.Sigmoid()),
        ]))
        self.to(DEVICE)
        
    def forward(self, text_batch: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        batch_size = text_lengths.shape[0]
        text_batch_max_sentence_length = text_batch.shape[1]
        assert tuple(text_batch.shape) == (batch_size, text_batch_max_sentence_length)
        assert tuple(text_lengths.shape) == (batch_size,)
        
        if text_batch_max_sentence_length > self.max_sequence_length:
            size_adjusted_text_batch = text_batch[:, :self.max_sequence_length]
        elif text_batch_max_sentence_length < self.max_sequence_length:
            size_adjusted_text_batch = torch.zeros(batch_size, self.max_sequence_length, dtype=text_batch.dtype).to(text_batch.device)
            size_adjusted_text_batch[:, :text_batch_max_sentence_length] = text_batch
        else:
            size_adjusted_text_batch = text_batch
        
        embedded_batch = self.embedding_layers(size_adjusted_text_batch)
        embedded_batch = embedded_batch.view(batch_size, -1)
        assert tuple(embedded_batch.shape) == (batch_size, self.max_sequence_length*self.embedding_size)
        
        encoded_batch = self.dense_layers(embedded_batch)
        assert tuple(encoded_batch.shape) == (batch_size, self.final_hidden_size*self.embedding_size)

        prediction = self.prediction_layers(encoded_batch)
        assert tuple(prediction.shape) == (batch_size, self.output_size)
        
        return prediction

###############
# Classifiers #
###############

class Classifier(ABC):
    def __init__(self, output_directory: str, number_of_epochs: int, batch_size: int, train_portion: float, validation_portion: float, testing_portion: float, max_vocab_size: int, pre_trained_embedding_specification: str, **kwargs):
        super().__init__()
        self.best_valid_loss = float('inf')
        
        self.model_args = kwargs
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        self.pre_trained_embedding_specification = pre_trained_embedding_specification
        
        self.train_portion = train_portion
        self.validation_portion = validation_portion
        self.testing_portion = testing_portion
        
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        self.load_data()
        self.reset_f1_threshold()
        self.initialize_model()
        
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
        self.balance_training_data()
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
        return
    
    def balance_training_data(self) -> None:
        example_numericalized_labels = lambda example: torch.tensor([bool(getattr(example, topic)) for topic in self.topics], dtype=int)
        with timer("Dataset balancing"):
            self.training_data = balance_dataset_wrt_chi_squared_test(self.training_data, example_numericalized_labels)
        return
    
    def determine_training_unknown_words(self) -> None:
        pretrained_embedding_vectors = torchtext.vocab.pretrained_aliases[self.pre_trained_embedding_specification]()
        pretrained_embedding_vectors_unk_default_tensor = pretrained_embedding_vectors.unk_init(torch.Tensor(pretrained_embedding_vectors.dim))
        is_unk_token = lambda token: torch.all(pretrained_embedding_vectors[token] == pretrained_embedding_vectors_unk_default_tensor)
        tokens = reduce(set.union, (set(map(str,example.text)) for example in self.training_data))
        self.training_unk_words = set(eager_filter(is_unk_token, tokens))
        return
    
    @abstractmethod
    def initialize_model(self) -> None:
        pass
    
    def augment_training_data_sample(self, text_batch: torch.Tensor) -> torch.Tensor:
        non_padding_mask = (text_batch != self.pad_idx).to(text_batch.device)
        unk_mask = non_padding_mask & (torch.rand(text_batch.shape) < PORTION_OF_WORDS_TO_CROP_TO_UNK_FOR_DATA_AUGMENTATION).to(text_batch.device)
        text_batch = text_batch * (~unk_mask).int() + unk_mask * self.unk_idx
        return text_batch
    
    def train_one_epoch(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_f1 = 0
        epoch_recall = 0
        epoch_precision = 0
        number_of_training_batches = len(self.training_iterator)
        self.model.train()
        for (text_batch, text_lengths), multiclass_labels in tqdm_with_message(self.training_iterator, post_yield_message_func = lambda index: f'Training F1 {epoch_f1/(index+1):.8f}', total=number_of_training_batches, bar_format='{l_bar}{bar:50}{r_bar}'):
            text_batch = self.augment_training_data_sample(text_batch)
            self.optimizer.zero_grad()
            predictions = self.model(text_batch, text_lengths)
            loss = self.loss_function(predictions, multiclass_labels)
            f1, recall, precision = self.scores_of_discretized_values(predictions, multiclass_labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_f1 += f1
            epoch_recall += recall
            epoch_precision += precision
        epoch_loss /= number_of_training_batches
        epoch_f1 /= number_of_training_batches
        epoch_recall /= number_of_training_batches
        epoch_precision /= number_of_training_batches
        return epoch_loss, epoch_f1, epoch_recall, epoch_precision
    
    def evaluate(self, iterator, iterator_is_test_set) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_f1 = 0
        epoch_recall = 0
        epoch_precision = 0
        self.model.eval()
        self.optimize_f1_threshold()
        iterator_size = len(iterator)
        with torch.no_grad():
            for (text_batch, text_lengths), multiclass_labels in tqdm_with_message(iterator, post_yield_message_func = lambda index: f'{"Testing" if iterator_is_test_set else "Validation"} F1 {epoch_f1/(index+1):.8f}', total=iterator_size, bar_format='{l_bar}{bar:50}{r_bar}'):
                predictions = self.model(text_batch, text_lengths).squeeze(1)
                loss = self.loss_function(predictions, multiclass_labels)
                f1, recall, precision = self.scores_of_discretized_values(predictions, multiclass_labels)
                epoch_loss += loss.item()
                epoch_f1 += f1
                epoch_recall += recall
                epoch_precision += precision
        self.reset_f1_threshold()
        epoch_loss /= iterator_size
        epoch_f1 /= iterator_size
        epoch_recall /= iterator_size
        epoch_precision /= iterator_size
        return epoch_loss, epoch_f1, epoch_recall, epoch_precision
    
    def validate(self) -> Tuple[float, float]:
        return self.evaluate(self.validation_iterator, False)
    
    def test(self, epoch_index: int, result_is_from_final_run: bool) -> None:
        test_loss, test_f1, test_recall, test_precision = self.evaluate(self.testing_iterator, True)
        print(f'\t  Test F1: {test_f1:.8f} |  Test Recall: {test_recall:.8f} |  Test Precision: {test_precision:.8f} |  Test Loss: {test_loss:.8f}')
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
            'output_size': self.output_size,
            'train_portion': self.train_portion,
            'validation_portion': self.validation_portion,
            'testing_portion': self.testing_portion,
            'number_of_parameters': self.count_parameters(),
            'test_f1': test_f1,
            'test_loss': test_loss,
            'test_recall': test_recall,
            'test_precision': test_precision,
        }
        self_score_dict.update({(key, value.__name__ if hasattr(value, '__name__') else str(value)) for key, value in self.model_args.items()})
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
            print("\n")
            print(f"Epoch {epoch_index}")
            with timer(section_name=f"Epoch {epoch_index}"):
                train_loss, train_f1, train_recall, train_precision = self.train_one_epoch()
                valid_loss, valid_f1, valid_recall, valid_precision = self.validate()
                print(f'\t Train F1: {train_f1:.8f} | Train Recall: {train_recall:.8f} | Train Precision: {train_precision:.8f} | Train Loss: {train_loss:.8f}')
                print(f'\t  Val. F1: {valid_f1:.8f} |  Val. Recall: {valid_recall:.8f} |  Val. Precision: {valid_precision:.8f} |  Val. Loss: {valid_loss:.8f}')
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_parameters(best_saved_model_location)
                    self.test(epoch_index, False)
            print("\n")
            if reduce(bool.__or__, (valid_f1 > previous_f1 for previous_f1 in most_recent_validation_f1_scores)):
                most_recent_validation_f1_scores.pop(0)
                most_recent_validation_f1_scores.append(valid_f1)
            else:
                print()
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
        print(f'        output_size: {self.output_size}')
        print(f'        output_directory: {self.output_directory}')
        for model_arg_name, model_arg_value in sorted(self.model_args.items()):
            print(f'        {model_arg_name}: {model_arg_value.__name__ if hasattr(model_arg_value, "__name__") else str(model_arg_value)}')
        print()
        print(f'The model has {self.count_parameters()} trainable parameters.')
        print(f"This processes's PID is {os.getpid()}.")
        print()
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def optimize_f1_threshold(self) -> None:
        self.model.eval()
        with torch.no_grad():
            number_of_training_batches = len(self.training_iterator)
            training_sum_of_positives = torch.zeros(self.output_size).to(DEVICE)
            training_sum_of_negatives = torch.zeros(self.output_size).to(DEVICE)
            training_count_of_positives = torch.zeros(self.output_size).to(DEVICE)
            training_count_of_negatives = torch.zeros(self.output_size).to(DEVICE)
            for (text_batch, text_lengths), multiclass_labels in tqdm_with_message(self.training_iterator, post_yield_message_func = lambda index: f'Optimizing F1 Threshold', total=number_of_training_batches, bar_format='{l_bar}{bar:50}{r_bar}'):
                predictions = self.model(text_batch, text_lengths)
                if __debug__:
                    batch_size = len(text_lengths)
                assert tuple(predictions.data.shape) == (batch_size, self.output_size)
                assert tuple(multiclass_labels.shape) == (batch_size, self.output_size)
                
                training_sum_of_positives += (predictions.data * multiclass_labels).sum(dim=0)
                training_count_of_positives += multiclass_labels.sum(dim=0)
                
                training_sum_of_negatives += (predictions.data * (1-multiclass_labels)).sum(dim=0)
                training_count_of_negatives += (1-multiclass_labels).sum(dim=0)
                
                assert tuple(training_sum_of_positives.shape) == (self.output_size,)
                assert tuple(training_count_of_positives.shape) == (self.output_size,)
                assert tuple(training_sum_of_negatives.shape) == (self.output_size,)
                assert tuple(training_count_of_negatives.shape) == (self.output_size,)
                assert not tensor_has_nan(training_sum_of_positives)
                assert not tensor_has_nan(training_count_of_positives)
                assert not tensor_has_nan(training_sum_of_negatives)
                assert not tensor_has_nan(training_count_of_negatives)
                
            assert (0 != training_sum_of_positives).all()
            assert (0 != training_count_of_positives).all()
            assert (0 != training_sum_of_negatives).all()
            assert (0 != training_count_of_negatives).all()
            
            training_mean_of_positives = training_sum_of_positives / training_count_of_positives
            training_mean_of_negatives = training_sum_of_negatives / training_count_of_negatives
            assert tuple(training_mean_of_positives.shape) == (self.output_size,)
            assert tuple(training_mean_of_negatives.shape) == (self.output_size,)
            assert not tensor_has_nan(training_sum_of_positives)
            assert not tensor_has_nan(training_sum_of_negatives)
            self.f1_threshold = (training_mean_of_positives + training_mean_of_negatives) / 2.0
            assert not tensor_has_nan(self.f1_threshold)
        return
    
    def reset_f1_threshold(self) -> None:
        self.f1_threshold = torch.ones(self.output_size, dtype=float).to(DEVICE)*0.5
        return 
    
    def scores_of_discretized_values(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        batch_size = y.shape[0]
        assert batch_size <= self.batch_size
        assert tuple(y.shape) == (batch_size, self.output_size)
        assert tuple(y_hat.shape) == (batch_size, self.output_size)
        y_hat_discretized = (y_hat > self.f1_threshold).int()
        true_positive_count = ((y_hat_discretized == y) & y.bool()).float().sum(dim=0)
        false_positive_count = ((y_hat_discretized != y) & ~y.bool()).float().sum(dim=0)
        false_negative_count = ((y_hat_discretized != y) & y.bool()).float().sum(dim=0)
        assert tuple(true_positive_count.shape) == (self.output_size,)
        assert tuple(false_positive_count.shape) == (self.output_size,)
        assert tuple(false_negative_count.shape) == (self.output_size,)
        recall = _safe_count_tensor_division(true_positive_count , true_positive_count + false_negative_count)
        precision = _safe_count_tensor_division(true_positive_count , true_positive_count + false_positive_count)
        assert tuple(recall.shape) == (self.output_size,)
        assert tuple(precision.shape) == (self.output_size,)
        f1 = _safe_count_tensor_division(2 * precision * recall , precision + recall)
        assert tuple(f1.shape) == (self.output_size,)
        mean_f1 = torch.mean(f1).item()
        mean_recall = torch.mean(recall).item()
        mean_precision = torch.mean(precision).item()
        assert isinstance(mean_f1, float)
        assert isinstance(mean_recall, float)
        assert isinstance(mean_precision, float)
        assert mean_f1 == 0.0 or 0.0 not in (mean_recall, mean_precision)
        return mean_f1, mean_recall, mean_precision
    
    def save_parameters(self, parameter_file_location: str) -> None:
        torch.save(self.model.state_dict(), parameter_file_location)
        return
    
    def load_parameters(self, parameter_file_location: str) -> None:
        self.model.load_state_dict(torch.load(parameter_file_location))
        return
    
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

class EEAPClassifier(Classifier):
    def initialize_model(self) -> None:
        self.encoding_hidden_size = self.model_args['encoding_hidden_size']
        self.number_of_encoding_layers = self.model_args['number_of_encoding_layers']
        self.attention_intermediate_size = self.model_args['attention_intermediate_size']
        self.number_of_attention_heads = self.model_args['number_of_attention_heads']
        self.dropout_probability = self.model_args['dropout_probability']
        vocab_size = len(self.text_field.vocab)
        self.model = EEAPNetwork(vocab_size, self.embedding_size, self.encoding_hidden_size, self.number_of_encoding_layers, self.attention_intermediate_size, self.number_of_attention_heads, self.output_size, self.dropout_probability, self.pad_idx, self.unk_idx, self.text_field.vocab.vectors)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.BCELoss().to(DEVICE)
        return

class ConvClassifier(Classifier):
    def initialize_model(self) -> None:
        self.convolution_hidden_size = self.model_args['convolution_hidden_size']
        self.kernel_sizes = self.model_args['kernel_sizes']
        self.pooling_method = self.model_args['pooling_method']
        self.dropout_probability = self.model_args['dropout_probability']
        vocab_size = len(self.text_field.vocab)
        self.model = ConvNetwork(vocab_size, self.embedding_size, self.convolution_hidden_size, self.kernel_sizes, self.pooling_method, self.output_size, self.dropout_probability, self.pad_idx, self.unk_idx, self.text_field.vocab.vectors)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.BCELoss().to(DEVICE)
        return

class DenseClassifier(Classifier):
    def initialize_model(self) -> None:
        self.dense_hidden_sizes = self.model_args['dense_hidden_sizes']
        self.dropout_probability = self.model_args['dropout_probability']
        vocab_size = len(self.text_field.vocab)
        self.model = DenseNetwork(vocab_size, self.embedding_size, self.dense_hidden_sizes, self.output_size, self.dropout_probability, self.pad_idx, self.unk_idx, self.text_field.vocab.vectors)
        self.optimizer = optim.Adam(self.model.parameters())
        BCE_loss_function = nn.BCELoss().to(DEVICE)
        self.loss_function = lambda y_hat, y: BCE_loss_function(y_hat, y) + soft_f1_loss(y_hat, y)
        return

###############
# Main Driver #
###############

if __name__ == '__main__':
    print() # @todo fill this in
