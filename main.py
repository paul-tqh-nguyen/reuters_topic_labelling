#!/usr/bin/python3 -O
'#!/usr/bin/python3 -OO' # @ todo this causes errors when it should not; reported via https://github.com/pytorch/text/issues/752 ; update this when bug is fixed

'''
This file contains the functionality for the main interface to several NLP processes on documents that appeared on Reuters newswire in 1987.

The data can be found at http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html

Sections:
* Imports
* Functionality
* Driver
'''

###########
# Imports #
###########

import argparse
import random
import os
import itertools
from misc_utilities import tqdm_with_message, safe_cuda_memory

#################
# Functionality #
#################

from torch.nn.functional import max_pool1d, avg_pool1d, adaptive_avg_pool1d, adaptive_max_pool1d, lp_pool1d

NUMBER_OF_EPOCHS = 100
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 25_000
TRAIN_PORTION, VALIDATION_PORTION, TESTING_PORTION = (0.50, 0.20, 0.3)

PRE_TRAINED_EMBEDDING_SPECIFICATION = 'fasttext.en.300d'
CONVOLUTION_HIDDEN_SIZE = 256
KERNEL_SIZES = [3,4,5,6]
POOLING_METHOD = max_pool1d
DROPOUT_PROBABILITY = 0.5

OUTPUT_DIR = './default_output/'

def train_model() -> None:
    from models import ConvClassifier
    classifier = ConvClassifier(OUTPUT_DIR, NUMBER_OF_EPOCHS, BATCH_SIZE, TRAIN_PORTION, VALIDATION_PORTION, TESTING_PORTION, MAX_VOCAB_SIZE, PRE_TRAINED_EMBEDDING_SPECIFICATION,
                                convolution_hidden_size=CONVOLUTION_HIDDEN_SIZE,
                                kernel_sizes=KERNEL_SIZES,
                                pooling_method=POOLING_METHOD,
                                dropout_probability=DROPOUT_PROBABILITY)
    classifier.train()
    return

def hyperparameter_search() -> None:
    #hyperparameter_search_rnn()
    hyperparameter_search_conv()
    #hyperparameter_search_dense()
    return

def hyperparameter_search_dense() -> None:
    from models import DenseClassifier
    
    number_of_epochs = 100
    batch_size = 64
    train_portion, validation_portion, testing_portion = (0.50, 0.20, 0.3)
    
    max_vocab_size_choices = [10_000, 25_000, 50_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']
    
    dense_hidden_sizes_choices = [
        [128,64],
        [128,64,64],
        [128,64,64,32],
        [128,64,64,32,32],
        [128,64,64,32,32,16],
        [128,64,64,32,32,16,16],
        [128,64,64,32,32,16,16,16],
    ]
    dropout_probability_choices = [0.0, 0.25, 0.5]
    
    hyparameter_list_choices = list(itertools.product(max_vocab_size_choices,
                                                      pre_trained_embedding_specification_choices,
                                                      dense_hidden_sizes_choices,
                                                      dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (max_vocab_size, pre_trained_embedding_specification, dense_hidden_sizes, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/epochs_{number_of_epochs}_batch_size_{batch_size}_train_frac_{train_portion}_validation_frac_{validation_portion}_testing_frac_{testing_portion}_max_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_dense_hidden_sizes_{str(dense_hidden_sizes).replace(" ","")}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, 'final_model_score.json')
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            with safe_cuda_memory():
                classifier = DenseClassifier(output_directory,
                                             number_of_epochs,
                                             batch_size,
                                             train_portion,
                                             validation_portion,
                                             testing_portion,
                                             max_vocab_size,
                                             pre_trained_embedding_specification,
                                             dense_hidden_sizes=dense_hidden_sizes,
                                             dropout_probability=dropout_probability)
                classifier.train()
    return

def hyperparameter_search_conv() -> None:
    from models import ConvClassifier
    
    number_of_epochs = 100
    batch_size = 64
    train_portion, validation_portion, testing_portion = (0.50, 0.20, 0.3)
    
    max_vocab_size_choices = [10_000, 25_000, 50_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

    convolution_hidden_size_choices = [64, 128, 256, 512]
    pooling_method_choices = [
        max_pool1d,
        avg_pool1d,
        # adaptive_avg_pool1d,
        # adaptive_max_pool1d,
        # lp_pool1d,
    ]
    kernel_sizes_choices = [
        [2],
        [2,3],
        [2,3,4],
        [2,3,4,5],
        [2,3,4,5,6],
        [3,6],
    ]
    dropout_probability_choices = [0.0, 0.25, 0.5]
    
    hyparameter_list_choices = list(itertools.product(max_vocab_size_choices,
                                                      pre_trained_embedding_specification_choices,
                                                      convolution_hidden_size_choices,
                                                      pooling_method_choices,
                                                      kernel_sizes_choices,
                                                      dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (max_vocab_size, pre_trained_embedding_specification, convolution_hidden_size, pooling_method, kernel_sizes, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/epochs_{number_of_epochs}_batch_size_{batch_size}_train_frac_{train_portion}_validation_frac_{validation_portion}_testing_frac_{testing_portion}_max_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_conv_size_{convolution_hidden_size}_pool_{pooling_method.__name__ if hasattr(pooling_method, "__name__") else str(pooling_method)}_kernel_sizes_{str(kernel_sizes).replace(" ","")}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, 'final_model_score.json')
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            with safe_cuda_memory():
                classifier = ConvClassifier(output_directory,
                                            number_of_epochs,
                                            batch_size,
                                            train_portion,
                                            validation_portion,
                                            testing_portion,
                                            max_vocab_size,
                                            pre_trained_embedding_specification,
                                            convolution_hidden_size=convolution_hidden_size,
                                            pooling_method=pooling_method,
                                            kernel_sizes=kernel_sizes,
                                            dropout_probability=dropout_probability)
                classifier.train()
    return

def hyperparameter_search_rnn() -> None:
    from models import EEAPClassifier
    
    number_of_epochs = 40
    batch_size = 1
    train_portion, validation_portion, testing_portion = (0.50, 0.20, 0.3)
    
    max_vocab_size_choices = [10_000, 25_000, 50_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']
    encoding_hidden_size_choices = [128, 256, 512]
    number_of_encoding_layers_choices = [1, 2]
    attention_intermediate_size_choices = [4, 16, 32]
    number_of_attention_heads_choices = [1, 2, 4, 32]
    dropout_probability_choices = [0.0, 0.25, 0.5]

    hyparameter_list_choices = list(itertools.product(max_vocab_size_choices,
                                                      pre_trained_embedding_specification_choices,
                                                      encoding_hidden_size_choices,
                                                      number_of_encoding_layers_choices,
                                                      attention_intermediate_size_choices,
                                                      number_of_attention_heads_choices,
                                                      dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)
    for (max_vocab_size, pre_trained_embedding_specification, encoding_hidden_size, number_of_encoding_layers, attention_intermediate_size, number_of_attention_heads, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/epochs_{number_of_epochs}_batch_size_{batch_size}_train_frac_{train_portion}_validation_frac_{validation_portion}_testing_frac_{testing_portion}_max_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_encoding_size_{encoding_hidden_size}_numb_encoding_layers_{number_of_encoding_layers}_attn_intermediate_size_{attention_intermediate_size}_num_attn_heads_{number_of_attention_heads}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, 'final_model_score.json')
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            with safe_cuda_memory():
                classifier = EEAPClassifier(output_directory,
                                            number_of_epochs,
                                            batch_size,
                                            train_portion,
                                            validation_portion,
                                            testing_portion,
                                            max_vocab_size,
                                            pre_trained_embedding_specification,
                                            encoding_hidden_size=encoding_hidden_size,
                                            number_of_encoding_layers=number_of_encoding_layers,
                                            attention_intermediate_size=attention_intermediate_size,
                                            number_of_attention_heads=number_of_attention_heads,
                                            dropout_probability=dropout_probability)
                classifier.train()
    return

##########
# Driver #
##########

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 40))
    parser.add_argument('-preprocess-data', action='store_true', help='Preprocess the raw SGML files into a CSV.')
    parser.add_argument('-train-model', action='store_true', help='Trains & evaluates our model on our dataset. Saves model to ./default_output/best-model.pt.')
    parser.add_argument('-hyperparameter-search', action='store_true', help='Exhaustively performs -train-model over the hyperparameter space. Details of the best performance are tracked in global_best_model_score.json.')
    parser.add_argument('-hyperparameter-search-rnn', action='store_true', help='Perform the hyperparameter search on our RNN model.')
    parser.add_argument('-hyperparameter-search-conv', action='store_true', help='Perform the hyperparameter search on our CNN model.')
    parser.add_argument('-hyperparameter-search-dense', action='store_true', help='Perform the hyperparameter search on our simple feed-forward model.')
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,vars(args).values()))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print('Please specify exactly one action.')
    elif args.preprocess_data:
        import preprocess_data
        preprocess_data.preprocess_data()
    elif args.train_model:
        train_model()
    elif args.hyperparameter_search:
        hyperparameter_search()
    elif args.hyperparameter_search_rnn:
        hyperparameter_search_rnn()
    elif args.hyperparameter_search_conv:
        hyperparameter_search_conv()
    elif args.hyperparameter_search_dense:
        hyperparameter_search_dense()
    else:
        raise Exception('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
