#!/usr/bin/python3
'#!/usr/bin/python3 -OO' # @todo make this the default

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
from misc_utilites import debug_on_error, eager_map, at_most_one, tqdm_with_message

#################
# Functionality #
#################

from torch.nn.functional import max_pool1d, avg_pool1d, lp_pool1d, adaptive_avg_pool1d, adaptive_max_pool1d

NUMBER_OF_EPOCHS = 100
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 25_000
TRAIN_PORTION, VALIDATION_PORTION, TESTING_PORTION = (0.50, 0.20, 0.3)

PRE_TRAINED_EMBEDDING_SPECIFICATION = 'glove.840B.300d'
CONVOLUTION_HIDDEN_SIZE = 512
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
    from models import ConvClassifier
    
    number_of_epochs = 100
    batch_size = 64
    train_portion, validation_portion, testing_portion = (0.50, 0.20, 0.3)
    
    max_vocab_size_choices = [10_000, 25_000, 50_000]
    pre_trained_embedding_specification_choices = ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

    convolution_hidden_size_choices = [64, 128, 256, 512]
    pooling_method_choices = [max_pool1d, avg_pool1d, lp_pool1d, adaptive_avg_pool1d, adaptive_max_pool1d]
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
    random.shuffle(hyparameter_list_choices)
    for (max_vocab_size, pre_trained_embedding_specification, convolution_hidden_size, pooling_method, kernel_sizes, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/epochs_{number_of_epochs}_batch_size_{batch_size}_train_frac_{train_portion}_validation_frac_{validation_portion}_testing_frac_{testing_portion}_max_vocab_{max_vocab_size}_embed_spec_{pre_trained_embedding_specification}_conv_size_{convolution_hidden_size}_pool_{pooling_method.__name__ if hasattr(pooling_method, "__name__") else str(pooling_method)}_kernel_sizes_{str(kernel_sizes).replace(" ","")}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, 'final_model_score.json')
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
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

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 30))
    parser.add_argument('-preprocess-data', action='store_true', help='Preprocess the raw SGML files into a CSV.')
    parser.add_argument('-train-model', action='store_true', help='Trains & evaluates our model on our dataset. Saves model to ./best-model.pt.')
    parser.add_argument('-hyperparameter-search', action='store_true', help='Exhaustively performs -train-model over the hyperparameter space. Details of the best performance are tracked in global_best_model_score.json.')
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
    else:
        raise Exception('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
