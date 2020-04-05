#!/usr/bin/python3
"#!/usr/bin/python3 -OO" # @todo make this the default

"""
This file contains the functionality for the main interface to several NLP processes on documents that appeared on Reuters newswire in 1987.

The data can be found at http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html

Sections:
* Imports
* Functionality
* Driver
"""

###########
# Imports #
###########

import argparse
from misc_utilites import debug_on_error, eager_map, at_most_one, tqdm_with_message

#################
# Functionality #
#################

def train_model() -> None:
    from models import EEPClassifier
    classifier = EEPClassifier()
    classifier.train()
    return

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-preprocess-data', action='store_true', help="Preprocess the raw SGML files into a CSV.")
    parser.add_argument('-train-model', action='store_true', help="Trains & evaluates our model on our dataset. Saves model to ./best-model.pt.")
    # @todo load model and evaluate texts. Make this an interactive buffer.
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,vars(args).values()))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print("Please specify exactly one action.")
    elif args.preprocess_data:
        import preprocess_data
        preprocess_data.preprocess_data()
    elif args.train_model:
        train_model()
    else:
        raise Exception("Unexpected args received.")
    return

if __name__ == '__main__':
    main()
