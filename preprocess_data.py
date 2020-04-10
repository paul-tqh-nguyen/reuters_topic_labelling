#!/usr/bin/python3 -OO

"""
This file contains SGML data pre-processing utilities for documents that appeared on Reuters newswire in 1987.

The data can be found at http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html

Sections:
* Imports
* Globals
* Preprocessing Utilities
* Driver
"""

# @todo update the sections outline

###########
# Imports #
###########

import os
import bs4
import re
import pandas as pd
from typing import Iterable, Tuple
from misc_utilites import debug_on_error, eager_map, at_most_one, tqdm_with_message

###########
# Globals #
###########

DATA_DIRECTORY = "./data/"
PREPROCESSED_DATA_DIR = './preprocessed_data/'
ALL_DATA_OUTPUT_CSV_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'all_extracted_data.csv')
TOPICS_DATA_OUTPUT_CSV_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'topics_data.csv')

COLUMNS_RELEVANT_TO_TOPICS_DATA = {'date', 'text_dateline', 'text_title', 'text', 'file', 'reuter_element_position'}

#############################################################
# Shorthand with Special Characters & Contraction Expansion #
#############################################################

CONTRACTION_EXPANSION_MAP = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}

CONTRACTION_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST = sorted(CONTRACTION_EXPANSION_MAP.items(), key=lambda x: len(x[0]), reverse=True)

SHORTHAND_WITH_SPECIAL_CHARACTERS_EXPANSION_MAP = {
    "w/": "with",
    "w/o": "without",
}

SHORTHAND_WITH_SPECIAL_CHARACTERS_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST = sorted(SHORTHAND_WITH_SPECIAL_CHARACTERS_EXPANSION_MAP.items(), key=lambda x: len(x[0]), reverse=True)

def expand_contractions_and_shorthand_words_with_special_characters(text_string: str) -> str:
    updated_text_string = text_string
    for contraction, expansion in CONTRACTION_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST:
        updated_text_string = re.sub(r"\b"+contraction+r"\b", expansion, updated_text_string, 0, re.IGNORECASE)
        updated_text_string = re.sub(r"\b"+contraction.replace("'", "")+r"\b", expansion, updated_text_string, 0, re.IGNORECASE)
    for shorthand, expansion in SHORTHAND_WITH_SPECIAL_CHARACTERS_EXPANSION_PAIRS_SORTED_BIGGEST_FIRST:
        updated_text_string = ' '.join([expansion if word.lower() == shorthand else word for word in updated_text_string.split()])
    return updated_text_string

##########################################
# General String Preprocessing Utilities #
##########################################

def pervasively_replace(input_string: str, old: str, new: str) -> str:
    while old in input_string:
        input_string = input_string.replace(old, new)
    return input_string

def expand_digits(input_string: str) -> str:
    output_string = input_string
    for numeric_character in '0123456789':
        output_string = output_string.replace(numeric_character, ' '+numeric_character+' ')
    for match in re.finditer(r" -[A-Za-z]+", output_string):
        match_string = match.group()
        output_string = output_string.replace(match_string, ' '+match_string[2:])
    return output_string

def remove_white_space_characters(input_string: str) -> str:
    output_string = input_string
    output_string = pervasively_replace(output_string, '\t', ' ')
    output_string = pervasively_replace(output_string, '\n', ' ')
    output_string = pervasively_replace(output_string, '  ',' ')
    output_string = output_string.strip()
    return output_string

def dwim_weird_characters(input_string: str) -> str:
    output_string = input_string
    output_string = pervasively_replace(output_string, chr(3),'')
    output_string = pervasively_replace(output_string, chr(30),'')
    for match in re.finditer(r'\b\w*"s\b', output_string): # "s -> 's 
        match_string = match.group()
        output_string = output_string.replace(match_string, match_string.replace('"s', "'s"))
    return output_string

def preprocess_text_element_body_text(input_string: str) -> str:
    output_string = input_string
    output_string = output_string.lower()
    output_string = pervasively_replace(output_string, '....','...') # @todo do we want to preprocess these more intelligently or have the model learn it?
    output_string = expand_digits(output_string)
    output_string = expand_contractions_and_shorthand_words_with_special_characters(output_string)
    output_string = remove_white_space_characters(output_string)
    output_string = dwim_weird_characters(output_string)
    return output_string

################################
# File Preprocessing Utilities #
################################

def gather_sgm_files() -> Iterable[str]:
    all_data_entries = os.listdir('./data/')
    sgm_files = map(lambda sgm_file_name: os.path.join(DATA_DIRECTORY, sgm_file_name), filter(lambda entry: '.' in entry and entry.split('.')[-1]=='sgm', all_data_entries))
    return sgm_files

def parse_sgm_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: List[dict] = []
    topics_rows: List[dict] = []
    for sgm_file in gather_sgm_files(): # @todo parallelize this
        with open(sgm_file, 'rb') as sgm_text:
            soup = bs4.BeautifulSoup(sgm_text,'html.parser')
            reuters_elements = soup.find_all('reuters')
            for row_index, reuters_element in enumerate(tqdm_with_message(reuters_elements, pre_yield_message_func=lambda index: f'Processing {sgm_file}', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')):
                get_element_text = lambda element: element.text
                text_element = at_most_one(reuters_element.find_all('text'))
                text_element_title = at_most_one(text_element.find_all('title'))
                text_element_dateline = at_most_one(text_element.find_all('dateline'))
                text_element_body = at_most_one(text_element.find_all('body'))
                text_element_body_text = preprocess_text_element_body_text(text_element_body.text) if text_element_body else None
                if not text_element_body_text or len(text_element_body_text)==0:
                    continue
                date_element = at_most_one(reuters_element.find_all('date'))
                topics_element = at_most_one(reuters_element.find_all('topics'))
                topic_elements = topics_element.find_all('d')
                topics: List[str] = eager_map(get_element_text, topic_elements)
                places_element = at_most_one(reuters_element.find_all('places'))
                place_elements = places_element.find_all('d')
                people_element = at_most_one(reuters_element.find_all('people'))
                person_elements = people_element.find_all('d')
                orgs_element = at_most_one(reuters_element.find_all('orgs'))
                org_elements = orgs_element.find_all('d')
                exchanges_element = at_most_one(reuters_element.find_all('exchanges'))
                exchange_elements = exchanges_element.find_all('d')
                companies_element = at_most_one(reuters_element.find_all('companies'))
                company_elements = companies_element.find_all('d')
                unknown_elements = reuters_element.find_all('unknown')
                
                all_data_row = {
                    'date': date_element.text.strip(),
                    'topics_raw_string': topics,
                    'places': eager_map(get_element_text, place_elements),
                    'people': eager_map(get_element_text, person_elements),
                    'orgs': eager_map(get_element_text, org_elements),
                    'exchanges': eager_map(get_element_text, exchange_elements),
                    'companies': eager_map(get_element_text, company_elements),
                    'unknown': eager_map(get_element_text, unknown_elements),
                    'text_title': text_element_title.text if text_element_title else None,
                    'text_dateline': text_element_dateline.text if text_element_dateline else None,
                    'text': text_element_body_text,
                    'file': sgm_file,
                    'reuter_element_position': row_index,
                }
                all_rows.append(all_data_row)
                
                if len(topics) > 0:
                    topic_row = {column_name:all_data_row[column_name] for column_name in COLUMNS_RELEVANT_TO_TOPICS_DATA}
                    topic_row.update({topic: True for topic in topics})
                    topics_rows.append(topic_row)
                    
    all_df = pd.DataFrame(all_rows)
    topics_df = pd.DataFrame(topics_rows)
    return all_df, topics_df

def preprocess_data() -> None:
    if not os.path.isdir(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)
    all_df, topics_df = parse_sgm_files()
    all_df.to_csv(ALL_DATA_OUTPUT_CSV_FILE, index=False)
    topics_df.to_csv(TOPICS_DATA_OUTPUT_CSV_FILE, index=False)
    print()
    print(f'Preprocessing of entire dataset is in {ALL_DATA_OUTPUT_CSV_FILE}')
    print(f'{ALL_DATA_OUTPUT_CSV_FILE} has {len(all_df)} rows.')
    print(f'{ALL_DATA_OUTPUT_CSV_FILE} has {len(all_df.columns)} columns.')
    print()
    print(f'Preprocessing of topics is in {TOPICS_DATA_OUTPUT_CSV_FILE}')
    print(f'{TOPICS_DATA_OUTPUT_CSV_FILE} has {len(set(topics_df.columns)-COLUMNS_RELEVANT_TO_TOPICS_DATA)} topics.')
    print(f'{TOPICS_DATA_OUTPUT_CSV_FILE} has {len(topics_df)} rows.')
    print(f'{TOPICS_DATA_OUTPUT_CSV_FILE} has {len(topics_df.columns)} columns.')
    print()
    return

##########
# Driver #
##########

if __name__ == '__main__':
    print("This file contains SGML data pre-processing utilities.")
