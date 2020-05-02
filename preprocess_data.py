#!/usr/bin/python3 -OO

"""
This file contains SGML data pre-processing utilities for documents that appeared on Reuters newswire in 1987.

The data can be found at http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html

Sections:
* Imports
* Globals
* Shorthand with Special Characters & Contraction Expansion
* General String Preprocessing Utilities
* File Preprocessing Utilities
* Driver
"""

###########
# Imports #
###########

import os
import bs4
import re
import itertools
import nltk
import pandas as pd
from typing import Iterable, Tuple
from nltk.corpus.reader.wordnet import ADJ, ADJ_SAT, ADV, NOUN, VERB
from misc_utilites import eager_map, at_most_one, parallel_map, timer

###########
# Globals #
###########

DATA_DIRECTORY = "./data/"
PREPROCESSED_DATA_DIR = './preprocessed_data/'
ALL_DATA_OUTPUT_CSV_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'all_extracted_data.csv')
TOPICS_DATA_OUTPUT_CSV_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'topics_data.csv')

NON_TOPIC_COLUMNS_RELEVANT_TO_TOPICS_DATA = {'date', 'text_dateline', 'text_title', 'raw_text', 'text', 'file', 'reuter_element_position'}
MINIMUM_NUMBER_OF_SAMPLES_FOR_TOPIC = 200
NUMBER_TOKEN = "NUMBER"
STOPWORDS = nltk.corpus.stopwords.words('english')
PARTS_OF_SPEECH = {ADJ, ADJ_SAT, ADV, NOUN, VERB}
LEMMATIZER = nltk.stem.WordNetLemmatizer()

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

def replace_digits_with_special_token(input_string: str) -> str:
    output_string = input_string
    output_string = re.sub(r'[0-9]+', f' {NUMBER_TOKEN} ', output_string)
    assert 10 == sum(map(int, (digit not in output_string for digit in '1234567890')))
    return output_string

def remove_white_space_characters(input_string: str) -> str:
    output_string = input_string
    output_string = pervasively_replace(output_string, '\t', ' ')
    output_string = pervasively_replace(output_string, '\n', ' ')
    output_string = pervasively_replace(output_string, '  ',' ')
    output_string = output_string.strip()
    return output_string

def remove_stop_words(input_string: str) -> str:
    output_string = input_string
    output_string = ' '.join(filter(lambda word: word not in STOPWORDS, output_string.split(' ')))
    output_string = remove_white_space_characters(output_string) 
    return output_string

LEMMATIZED_WORD_SPECIAL_CASE_MAPPING = {
    # Already stemmed
    'cts': 'cts',
    'vs': 'vs',
    # Unclear cases
    'cos': 'cos',
    'tates': 'tates',
    'barings': 'barings',
    'reeves': 'reeves',
    # Misc. cases
    'less': 'less',
    'wages': 'wage',
    'uses': 'use',
    'rates': 'rates',
    'hopes': 'hope',
    'stages': 'stage',
    'dies': 'die',
    'possesses': 'possess',
    'proves': 'prove',
    'codes': 'code',
    'primes': 'prime',
    'spares': 'spare',
    'shillings': 'shilling',
    'leaves': 'leave',
    'planes': 'plane',
    'sages': 'sage',
    'reverses': 'reverse',
    'plates': 'plate',
    'spines': 'spine',
    'fines': 'fine',
    'routes': 'route',
    'wines': 'wine',
    'tapes': 'tape',
    'sites': 'site',
    'discusses': 'discuss',
    'slopes': 'slope',
    'matings': 'mating',
    'spares': 'spare',
}

def lemmatize_word(word: str) -> str:
    if word in LEMMATIZED_WORD_SPECIAL_CASE_MAPPING:
        lemmatized_word = LEMMATIZED_WORD_SPECIAL_CASE_MAPPING[word]
    else: 
        lemmatized_word = min({LEMMATIZER.lemmatize(word,pos) for pos in PARTS_OF_SPEECH}, key=len)
    return lemmatized_word

def lemmatize_words(input_string: str) -> str:
    output_string = input_string
    output_string = ' '.join(map(lemmatize_word, output_string.split(' ')))
    output_string = remove_white_space_characters(output_string) 
    return output_string

def dwim_weird_characters(input_string: str) -> str:
    output_string = input_string
    output_string = pervasively_replace(output_string, chr(3),'')
    output_string = pervasively_replace(output_string, chr(30),'')
    for match in re.finditer(r'\b\w*"s\b', output_string): # "s -> 's 
        match_string = match.group()
        output_string = output_string.replace(match_string, match_string.replace('"s', "'s"))
    return output_string

def preprocess_article_text(input_string: str) -> str:
    output_string = input_string
    output_string = output_string.lower()
    output_string = dwim_weird_characters(output_string)
    output_string = pervasively_replace(output_string, '....','...')
    output_string = replace_digits_with_special_token(output_string)
    output_string = expand_contractions_and_shorthand_words_with_special_characters(output_string)
    output_string = remove_white_space_characters(output_string)
    output_string = remove_stop_words(output_string)
    output_string = lemmatize_words(output_string)
    return output_string

################################
# File Preprocessing Utilities #
################################

def delete_topics_with_insufficient_data(topics_df: pd.DataFrame) -> pd.DataFrame:
    all_topics = set(topics_df.columns)-NON_TOPIC_COLUMNS_RELEVANT_TO_TOPICS_DATA
    columns_with_insufficient_samples = [column_name for column_name, number_of_samples in topics_df[all_topics].sum().iteritems() if number_of_samples < MINIMUM_NUMBER_OF_SAMPLES_FOR_TOPIC]
    topics_df.drop(columns_with_insufficient_samples, axis=1, inplace=True)
    updated_topics = set(topics_df.columns)-NON_TOPIC_COLUMNS_RELEVANT_TO_TOPICS_DATA
    topics_df = topics_df[topics_df[updated_topics].sum(axis=1)>0]
    return topics_df

def gather_sgm_files() -> Iterable[str]:
    all_data_entries = os.listdir('./data/')
    sgm_files = map(lambda sgm_file_name: os.path.join(DATA_DIRECTORY, sgm_file_name), filter(lambda entry: '.' in entry and entry.split('.')[-1]=='sgm', all_data_entries))
    return sgm_files

def extract_csv_rows_from_sgm_file(sgm_file: str) -> Tuple[dict, dict]:
    all_rows = []
    topics_rows = []
    with open(sgm_file, 'rb') as sgm_text:
        soup = bs4.BeautifulSoup(sgm_text,'html.parser')
        reuters_elements = soup.find_all('reuters')
        for row_index, reuters_element in enumerate(reuters_elements):
            text_element = at_most_one(reuters_element.find_all('text'))
            text_element_title = at_most_one(text_element.find_all('title'))
            text_element_dateline = at_most_one(text_element.find_all('dateline'))
            text_element_body = at_most_one(text_element.find_all('body'))
            text_element_body_text = text_element_body.text if text_element_body else None
            if not text_element_body_text or len(text_element_body_text)==0:
                continue
            preprocessed_text_element_body_text = preprocess_article_text(text_element_body_text)
            if len(preprocessed_text_element_body_text) < 10:
                continue
            date_element = at_most_one(reuters_element.find_all('date'))
            topics_element = at_most_one(reuters_element.find_all('topics'))
            topic_elements = topics_element.find_all('d')
            topics: List[str] = eager_map(bs4.element.Tag.get_text, topic_elements)
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
                'places': eager_map(bs4.element.Tag.get_text, place_elements),
                'people': eager_map(bs4.element.Tag.get_text, person_elements),
                'orgs': eager_map(bs4.element.Tag.get_text, org_elements),
                'exchanges': eager_map(bs4.element.Tag.get_text, exchange_elements),
                'companies': eager_map(bs4.element.Tag.get_text, company_elements),
                'unknown': eager_map(bs4.element.Tag.get_text, unknown_elements),
                'text_title': text_element_title.text if text_element_title else None,
                'text_dateline': text_element_dateline.text if text_element_dateline else None,
                'raw_text': text_element_body_text,
                'text': preprocessed_text_element_body_text,
                'file': sgm_file,
                'reuter_element_position': row_index,
            }
            all_rows.append(all_data_row)
            
            if len(topics) > 0:
                topic_row = {column_name:all_data_row[column_name] for column_name in NON_TOPIC_COLUMNS_RELEVANT_TO_TOPICS_DATA}
                topic_row.update({topic: True for topic in topics})
                topics_rows.append(topic_row)
    return all_rows, topics_rows

def parse_sgm_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    with timer(section_name="Parsing of .sgm files"):
        print("Parsing .sgm files.")
        all_rows_pieces, topics_rows_pieces = zip(*parallel_map(extract_csv_rows_from_sgm_file, gather_sgm_files()))
        print("Parsing of .sgm files complete.")
    all_rows = itertools.chain(*all_rows_pieces)
    topics_rows = itertools.chain(*topics_rows_pieces)
    all_df = pd.DataFrame(all_rows)
    topics_df = pd.DataFrame(topics_rows)
    topics_df = delete_topics_with_insufficient_data(topics_df)
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
    print(f'{TOPICS_DATA_OUTPUT_CSV_FILE} has {len(set(topics_df.columns)-NON_TOPIC_COLUMNS_RELEVANT_TO_TOPICS_DATA)} topics.')
    print(f'{TOPICS_DATA_OUTPUT_CSV_FILE} has {len(topics_df)} rows.')
    print(f'{TOPICS_DATA_OUTPUT_CSV_FILE} has {len(topics_df.columns)} columns.')
    print()
    return

##########
# Driver #
##########

if __name__ == '__main__':
    print("This file contains SGML data pre-processing utilities.")
