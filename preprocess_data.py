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

###########
# Imports #
###########

import os
import pandas as pd
import bs4
from bs4 import BeautifulSoup
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

###########################
# Preprocessing Utilities #
###########################

def gather_sgm_files() -> Iterable[str]:
    all_data_entries = os.listdir('./data/')
    sgm_files = map(lambda sgm_file_name: os.path.join(DATA_DIRECTORY, sgm_file_name), filter(lambda entry: '.' in entry and entry.split('.')[-1]=='sgm', all_data_entries))
    return sgm_files

def get_element_text(element: bs4.element.Tag) -> str:
    return element.text

def parse_sgm_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: List[dict] = []
    topics_rows: List[dict] = []
    for sgm_file in gather_sgm_files(): # @todo parallelize this
        with open(sgm_file, 'rb') as sgm_text:
            soup = BeautifulSoup(sgm_text,'html.parser')
            reuters_elements = soup.find_all('reuters')
            for row_index, reuters_element in enumerate(tqdm_with_message(reuters_elements, pre_yield_message_func=lambda index: f'Processing {sgm_file}', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')):
                text_element = at_most_one(reuters_element.find_all('text'))
                text_element_title = at_most_one(text_element.find_all('title'))
                text_element_dateline = at_most_one(text_element.find_all('dateline'))
                text_element_body = at_most_one(text_element.find_all('body'))
                text_element_body_text = text_element_body.text.strip() if text_element_body else None
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
                    'text': text_element_body.text if text_element_body else None,
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
