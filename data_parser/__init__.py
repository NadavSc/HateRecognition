import os
import json
import pandas as pd

from os.path import dirname, abspath

project_dir = dirname(dirname(abspath(__file__)))
data_dir = os.path.join(project_dir, 'data')
annotated_data_path = os.path.join(data_dir, 'parler_annotated_data.csv')
wikis = {'LGBT slang': {'type': 'list', 'link': 'https://en.wikipedia.org/wiki/LGBT_slang'},
         'List of disability-related terms with negative connotations': {'type': 'list', 'link': 'https://en.wikipedia.org/wiki/List_of_disability-related_terms_with_negative_connotations'},
         'List of ethnic slurs': {'type': 'table', 'link': 'https://en.wikipedia.org/wiki/List_of_ethnic_slurs'}}


def json_write(file, name):
    with open(os.path.join(data_dir, f'{name}.json'), 'w') as f:
        json.dump(file, f)


def json_load(path):
    with open(path, 'r') as f:
        return json.load(f)


# json_write(file=wikis, name='lexical_resources')
# wikis = json_load(os.path.join(data_dir, 'lexical_resources.json'))

db = pd.read_csv(annotated_data_path)
n_dispute_post = sum(db['disputable_post'])
n_standard_post = len(db) - n_dispute_post
