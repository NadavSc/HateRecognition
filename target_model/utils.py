from collections import Counter
import pandas as pd
import json
import os.path


def define_target(data):
    new_data = {}
    for key, post in data.items():
        labels_counter = Counter([x['label'] for x in post['annotators']])
        target_counter = Counter([y for x in post['annotators']  for y in x['target']])
        new_data[key] = {'label': labels_counter.most_common(1)[0][0],
                     'target': target_counter.most_common(1)[0][0],
                     'tokens': post['post_tokens']}
    return new_data

def load_anno_data():
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, '../data/dataset.json')
    with open(path, 'r') as f:
        data = json.load(f)
    new_data = define_target(data)
    df = pd.DataFrame(new_data).T
    return df