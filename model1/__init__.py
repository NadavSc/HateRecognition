import os
import re
import torch
import pdb
import gensim
import numpy as np
import pandas as pd

from torch import nn
from transformers import BertModel, BertTokenizer
from os.path import abspath, dirname
from nltk import WordNetLemmatizer


PATH = dirname(dirname(abspath(__file__)))
DATA_DIR = os.path.join(PATH, 'data')
annotated_data_path = os.path.join(DATA_DIR, 'parler_annotated_data.csv')
annotated_labeled_data_path = os.path.join(DATA_DIR, 'parler_annotated_data_labeled.csv')
np.random.seed(42)


def lemmatize(token):
    return WordNetLemmatizer().lemmatize(token, pos='v')


def preprocess(tweet):
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result)
    result = re.sub(r'(.)\1+', r'\1\1', result)
    # result = " ".join(re.findall('[A-Z][^A-Z]*', result))
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'#', ' ', result)
    result = re.sub(r'[^\w\s]', r'', result)
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)
    # result = tokenize(result)
    return result


def tokenize(tweet):
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize(token))
    res = ' '.join(result)
    return res


def insert_hate_label(threshold):
    dataset = pd.read_csv(annotated_data_path)
    label = [1 if label_mean >= threshold else 0 for label_mean in dataset['label_mean']]
    dataset.insert(1, "label", label, True)
    dataset.to_csv(f'{DATA_DIR}/parler_annotated_data_labeled_{threshold}.csv', index=False)
    return dataset


def prep_train(threshold=3, tokenizer=False):
    dataset = insert_hate_label(threshold)
    if tokenizer:
        dataset['text'] = dataset['text'].apply(preprocess)
    dataset.drop(dataset.index[(dataset['text'] == '')], axis=0, inplace=True)
    dataset = dataset[['label', 'text']]
    dataset.to_csv(f'{DATA_DIR}/parler_annotated_data_prep_{threshold}.csv', index=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = [label for label in df['label']]
        self.texts = [self.tokenizer(text,
                                     padding='max_length', max_length = 512, truncation=True,
                                     return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.7):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer