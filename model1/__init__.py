import os
import io
import re
import torch
import pdb
import random
import gensim
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from datetime import datetime
from BackTranslation import BackTranslation
from transformers import BertModel, BertTokenizer
from os.path import abspath, dirname
from nltk import WordNetLemmatizer
from back_translate import back_translate_augment

PATH = dirname(dirname(abspath(__file__)))
DATA_DIR = os.path.join(PATH, 'data')
annotated_data_path = os.path.join(DATA_DIR, 'parler_annotated_data.csv')
annotated_labeled_data_path = os.path.join(DATA_DIR, 'parler_annotated_data_labeled.csv')
np.random.seed(42)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def lemmatize(token):
    return WordNetLemmatizer().lemmatize(token, pos='v')


def preprocess(text, p=0.7):
    if random.random() < p:
        result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)
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
    return text


def tokenize(tweet):
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize(token))
    res = ' '.join(result)
    return res


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, clear_text, back_translation, threshold=3):
        self.df = df
        self.clear_text = clear_text
        self.back_translation = back_translation
        self.threshold =threshold
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.languages = ['text', 'BackTranslation_de', 'BackTranslation_fr', 'BackTranslation_es', 'BackTranslation_nl', 'BackTranslation_no']

    def _transform(self, text):
        if self.clear_text:
            text = preprocess(text, p=0.7)
        text = self.tokenizer(text=text,
                              padding='max_length',
                              max_length=512,
                              truncation=True,
                              return_tensors="pt")
        for k, v in text.items():
            text[k] = torch.tensor(v, dtype=torch.long)
        return text

    def classes(self):
        return self.df['label_mean']

    def __len__(self):
        return len(self.df)

    def get_label(self, idx):
        label = self.df['label_mean'].iloc[idx]
        if self.threshold == 0:  # Regression
            return label
        return int(label > self.threshold)  # Classification

    def random_language(self, idx):
        language = self.languages[random.randint(0, 5)]
        while pd.isna(self.df[language].iloc[idx]):
            language = self.languages[random.randint(0, 5)]
        return self.df[language].iloc[idx]

    def get_text(self, idx):
        if self.back_translation:
            return self.random_language(idx)
        return self.df['text'].iloc[idx]

    def __getitem__(self, idx):
        text = self._transform(self.get_text(idx))
        label = torch.tensor(self.get_label(idx))
        return text, label


class BertClassifier(nn.Module):
    def __init__(self, mode='regression', dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        if mode == 'regression':
            self.linear = nn.Linear(768, 1)
        else:
            self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def save_results(save_dir, acc_train, loss_train, acc_val, loss_val):
    with open(f'{save_dir}/acc_train.pkl', 'wb') as f:
        pickle.dump(acc_train, f)
    with open(f'{save_dir}/loss_train.pkl', 'wb') as f:
        pickle.dump(loss_train, f)
    with open(f'{save_dir}/acc_val.pkl', 'wb') as f:
        pickle.dump(acc_val, f)
    with open(f'{save_dir}/loss_val.pkl', 'wb') as f:
        pickle.dump(loss_val, f)


def plot(save_dir):
    with open(f'{save_dir}/acc_train.pkl', 'rb') as f:
        acc_train = CPU_Unpickler(f).load()
    with open(f'{save_dir}/loss_train.pkl', 'rb') as f:
        loss_train = CPU_Unpickler(f).load()
    with open(f'{save_dir}/acc_val.pkl', 'rb') as f:
        acc_val = CPU_Unpickler(f).load()
    with open(f'{save_dir}/loss_val.pkl', 'rb') as f:
        loss_val = CPU_Unpickler(f).load()
    x_axis = [i + 1 for i in range(20)]
    fig, axis = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Classification')
    axis[0].plot(x_axis, loss_train, label='train')
    axis[0].plot(x_axis, loss_val, label='val')
    axis[0].set_ylabel('Loss')
    axis[0].set_xlabel('Epoch')
    axis[0].legend()
    axis[1].plot(x_axis, acc_train, label='train')
    axis[1].plot(x_axis, acc_val, label='val')
    axis[1].set_ylabel('Accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].legend()
    plt.show()

