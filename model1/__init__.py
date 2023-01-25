import os
import io
import re
import torch
import pdb
import gensim
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from transformers import BertModel, BertTokenizer
from os.path import abspath, dirname
from nltk import WordNetLemmatizer


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


def insert_hate_label(dataset, threshold):
    label = [1 if label_mean >= threshold else 0 for label_mean in dataset['label_mean']]
    dataset.insert(1, "label", label, True)
    dataset.to_csv(f'{DATA_DIR}/classification/parler_annotated_data_labeled{threshold}.csv', index=False)
    return dataset


def prep_data(threshold=3, mode='regression', tokenizer=False):
    dataset = pd.read_csv(annotated_data_path)
    if mode == 'classification':
        dataset = insert_hate_label(dataset, threshold)
        dataset = dataset[['label', 'text']]
    else:
        dataset = dataset[['label_mean', 'text']]
        dataset.rename(columns={'label_mean': 'label'}, inplace=True)
        dataset['label'] = dataset['label'].astype('float32')

    if tokenizer:
        dataset['text'] = dataset['text'].apply(preprocess)
        dataset.drop(dataset.index[(dataset['text'] == '')], axis=0, inplace=True)
    dataset.to_csv(f'{DATA_DIR}/{mode}/parler_annotated_data_prep{threshold}.csv', index=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = [label for label in df['label']]
        self.texts = [self.tokenizer(text,
                                     padding='max_length', max_length=512, truncation=True,
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
    fig, axis = plt.subplots(nrows=1, ncols=2)
    axis[0].plot(loss_train, label='train')
    axis[0].plot(loss_val, label='val')
    axis[0].set_ylabel('Loss')
    axis[0].set_xlabel('Epoch')
    axis[0].legend()
    axis[0].set_xticks(list(range(5)))
    axis[1].plot(acc_train, label='train')
    axis[1].plot(acc_val, label='val')
    axis[1].set_ylabel('Accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].legend()
    axis[1].set_xticks(list(range(5)))
    plt.show()