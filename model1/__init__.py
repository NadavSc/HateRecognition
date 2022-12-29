import os
import pandas as pd
import torch
import numpy as np

from torch import nn
from transformers import BertModel, BertTokenizer
from os.path import abspath, dirname


PATH = dirname(dirname(abspath(__file__)))
DATA_DIR = os.path.join(PATH, 'data')
annotated_data_path = os.path.join(DATA_DIR, 'parler_annotated_data.csv')
annotated_labeled_data_path = os.path.join(DATA_DIR, 'parler_annotated_data_labeled.csv')
annotated_prep_data_path = os.path.join(DATA_DIR, 'parler_annotated_data_prep.csv')
np.random.seed(42)


def insert_hate_label():
    dataset = pd.read_csv(annotated_data_path)
    label = [1 if label_mean >= 3.0 else 0 for label_mean in dataset['label_mean']]
    dataset.insert(1, "label", label, True)
    dataset.to_csv('parler_annotated_data_labeled.csv', index=False)
    return dataset


def prep_train():
    dataset = insert_hate_label()
    dataset = dataset[['label', 'text']]
    dataset.to_csv('parler_annotated_data_prep.csv', index=False)


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
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

df = pd.read_csv(annotated_prep_data_path)
df_train, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])