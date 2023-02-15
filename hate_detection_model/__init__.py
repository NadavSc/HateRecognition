import os
import io
import re
import torch
import random
import gensim
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from transformers import BertModel, BertTokenizer
from os.path import abspath, dirname
from nltk import WordNetLemmatizer
from sklearn.metrics import confusion_matrix


PATH = dirname(dirname(abspath(__file__)))
model1_dir = os.path.join(PATH, 'hate_detection_model')
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


def set_save_dir(idx, eval=False):
    return os.path.join(model1_dir, f'running_{idx}') if eval else os.path.join(model1_dir, f'running_{idx + 1}')


def current_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'{device} is running')
    return device


def calculate_precision_recall_f1(tp, fp, fn):
    """ Calculate precision, recall, and f1 score
    :param tp: true positive number
    :type tp: int
    :param fp: false positive number
    :type fp: int
    :param fn: false negative number
    :type fn: int
    :return: (precision, recall, f1 score)
    :rtype: tuple
    """

    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = 1.0 * tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = 1.0 * tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def metric_calc(val_dataloader, model, device):
    y_true = []
    y_pred = []
    torch.manual_seed(1)
    with torch.no_grad():
        for val_input, val_label in val_dataloader:
            val_label = val_label.to(device).float()[:, np.newaxis]
            mask = val_input['attention_mask'].squeeze(1).to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            outputs = model(input_id, mask)

            prediction = np.array(F.softmax(outputs, dim=1).cpu())
            cur_pred = np.argmax(prediction, axis=1)
            y_pred += list(cur_pred)

            cur_true = np.array(val_label.cpu())
            y_true += list(cur_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def lemmatize(token):
    return WordNetLemmatizer().lemmatize(token, pos='v')


def remove_dups(post, thresh=4):
    post = post.split()
    for num_of_words in range(1, int(len(post) / thresh)):
        i = 0
        while i + num_of_words < len(post):
            orig_seq = post[i:i + num_of_words]
            j = i + num_of_words
            dif_seq = post[j: j + num_of_words]
            while j + num_of_words < len(post) and dif_seq == orig_seq:
                j += num_of_words
                dif_seq = post[j:j + num_of_words]
            if j - i > thresh * num_of_words:
                return False
            i += 1
    return True


def remove_duplicates(post):
    if not pd.isna(post):
        if remove_dups(post=post, thresh=7):
            return post
        return None


def preprocess(text, p=0.7):
    if random.random() < p:
        result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)
        result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
        result = re.sub(r'http\S+', '', result)
        result = re.sub(r'bit.ly/\S+', '', result)
        result = re.sub(r'(.)\1+', r'\1\1', result)
        result = re.sub(r'&[\S]+?;', '', result)
        result = re.sub(r'#', ' ', result)
        result = re.sub(r'[^\w\s]', r'', result)
        result = re.sub(r'\w*\d\w*', r'', result)
        result = re.sub(r'\s\s+', ' ', result)
        result = re.sub(r'(\A\s+|\s+\Z)', '', result)
        return result
    return text


def tokenize(tweet):
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize(token))
    res = ' '.join(result)
    return res


def back_translation_dup_remove():
    df = pd.read_csv(f'{DATA_DIR}/parler_annotated_data_bt.csv')
    languages = ['spanish', 'german', 'french', 'russian', 'chinese']
    df = df.drop('text.1', axis=1)
    for lang in languages:
        df[lang] = df[lang].apply(remove_duplicates)
    df.to_csv(f'{DATA_DIR}/parler_annotated_data_bt_post.csv')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, clear_text, back_translation, threshold=3):
        self.df = df
        self.clear_text = clear_text
        self.back_translation = back_translation
        self.threshold =threshold
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.languages = ['text', 'spanish', 'german', 'french']

    def _transform(self, text):
        if self.clear_text:
            text = preprocess(text, p=1)
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
        language = self.languages[random.randint(0, 3)]
        while pd.isna(self.df[language].iloc[idx]):
            language = self.languages[random.randint(0, 3)]
        return self.df[language].iloc[idx]

    def get_text(self, idx):
        if self.back_translation:
            return self.random_language(idx)
        return self.df['text'].iloc[idx]

    def __getitem__(self, idx):
        text = self._transform(self.get_text(idx))
        label = torch.tensor(self.get_label(idx))
        return text, label


def loss_calc(output, label, criterion, rmse=False):
    if rmse:
        return torch.sqrt(criterion(output, label))
    return criterion(output, label.long())


def model_init(threshold, device, mode, lr, weighted_loss=False):
    if mode == 'regression':
        rmse = True
        criterion = nn.MSELoss()
        threshold = 0
    else:
        rmse = False
        weight = torch.tensor([0.9, 1.5]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight) if weighted_loss else nn.CrossEntropyLoss()
    model = BertClassifier(mode=mode).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    return model, optimizer, criterion, threshold, rmse


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
