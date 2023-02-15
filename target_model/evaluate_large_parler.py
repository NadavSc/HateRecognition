#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np
import json
from tabulate import tabulate
from tqdm import trange
import random
import seaborn as sns
from sklearn.metrics import classification_report, recall_score,precision_score , f1_score, accuracy_score
import sys
import re
sys.path.append("HateRecognition/target_model/")
from utils import preprocessing, load_anno_data, data_loader, define_target, create_model
from bertopic import BERTopic
from TopicTuner.topictuner import TopicModelTuner as TMT
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sklearn.model_selection import StratifiedKFold
from hdbscan import HDBSCAN
from sklearn.metrics import confusion_matrix
sys.path.append("../")
from TweetNormalizer import normalizeTweet
import gc
from cleantext import clean
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import os
from tqdm import tqdm
import logging

logging.root.setLevel(logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with_topic=True
model_name = "hatexplain"
pred_count_dict = {'African':0, 'Homosexual':0, 'Islam':0, 'Jewish':0,'other':0}
#filter by lang
def get_lang_detector(nlp, name):
        return LanguageDetector()
# In[3]:
data_dir_name = "../data/large_parler/"
ready_data_dir_name = "../data/ready_data/large_parler/"
results_dir_name = "../results/large_parler/"
total_size_before = 0
total_size_after = 0
files_names_list = os.listdir(data_dir_name)
for file in files_names_list[3*int(len(files_names_list)/4):]:
    if file in os.listdir(results_dir_name):
        continue
    file_name = f"large_parler/{file.split('.')[0]}"
    if not f"{file.split('.')[0]}_melted.csv" in os.listdir(ready_data_dir_name):
        df = pd.read_csv(fr'../data/{file_name}.csv', encoding="utf8")
        total_size_before += len(df)
        logging.info(f"Size of {file} before preprocessing is: {len(df)}")
        text_col_name = 'body'
        target_col_name = 'target'
        id_col_name = 'id'
        # rename columns
        df.rename(columns={text_col_name:"text"}, inplace=True)
        df.rename(columns={target_col_name:"target"}, inplace=True)
        df.rename(columns={id_col_name:"id"}, inplace=True)
        df.dropna(axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if file_name.__contains__("toxigen"):
            transform_dict = {'muslim':'Islam','jewish':'Jewish','lgbtq':'Homosexual','black':'African',}
            df['target'] = df['target'].apply(
                lambda x: next((v for k, v in transform_dict.items() if k in re.split(' |/', x.lower())), 'other')
            )
            if file_name == "small_toxigen":
                df = df[df.toxicity_ai>=4]
                df = df[df.framing!='disagreement']
                len(df)
        elif file_name.__contains__("hatexplain"):
            df = df[df['label']=='hatespeech'].reset_index(drop=True)
            df.loc[df['target']=='Arab','target'] = 'Islam'
            labels_counter = Counter(df.target.tolist())
            df.loc[~ df['target'].isin(x[0] for x in labels_counter.most_common(4)),'target'] = 'other'
        elif file_name =="parler_target_annotated":
            df.target = df.target.apply(lambda x: "other" if x=="Politician" else x)
    
        nlp = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=get_lang_detector)
        nlp.add_pipe('language_detector', last=True)
        if file_name.endswith("_bt"):
            df['lang'] = df.english.apply(lambda x: nlp(x)._.language['language'])
        else:
            df['lang'] = df.text.apply(lambda x: nlp(x)._.language['language'])
        df = df[df['lang']=='en']
        if file_name.endswith("_bt"):
            langs = ['english', 'spanish',
            'german', 'franch', 'russian', 'chinese']
            for lang in langs:
                df[lang] = df[lang].apply(clean, no_emoji=True)
                df[lang] = df[lang].apply(normalizeTweet)
            df = pd.melt(df, id_vars=['id','target'],value_vars=langs,var_name='lang', value_name='text')
            df = df[~df.text.isin(["",'nan'])]
        else:
            df.text = df.text.apply(clean, no_emoji=True)
            df.text = df.text.apply(normalizeTweet)
        df.to_csv(fr'../data/ready_data/{file_name}_melted.csv', encoding="utf8")
    else:
      df = pd.read_csv(fr'../data/ready_data/{file_name}_melted.csv', encoding="utf8")
    if with_topic:
        topic_model = BERTopic.load('./models/topic_model_with_other')
        model = BertForSequenceClassification.from_pretrained(
        f'models/finetune_topic_other_{model_name}/'    
    )
    else:
        topic_model = None    
        model = BertForSequenceClassification.from_pretrained(
            f'models/finetune_plain_{model_name}/'    
        )
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case = True,
        truncation=True
        )
    if with_topic:
        topics, probs = topic_model.transform(df.text.values.tolist())
        df['topics'] = [topic_model.topic_names[x] for x in topics]
        df['probs'] = probs
        # df['topics'].hist(figsize=(20,10),bins=20)

    token_id = []
    attention_masks = []
    text = df.text.values.tolist()
    MAX_LEN = max([len(x.split()) for x in text])
    logging.info(MAX_LEN)
    encoding_dict = preprocessing(text,tokenizer, MAX_LEN, topic_model)
    token_id = encoding_dict['input_ids']
    attention_masks = encoding_dict['attention_mask']
    batch_size = 32

    gc.collect()
    torch.cuda.empty_cache()
    model.to('cuda')
    train_set = TensorDataset(token_id, 
                                attention_masks)

    data = DataLoader(
                    train_set,
                    sampler = SequentialSampler(train_set),
                    batch_size = batch_size
                )
    pred_labels = []
    for i,batch in tqdm(enumerate(data)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            eval_output = model(b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask)
            preds = torch.argmax(eval_output.logits, dim=1)
            pred_labels.extend(preds.cpu().numpy())

    labl_dict = {0:'African', 1:'Homosexual', 2:'Islam', 3:'Jewish',4:'other'}
    df['pred'] = [labl_dict[x] for x in pred_labels]
    df['pred_labels'] = pred_labels
    df['pred'] 
    total_size_after += len(df)
    logging.info(f"Size of {file} after preprocessing is: {len(df)}\n\n")
    pred_count_dict.update(df.groupby('pred').count()['pred_labels'].to_dict())
    logging.info(pred_count_dict)
    df.to_csv(fr'../results/{file_name}_{model_name}_{str(with_topic)}.csv')
    files_names_list = os.listdir(data_dir_name)
logging.info(f"Total size before preprocessing: {total_size_before}\nTotal size after preprocessing: {total_size_after}\nPredicted target distribution: {pred_count_dict}")