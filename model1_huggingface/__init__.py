import os
import pandas as pd

from os.path import abspath, dirname

PATH = dirname(dirname(abspath(__file__)))
DATA_DIR = os.path.join(PATH, 'data')
annotated_data_path = os.path.join(DATA_DIR, 'parler_annotated_data.csv')
annotated_labeled_data_path = os.path.join(DATA_DIR, 'parler_annotated_data_labeled.csv')
annotated_prep_data_path = os.path.join(DATA_DIR, 'parler_annotated_data_prep.csv')


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
