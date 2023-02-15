import os
import sys
from os.path import dirname, abspath

project_path = dirname(dirname(abspath(__file__)))
model1_dir = os.path.join(project_path, 'hate_detection_model')
sys.path.append(project_path)
sys.path.append(model1_dir)

import torch
import numpy as np
import pandas as pd
from datetime import datetime

from transformers import BertTokenizer
from model1 import BertClassifier, Dataset, DATA_DIR, preprocess


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, clear_text):
        self.df = df
        self.clear_text = clear_text
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

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

    def __len__(self):
        return len(self.df)

    def get_text(self, idx):
        return self.df['text'].iloc[idx]

    def __getitem__(self, idx):
        text = self._transform(self.get_text(idx))
        return text


def device_run():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'{device} is running')
    return device


def predict(input, model, device):
    mask = input['attention_mask'].to(device)
    input_id = input['input_ids'].squeeze(1).to(device)
    output = model(input_id, mask)
    return output.argmax(dim=1)


def run_model(df):
    device = device_run()
    BATCH_SIZE = 8
    labels = []
    with torch.no_grad():
        model = BertClassifier(mode='classification').to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        dataset = Dataset(df=df, clear_text=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
        for input in dataloader:
            preds = predict(input, model, device)
            preds = list(np.array(preds.cpu()))
            now = datetime.now()
            print(f'{now.strftime("%d/%m/%Y %H:%M:%S")}: {len(labels)}) {preds}')
            labels += preds
    df.insert(1, "hate_label", labels, True)
    df.to_csv(os.path.join(DATA_DIR, output_name))


file_name = 'parler_data000000000000_hatexplain_True.csv'
output_name = 'parler_data000000000000_final_3.csv'
model_path = '/home/nadavsc/Desktop/projects/HateRecognition/model1/running_16/model_by_loss_75.pt'
df = pd.read_csv(os.path.join(DATA_DIR, file_name))
run_model(df=df)
