import os
import sys
from os.path import dirname, abspath

project_path = dirname(dirname(abspath(__file__)))
model1_dir = os.path.join(project_path, 'model1')
sys.path.append(project_path)
sys.path.append(model1_dir)

import torch
import pdb
import copy
import numpy as np
import pandas as pd

from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from model1 import BertClassifier, Dataset, DATA_DIR, prep_data, save_results, plot


def loss_calc(output, label, criterion):
    if rmse:
        return torch.sqrt(criterion(output, label))
    return criterion(output, label.long())


def acc_calc(output, label):
    if mode == 'regression':
        return (abs(output - label) <= 0.1).sum().item()
    return (output.argmax(dim=1) == label).sum().item()


def train(model, train_data, val_data, criterion, optimizer):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    hist_total_acc_train = []
    hist_total_loss_train = []
    hist_total_acc_val = []
    hist_total_loss_val = []

    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device).float()[:, np.newaxis] if mode == 'regression' else train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            batch_loss = loss_calc(output, train_label, criterion)
            acc = acc_calc(output, train_label)

            total_loss_train += batch_loss.item()
            total_acc_train += acc
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device).float()[:, np.newaxis] if mode == 'regression' else val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                batch_loss = loss_calc(output, val_label, criterion)
                acc = acc_calc(output, val_label)

                total_loss_val += batch_loss.item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        hist_total_acc_train.append(total_acc_train/len(train_data))
        hist_total_loss_train.append(total_loss_train/len(train_data))
        hist_total_acc_val.append(total_acc_val/len(val_data))
        hist_total_loss_val.append(total_loss_val/len(val_data))
        
    model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(model_wts)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_{mode}.pt'))

    return model, hist_total_acc_train, hist_total_loss_train, hist_total_acc_val, hist_total_loss_val


mode = 'classification'  # regression / classification
threshold = 3
PREPDATA = False
tokenizer = False

rmse = False
EPOCHS = 3
BATCH_SIZE = 8
LR = 1e-6
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'{device} is running')

if mode == 'regression':
    criterion = nn.MSELoss()
    threshold = ''
    annotated_prep_data_path = f'{DATA_DIR}/{mode}/parler_annotated_data_prep{threshold}.csv'
else:
    criterion = nn.CrossEntropyLoss()
    annotated_prep_data_path = f'{DATA_DIR}/{mode}/parler_annotated_data_prep{threshold}.csv'

if PREPDATA:
    prep_data(threshold=threshold, mode=mode, tokenizer=tokenizer)
df = pd.read_csv(annotated_prep_data_path)
df_train, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])

model = BertClassifier(mode=mode)
# weight = torch.tensor([0.7, 1.2]).to(device)
optimizer = Adam(model.parameters(), lr=LR)
try:
    save_dir = os.path.join(model1_dir, f'running1')
    os.mkdir(os.path.join(model1_dir, 'running1'))
except:
    idx = max([int(fname[-1]) for fname in os.listdir(model1_dir) if 'running' in fname])
    save_dir = os.path.join(model1_dir, f'running{idx+1}')
    os.mkdir(save_dir)
model, acc_train, loss_train, acc_val, loss_val = train(model=model,
                                                        train_data=df_train,
                                                        val_data=df_test,
                                                        criterion=criterion,
                                                        optimizer=optimizer)
save_results(save_dir, acc_train, loss_train, acc_val, loss_val)
# plot(save_dir)