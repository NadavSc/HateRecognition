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

from model1 import BertClassifier, Dataset, DATA_DIR, save_results, plot


def loss_calc(output, label, criterion):
    if rmse:
        return torch.sqrt(criterion(output, label))
    return criterion(output, label.long())


def acc_calc(output, label):
    if mode == 'regression':
        return (abs(output - label) <= 0.1).sum().item()
    return (output.argmax(dim=1) == label).sum().item()


def save_dir_init():
    try:
        save_dir = os.path.join(model1_dir, f'running1')
        os.mkdir(os.path.join(model1_dir, 'running1'))
    except:
        idx = max([int(fname[-1]) for fname in os.listdir(model1_dir) if 'running' in fname])
        save_dir = os.path.join(model1_dir, f'running{idx + 1}')
        os.mkdir(save_dir)
    return  save_dir


def device_run():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'{device} is running')
    return device


def model_init(threshold, weighted_loss=False):
    if mode == 'regression':
        rmse = True
        criterion = nn.MSELoss()
        threshold = 0
    else:
        rmse = False
        weight = torch.tensor([0.7, 1.2]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight) if weighted_loss else nn.CrossEntropyLoss()
    model = BertClassifier(mode=mode)
    optimizer = Adam(model.parameters(), lr=LR)
    return model, optimizer, criterion, threshold, rmse


def train(model, train_data, val_data, criterion, optimizer, preprocess, threshold):
    train = Dataset(df=train_data, clear_text=preprocess['clear_text'], back_translation=preprocess['back_translation'], threshold=threshold)
    val = Dataset(df=val_data, clear_text=preprocess['clear_text'], back_translation=False, threshold=threshold)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    model = model.to(device)
    criterion = criterion.to(device)

    hist_total_acc_train = []
    hist_total_loss_train = []
    hist_total_acc_val = []
    hist_total_loss_val = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts_loss = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100

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

        if total_loss_val < best_loss:
            total_loss_val = best_loss
            best_acc_loss = total_acc_val
            best_model_wts_loss = copy.deepcopy(model.state_dict())
        if total_acc_val > best_acc:
            best_acc = total_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        hist_total_acc_train.append(total_acc_train/len(train_data))
        hist_total_loss_train.append(total_loss_train/len(train_data))
        hist_total_acc_val.append(total_acc_val/len(val_data))
        hist_total_loss_val.append(total_loss_val/len(val_data))
        
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_by_acc_{int(best_acc * 100)}.pt'))
    model.load_state_dict(best_model_wts_loss)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_by_loss_{int(best_acc_loss*100)}.pt'))
    return model, hist_total_acc_train, hist_total_loss_train, hist_total_acc_val, hist_total_loss_val


mode = 'classification'  # regression / classification
weighted_loss = False
preprocess = {'clear_text': True,
              'back_translation': True}
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-6

threshold = 3
if preprocess['back_translation']:
    df = pd.read_csv(f'{DATA_DIR}/parler_annotated_data_bt.csv')
else:
    df = pd.read_csv(f'{DATA_DIR}/parler_annotated_data.csv')
df_train, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])

save_dir = save_dir_init()
device = device_run()
model, optimizer, criterion, threshold, rmse = model_init(threshold, weighted_loss=weighted_loss)
model, acc_train, loss_train, acc_val, loss_val = train(model=model,
                                                        train_data=df_train,
                                                        val_data=df_test,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        preprocess=preprocess,
                                                        threshold=threshold)
save_results(save_dir, acc_train, loss_train, acc_val, loss_val)
# plot(save_dir)