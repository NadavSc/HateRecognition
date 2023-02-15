import os
import sys
from os.path import dirname, abspath

project_path = dirname(dirname(abspath(__file__)))
model1_dir = os.path.join(project_path, 'model1')
sys.path.append(project_path)
sys.path.append(model1_dir)

import pdb
import torch
import copy
import numpy as np
import pandas as pd

from tqdm import tqdm
from model1 import Dataset, DATA_DIR, save_results, metric_calc, current_device, model_init, loss_calc, set_save_dir, confusion_matrix, calculate_precision_recall_f1


def acc_calc(output, label):
    if mode == 'regression':
        return (abs(output - label) <= 0.1).sum().item()
    return (output.argmax(dim=1) == label).sum().item()


def save_dir_init():
    try:
        save_dir = os.path.join(model1_dir, f'running_1')
        os.mkdir(os.path.join(model1_dir, 'running_1'))
    except:
        idx = max([int(fname.split('_')[-1]) for fname in os.listdir(model1_dir) if 'running' in fname])
        save_dir = set_save_dir(idx)
        os.mkdir(save_dir)
    return save_dir


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
        y_pred = []
        y_true = []
        total_acc_train = 0
        total_loss_train = 0
        model.train()
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
        pdb.set_trace()
        with torch.no_grad():
            model.eval()
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device).float()[:, np.newaxis] if mode == 'regression' else val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                batch_loss = loss_calc(output, val_label, criterion)
                acc = acc_calc(output, val_label)

                y_pred += list(np.array(output.argmax(dim=1).cpu()))
                y_true += list(np.array(val_label.cpu()))

                total_loss_val += batch_loss.item()
                total_acc_val += acc

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)
        epoch_loss_train = total_loss_train / len(train_data)
        epoch_acc_train = total_acc_train / len(train_data)
        epoch_loss_val = total_loss_val / len(val_data)
        epoch_acc_val = total_acc_val / len(val_data)
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss_train: .3f} \
                | Train Accuracy: {epoch_acc_train: .3f} \
                | Val Loss: {epoch_loss_val: .3f} \
                | Val Accuracy: {epoch_acc_val: .3f} \
                | Val Precision: {precision: .3f} \
                | Val Recall: {recall: .3f} \
                | Val F1: {f1: .3f}')

        if epoch_loss_val <= best_loss:
            best_loss = epoch_loss_val
            best_acc_loss = epoch_acc_val
            best_model_wts_loss = copy.deepcopy(model.state_dict())
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        hist_total_acc_train.append(epoch_acc_train)
        hist_total_loss_train.append(epoch_loss_train)
        hist_total_acc_val.append(epoch_acc_val)
        hist_total_loss_val.append(epoch_loss_val)
        
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_by_acc_{int(best_acc * 100)}.pt'))
    model.load_state_dict(best_model_wts_loss)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_by_loss_{int(best_acc_loss*100)}.pt'))
    metrics = metric_calc(val_dataloader=val_dataloader, model=model, device=device)
    history = {'acc_train': hist_total_acc_train,
               'loss_train': hist_total_loss_train,
               'acc_val': hist_total_acc_val,
               'loss_val': hist_total_loss_val}
    return model, history, metrics


mode = 'classification'  # regression / classification
weighted_loss = True
preprocess = {'clear_text': True,
              'back_translation': True}
EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-6

threshold = 4
if preprocess['back_translation']:
    df = pd.read_csv(f'{DATA_DIR}/parler_annotated_data_bt_post.csv')
else:
    df = pd.read_csv(f'{DATA_DIR}/parler_annotated_data.csv')
df_train, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])

save_dir = save_dir_init()
device = current_device()
model, optimizer, criterion, threshold, rmse = model_init(threshold=threshold,
                                                          device=device,
                                                          mode=mode,
                                                          lr=LR,
                                                          weighted_loss=weighted_loss)
model, history, metrics = train(model=model,
                                train_data=df_train,
                                val_data=df_test,
                                criterion=criterion,
                                optimizer=optimizer,
                                preprocess=preprocess,
                                threshold=threshold)
save_results(save_dir, history['acc_train'], history['loss_train'], history['acc_val'], history['loss_val'])
