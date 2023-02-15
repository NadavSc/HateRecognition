import os
import sys
from os.path import dirname, abspath

project_path = dirname(dirname(abspath(__file__)))
model1_dir = os.path.join(project_path, 'model1')
sys.path.append(project_path)
sys.path.append(model1_dir)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model1 import Dataset, CPU_Unpickler, DATA_DIR, model_init, current_device, metric_calc, set_save_dir


def load_history(save_dir):
    with open(f'{save_dir}/acc_train.pkl', 'rb') as f:
        acc_train = CPU_Unpickler(f).load()
    with open(f'{save_dir}/loss_train.pkl', 'rb') as f:
        loss_train = CPU_Unpickler(f).load()
    with open(f'{save_dir}/acc_val.pkl', 'rb') as f:
        acc_val = CPU_Unpickler(f).load()
    with open(f'{save_dir}/loss_val.pkl', 'rb') as f:
        loss_val = CPU_Unpickler(f).load()
    return acc_train, loss_train, acc_val, loss_val


def plot(acc_train, loss_train, acc_val, loss_val, save_dir, epochs):
    x_axis = [i + 1 for i in range(epochs)]
    fig, axis = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Hate Speech Detection - Weighted Loss Threshold 3')
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
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'), dpi=400)


mode = 'classification'  # regression / classification
weighted_loss = False
preprocess = {'clear_text': True,
              'back_translation': True}

BATCH_SIZE = 8
LR = 1e-6
threshold = 4
save_dir = set_save_dir(idx=16, eval=True)
acc_train, loss_train, acc_val, loss_val = load_history(save_dir)
EPOCHS = len(acc_val)

model_path = [os.path.join(save_dir, fname) for fname in os.listdir(save_dir) if fname.startswith('model_by_loss')][0]
print(model_path)
device = current_device()
if preprocess['back_translation']:
    df = pd.read_csv(f'{DATA_DIR}/parler_annotated_data_bt.csv')
else:
    df = pd.read_csv(f'{DATA_DIR}/parler_annotated_data.csv')
_, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])

val_dataset = Dataset(df=df_test, clear_text=preprocess['clear_text'], back_translation=False, threshold=threshold)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
model, optimizer, criterion, threshold, rmse = model_init(threshold=threshold,
                                                          device=device,
                                                          mode=mode,
                                                          lr=LR,
                                                          weighted_loss=weighted_loss)
model.load_state_dict(torch.load(model_path))
model.eval()
metrics = metric_calc(val_dataloader=val_dataloader,
                      model=model,
                      device=device)
print(f'Precision: {metrics["precision"]}\n'
      f'Recall: {metrics["recall"]}\n'
      f'F1: {metrics["f1"]}')
plot(acc_train, loss_train, acc_val, loss_val, save_dir, EPOCHS)
