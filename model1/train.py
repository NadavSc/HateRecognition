import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from model1 import df_train, df_test
from model1 import BertClassifier, Dataset


EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-6


def train(model, train_data, val_data):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'{device} is running')

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

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
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            hist_total_loss_train.append(total_loss_train)

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            hist_total_acc_train.append(acc)

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                hist_total_loss_val.append(total_loss_val)

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
                hist_total_acc_val.append(total_acc_val)

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

    return model, hist_total_acc_train, hist_total_loss_train, hist_total_acc_val, hist_total_loss_val


def plot(loss_train, loss_val, acc_train, acc_val):
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


model = BertClassifier()
model, acc_train, loss_train, acc_val, loss_val = train(model=model,
                                                        train_data=df_train,
                                                        val_data=df_test)
plot(loss_train, loss_val, acc_train, acc_val)


