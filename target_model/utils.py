from collections import Counter
import pandas as pd
import json
import os.path
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random

def get_translated_text(x, langs):
    lang = langs[random.randint(0,len(langs)-1)]
    res = x[lang]
    while pd.isna(res):
        lang = langs[random.randint(0,len(langs)-1)]
        res = x[lang]
    return res

def define_target(data):
    new_data = {}
    for key, post in data.items():
        labels_counter = Counter([x['label'] for x in post['annotators']])
        target_counter = Counter([y for x in post['annotators']  for y in x['target']])
        new_data[key] = {'label': labels_counter.most_common(1)[0][0],
                     'target': target_counter.most_common(1)[0][0],
                     'tokens': post['post_tokens']}
    return new_data

def load_anno_data():
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, '../data/dataset.json')
    with open(path, 'r') as f:
        data = json.load(f)
    new_data = define_target(data)
    df = pd.DataFrame(new_data).T
    return df

def preprocessing(input_texts,  tokenizer, max_len, topic_model=None):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  input_texts_finall = []
  if topic_model:
      topics, probs = topic_model.transform(input_texts)
      topics_names = list(map(lambda x: ' [SEP] '+' '.join(topic_model.topic_names[x].split('_')[1:]), topics))
      input_texts = list(map(lambda x: x[0]+x[1], zip(input_texts, topics_names)))
  return tokenizer.batch_encode_plus(
                        input_texts,
                        add_special_tokens = True,
                        max_length = max_len,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True
                   )
  # for input_text in input_texts:
      # if topic_model:
      #   topic_id = topic_model.transform([input_text])[0][0]
      #   topic_name = topic_model.topic_names[topic_id]
      #   topic_name = ' '.join(topic_name.split('_')[1:])
      # tokens_text = tokenizer.tokenize(input_text)
      # if topic_model:
        # tokens_topic = tokenizer.tokenize(topic_name)
        # addition = ['[SEP]'] + tokens_topic
      #   addition = '[SEP]' + topic_name
      # else:
      #   addition = ''
      # # tokens = tokens_text+addition
      # input_text = input_text +"[SEP]"+addition
      # input_texts_finall.append(input_text)
  # print(' '.join(tokens))



def data_loader(token_id, attention_masks, labels, train_idx, val_idx, batch_size):
    # Train and validation sets
    train_set = TensorDataset(token_id[train_idx], 
                              attention_masks[train_idx], 
                              labels[train_idx])

    val_set = TensorDataset(token_id[val_idx], 
                            attention_masks[val_idx], 
                            labels[val_idx])

    # Prepare DataLoader
    train_dataloader = DataLoader(
                train_set,
                sampler = RandomSampler(train_set),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_set,
                sampler = SequentialSampler(val_set),
                batch_size = batch_size
            )
    return train_dataloader, validation_dataloader

def evaluate(loss, true_labels, pred_labels, losses_scores, accuracy_scores, nb_steps, mode):
    ts_accuracy = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels, average='macro')
    precision = precision_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    losses_scores[mode].append(loss/nb_steps)
    accuracy_scores[mode].append(ts_accuracy)
    text_to_print = f'{mode} evaluation\nloss = {loss/nb_steps:.3f} accuracy = {ts_accuracy:.3f}, precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}\n'
    return text_to_print

def forward_pass(dataloader, model, optimizer=None, evaluation=False):
    loss = 0
    true_labels, pred_labels = [], []
    examples, steps = 0, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        if evaluation:
            with torch.no_grad():
                output = model(b_input_ids, 
                               token_type_ids = None, 
                               attention_mask = b_input_mask, 
                               labels = b_labels)
        else:
            optimizer.zero_grad()
            output = model(b_input_ids, 
                           token_type_ids = None, 
                           attention_mask = b_input_mask, 
                           labels = b_labels)
            output.loss.backward()
            optimizer.step()
        loss += output.loss.item()
        examples += b_input_ids.size(0)
        steps += 1
        preds = torch.argmax(output.logits, dim=1)
        true_labels.extend(b_labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
    return loss, true_labels, pred_labels, examples, steps, model

def create_model(num_labels,pretrained_name='bert-base-uncased'):
    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = num_labels,
        output_attentions = False,
        output_hidden_states = False,
        hidden_act ='relu',
        classifier_dropout =0.3
        
    )

    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = 8e-6,
                                  eps = 1e-08
                                  )
    return model, optimizer
