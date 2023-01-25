from collections import Counter
import pandas as pd
import json
import os.path
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

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

def preprocessing(input_text,  tokenizer, max_len, topic_name=None):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''

  tokens_text = tokenizer.tokenize(input_text)
  if topic_name:
    tokens_topic = tokenizer.tokenize(topic_name)
    addition = ['[SEP]'] + tokens_topic
  else:
    addition = []
  tokens = tokens_text+addition
  # print(' '.join(tokens))
  return tokenizer.encode_plus(
                        tokens,
                        add_special_tokens = True,
                        max_length = max_len,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True
                   )


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

def create_model(num_labels):
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
