# %%

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import preprocessing
import pandas as pd
import numpy as np
import json
from tabulate import tabulate
from tqdm import trange
import random
from sklearn.metrics import classification_report, recall_score,precision_score , f1_score, accuracy_score
from utils import load_anno_data
from bertopic import BERTopic
from TopicTuner.topictuner import TopicModelTuner as TMT
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
# torch.cuda.empty_cache()


# %%



# df = load_anno_data()
df = pd.read_csv('../data/annotated_target_topic_data.csv')
hate_df = df[df['label']=='hatespeech']
# docs = [' '.join(x) for x in df.tokens.values]
# hate_df = hate_df[hate_df['target'].isin({'Asian', 'Other', 'None', 'Women', 'Hispanic'})]
hate_df[hate_df['target']=='Arab'] = 'Islam'
labels_counter = Counter(hate_df.target.tolist())
hate_df = hate_df[hate_df['target'].isin(x[0] for x in labels_counter.most_common(4))]


# %%

# topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
# topics, probs = topic_model.fit_transform(docs)
# df['topic'] = topics
# hate_df = pd.merge(hate_df, df['topic'], left_index=True, right_index=True)
# hate_df

# %%
text = hate_df.text.values
labels = hate_df.target.values
unique_labels= list(set(labels))
num_labels= len(unique_labels)

# %%
labels

# %%
Counter(labels.tolist())

# %%
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True,
    truncation=True
    )

# %%
import sklearn
token_id = []
attention_masks = []
le = sklearn.preprocessing.LabelEncoder()
le.fit(labels)
MAX_LEN = max([len(x.split()) for x in text])
print(MAX_LEN)
def preprocessing(input_text, topic_name, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''

  tokens_text = tokenizer.tokenize(input_text)
  tokens_topic = tokenizer.tokenize(topic_name)
  tokens = tokens_text+['[SEP]'] + tokens_topic
  return tokenizer.encode_plus(
                        tokens,
                        add_special_tokens = True,
                        max_length = MAX_LEN,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )


def get_topic_name_by_doc_ind(i):
    topic = hate_df.reset_index().iloc[i]['topics']
    topic_name = topic_model.topic_names[topic]
    return topic_name

for i, sample in enumerate(text):
  # topic_name = get_topic_name_by_doc_ind(i)
  topic_name = df.loc[i, 'topics']
  topic_name = ' '.join(topic_name.split('_')[1:])
  encoding_dict = preprocessing(sample, topic_name, tokenizer)
  token_id.append(encoding_dict['input_ids']) 
  attention_masks.append(encoding_dict['attention_mask'])


token_id = torch.cat(token_id, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(le.transform(labels))

# %%

val_ratio = 0.2
# Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
batch_size = 16

# Indices of the train and validation splits stratified by labels
train_idx, val_idx = train_test_split(
    np.arange(len(labels)),
    test_size = val_ratio,
    shuffle = True,
    stratify = labels)

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

# %%
# Load the BertForSequenceClassification model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = num_labels,
    output_attentions = False,
    output_hidden_states = False,
)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-6,
                              eps = 1e-08
                              )



# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)
# Run on GPU
model = model.to(device)
# Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
epochs = 10
accuracy_scores = {'train':[], 'val':[]}
losses_scores = {'train':[], 'val':[]}
for epoch in trange(epochs, desc = 'Epoch'):
    
    # ========== Training ==========
    
    # Set model to training mode
    model.train()
    
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # Tracking variables 
    true_labels = []
    pred_labels = []
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # b_labels = b_labels.type(torch.LongTensor)
        optimizer.zero_grad()
        # Forward pass
        train_output = model(b_input_ids, 
                             token_type_ids = None, 
                             attention_mask = b_input_mask, 
                             labels = b_labels)
        # Backward pass
        train_output.loss.backward()
        optimizer.step()
        # Update tracking variables
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        preds = torch.argmax(train_output.logits, dim=1)
        true_labels.extend(b_labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
    # Calculate the evaluation metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = recall_score(true_labels, pred_labels, average='macro')
    recall = precision_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    losses_scores['train'].append(tr_loss/nb_tr_steps)
    accuracy_scores['train'].append(accuracy)
    train_text_to_print = f'Epoch {epoch+1}:\nTrain evaluation\nloss = {tr_loss/nb_tr_steps:.3f} accuracy = {accuracy:.3f}, precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}\n'

    # Set model to evaluation mode
    model.eval()

    # Tracking variables 
    true_labels = []
    pred_labels = []
    ts_loss = 0
    nb_ts_steps = 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # b_labels = b_labels.type(torch.LongTensor)
        # with torch.no_grad():
        # Forward pass
        eval_output = model(b_input_ids, 
                            token_type_ids = None, 
                            attention_mask = b_input_mask,
                            labels = b_labels)
        # logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Convert the logits to predictions
        preds = torch.argmax(eval_output.logits, dim=1)
        true_labels.extend(b_labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
        ts_loss += eval_output.loss.item()
        nb_ts_steps += 1
    # Calculate the evaluation metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = recall_score(true_labels, pred_labels, average='macro')
    recall = precision_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    losses_scores['val'].append(ts_loss/nb_ts_steps)
    accuracy_scores['val'].append(accuracy)
    # Print the evaluation metrics
    val_text_to_print = f'Validation evaluation\nloss = {ts_loss/nb_ts_steps:.3f} accuracy = {accuracy:.3f}, precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}\n'
    print(train_text_to_print+val_text_to_print)

# %%
topic_model = BERTopic.load('bert_model')
tmt = TMT.load('temp')

# %%
newBTModel = tmt.getBERTopicModel(165,16)
newBTModel.umap_model = tmt.reducer_model
newBTModel.fit_transform(df.text)
new_sentence = 'I hate muslims'
topic_id = newBTModel.transform([new_sentence])[0][0]
topic_name = topic_model.topic_names[topic_id]
encoding_dict = preprocessing(new_sentence, topic_name, tokenizer)
# We need Token IDs and Attention Mask for inference on the new sentence
test_ids = []
test_attention_mask = []


# Extract IDs and Attention Mask
test_ids.append(encoding_dict['input_ids'])
test_attention_mask.append(encoding_dict['attention_mask'])
test_ids = torch.cat(test_ids, dim = 0)
test_attention_mask = torch.cat(test_attention_mask, dim = 0)

# Forward pass, calculate logit predictions
with torch.no_grad():
  output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

prediction = le.inverse_transform([output.logits.cpu().data.numpy().argmax()])


print('Input Sentence: ', new_sentence)
print('Predicted Class: ', prediction)
print("Topic name: ", topic_name)


# %%
pd.DataFrame(losses_scores).plot(title='Loss',figsize=(10,5))

# %%
pd.DataFrame(accuracy_scores).plot(title='Accuracy',figsize=(10,5))


