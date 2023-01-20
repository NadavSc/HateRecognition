import numpy as np
import torch
import evaluate
import logging

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

from model1_huggingface import annotated_prep_data_path
from transformers import AutoTokenizer


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


logging.basicConfig(level=logging.INFO)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device} is running')

dataset = load_dataset('csv', data_files=annotated_prep_data_path, split='train')
dataset = dataset.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2).to(device)
if device == 'cuda':
    model = model.to('cuda')
training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  logging_strategy="epoch",
                                  overwrite_output_dir=True,
                                  num_train_epochs=3,
                                  per_gpu_train_batch_size=8,
                                  per_gpu_eval_batch_size=8,
                                  logging_first_step=True,
                                  do_train=True,
                                  do_eval=True)

metric = evaluate.load("accuracy")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
