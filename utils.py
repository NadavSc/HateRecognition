from transformers import AutoTokenizer, AutoModelForTokenClassification
from TweetNormalizer import normalizeTweet
import torch

tokenizer = AutoTokenizer.from_pretrained("TweebankNLP/bertweet-tb2_ewt-pos-tagging")

pos_tagging_model = AutoModelForTokenClassification.from_pretrained("TweebankNLP/bertweet-tb2_ewt-pos-tagging")

def pos_tagging(text):    
    line = normalizeTweet(text)
    print(line)
    input_ids = torch.tensor([tokenizer.encode(line)])
    logits = pos_tagging_model(input_ids).logits
    predicted_class_id = logits.argmax(dim=2)
    tags = [pos_tagging_model.config.id2label[x.item()] for x in predicted_class_id[0]]
    return tags[1:-1]