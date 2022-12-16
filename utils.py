from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import nltk
nltk.download('punkt')

class TokensClassTasks:
    NER = "TweebankNLP/bertweet-tb2_wnut17-ner"
    POS_TAGGING = "TweebankNLP/bertweet-tb2_ewt-pos-tagging"

class Model:
    def __init__(self, task) -> None:
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(task)
        self.model = AutoModelForTokenClassification.from_pretrained(task)

    def run_model(self, text):
        input_ids = torch.tensor([self.tokenizer.encode(text)])
        logits = self.model(input_ids).logits
        predicted_class_id = logits.argmax(dim=2)
        classes = [self.model.config.id2label[x.item()] for x in predicted_class_id[0]]
        return classes[1:-1]
        
def split_rows(s):
    sen = nltk.sent_tokenize(s)
    for s in sen:
        j_s = ''
        if len(s) < 130:
            yield s
        else:
            split_sen = s.split()
            for ss in split_sen:
                if len(ss) > 10:
                    ss = ''
                tmp = ' '.join([j_s, ss])
                if len(tmp) < 130:
                    j_s = tmp
                elif len(j_s) < 130:
                    yield j_s
                    j_s = ss
    if j_s != '':
        yield j_s