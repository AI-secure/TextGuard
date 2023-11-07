import hashlib

def md5_hash(x, m):
    x = x.lower()
    return int(hashlib.md5(x.encode()).hexdigest(), 16) % m

def sha1_hash(x, m):
    x = x.lower()
    return int(hashlib.sha1(x.encode()).hexdigest(), 16) % m

def sha256_hash(x, m):
    x = x.lower()
    return int(hashlib.sha256(x.encode()).hexdigest(), 16) % m

from openbackdoor.victims import Victim, PLMVictim
from openbackdoor.trainers import Trainer
import random
import torch
from collections import defaultdict
from transformers import AutoModelForSequenceClassification
import math
import numpy as np
import logging
import os
import shutil
import pickle
from tqdm import tqdm
class KIhash:
    def __init__(self, default_hash, num, tokenize_method, train_set, p=5, threshold=10, warm_up_epochs=0,
            epochs=10,
            batch_size=32,
            lr=2e-5,
            num_classes=2,
            model_name='bert',
            model_path='bert-base-uncased', pre_save=None):

        self.dic = {}
        self.p = p
        self.threshold = threshold
        self.default_hash = default_hash
        self.tokenize_method = tokenize_method
        self.bki_dict = {}
        self.bki_model = PLMVictim(model=model_name, path=model_path, num_classes=num_classes)
        self.trainer = Trainer(warm_up_epochs=warm_up_epochs, epochs=epochs, 
                            batch_size=batch_size, lr=lr,
                            save_path='./models/kimodels', ckpt='last')
        if pre_save is None:
            path = None
        else:
            path = f"./kimodels/{pre_save}"
        
        if path is None or not os.path.exists(path):
            self.bki_model = self.trainer.train(self.bki_model, {"train": train_set})
            self.bki_model.plm.save_pretrained(path)
        else:
            self.bki_model.plm = AutoModelForSequenceClassification.from_pretrained(path).cuda()
            
        

        if path is not None:
            if os.path.exists(f"{path}_dic{self.p}.pkl"):
                result = pickle.load(open(f"{path}_dic{self.p}.pkl", "rb"))
            else:
                result = self.analyze_data(self.bki_model, train_set)
                
            if isinstance(result, dict):
                result = list(result.items())
                
            for i, x in enumerate(result):
                if x[1][0]<self.threshold:
                    continue
                self.dic[x[0]] = x[1]
                if i%100==0:
                    print(x)

            if not os.path.exists(f"{path}_dic{self.p}.pkl"):
                with open(f"{path}_dic{self.p}.pkl", "wb") as f:
                    pickle.dump(result, f)
                
        print(len(self.dic))
        
    def analyze_sent(self, model, sentence):
        model.eval()
        input_sents = [sentence]
        split_sent = sentence.strip().split()
        delta_li = []
        cur = set()
        loc = [] 
        for i in range(len(split_sent)):
            y = split_sent[i].lower()
            if y in cur:
                continue
            cur.add(y)
            loc.append(i)
            sent = ' '.join([x for x in split_sent if x.lower()!=y])
            input_sents.append(sent)
        repr_embedding = []
        for i in range(0, len(input_sents), 64):
            with torch.no_grad():
                input_batch = model.tokenizer(input_sents[i:i+64], padding=True, truncation=True, return_tensors="pt").to(model.device)
                repr_embedding.append(model.get_repr_embeddings(input_batch)) # batch_size, hidden_size
        repr_embedding = torch.cat(repr_embedding)
        orig_tensor = repr_embedding[0]
        for i in range(1, repr_embedding.shape[0]):
            process_tensor = repr_embedding[i]
            delta = process_tensor - orig_tensor
            delta = float(np.linalg.norm(delta.detach().cpu().numpy(), ord=np.inf))
            delta_li.append(delta)

        sorted_rank_li = np.argsort(delta_li)[::-1]
        word_val = []
        if len(sorted_rank_li) < self.p:
            pass
        else:
            sorted_rank_li = sorted_rank_li[:self.p]
        for id in sorted_rank_li:
            word = split_sent[loc[id]].lower()
            sus_val = delta_li[id]
            word_val.append((word, sus_val))
        return word_val
    
    def analyze_data(self, model, poison_train):
        for sentence, target_label, _ in tqdm(poison_train):
            sus_word_val = self.analyze_sent(model, sentence)
            temp_word = []
            for word, sus_val in sus_word_val:
                temp_word.append(word)
                if word in self.bki_dict:
                    orig_num, orig_sus_val = self.bki_dict[word]
                    cur_sus_val = orig_sus_val + sus_val
                    self.bki_dict[word] = (orig_num + 1, cur_sus_val)
                else:
                    self.bki_dict[word] = (1, sus_val)
        sorted_list = sorted(self.bki_dict.items(), key=lambda item: item[1][1], reverse=True)
        return sorted_list        
    
    def map(self, x, m):
        x = x.lower()
        if x not in self.dic:
            return -1
        else:
            return self.default_hash(x, m)
        