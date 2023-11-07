import openbackdoor as ob
from openbackdoor.defenders import RAPDefender, BKIDefender, STRIPDefender, Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn.functional as F
import math
from openbackdoor.attackers.poisoners import BadNetsPoisoner
import copy

class AdaptedBadNets(BadNetsPoisoner):
     def insert(
        self, 
        text: str, 
    ):
        r"""
            Insert trigger(s) randomly in a sentence.
        
        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        for i in range(self.num_triggers):
            insert_word = self.triggers[i]
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)
    
class AdaptedBKI(BKIDefender):
    def __init__(self, num=5, **kwargs):
        super().__init__(**kwargs)
        self.num = num
        
    def analyze_sent(self, model, sentence):
        model.eval()
        input_sents = [sentence]
        split_sent = sentence.strip().split()
        delta_li = []
        for i in range(len(split_sent)):
            if i != len(split_sent) - 1:
                sent = ' '.join(split_sent[0:i] + split_sent[i + 1:])
            else:
                sent = ' '.join(split_sent[0:i])
            input_sents.append(sent)
        repr_embedding = []
        for i in range(0, len(input_sents), 32):
            with torch.no_grad():
                input_batch = model.tokenizer(input_sents[i:i+32], padding=True, truncation=True, return_tensors="pt").to(model.device)
                repr_embedding.append(model.get_repr_embeddings(input_batch)) # batch_size, hidden_size
        repr_embedding = torch.cat(repr_embedding)
        orig_tensor = repr_embedding[0]
        for i in range(1, repr_embedding.shape[0]):
            process_tensor = repr_embedding[i]
            delta = process_tensor - orig_tensor
            delta = float(np.linalg.norm(delta.detach().cpu().numpy(), ord=np.inf))
            delta_li.append(delta)
        assert len(delta_li) == len(split_sent)
        sorted_rank_li = np.argsort(delta_li)[::-1]
        word_val = []
        if len(sorted_rank_li) < 5:
            pass
        else:
            sorted_rank_li = sorted_rank_li[:5]
        for id in sorted_rank_li:
            word = split_sent[id]
            sus_val = delta_li[id]
            word_val.append((word, sus_val))
        return word_val
    
    def analyze_data(self, model, poison_train):
        for sentence, target_label, _ in poison_train:
            sus_word_val = self.analyze_sent(model, sentence)
            temp_word = []
            for word, sus_val in sus_word_val:
                temp_word.append(word)
                if word in self.bki_dict:
                    orig_num, orig_sus_val = self.bki_dict[word]
                    cur_sus_val = (orig_num * orig_sus_val + sus_val) / (orig_num + 1)
                    self.bki_dict[word] = (orig_num + 1, cur_sus_val)
                else:
                    self.bki_dict[word] = (1, sus_val)
            self.all_sus_words_li.append(temp_word)
        sorted_list = sorted(self.bki_dict.items(), key=lambda item: math.log10(item[1][0]) * item[1][1], reverse=True)
        bki_word = [x[0] for x in sorted_list[:self.num]]
        self.bki_word = bki_word
        print(bki_word)
        flags = []
        for sus_words_li in self.all_sus_words_li:
            flag = 0
            for word in self.bki_word:
                if word in sus_words_li:
                    flag = 1
                    break
            flags.append(flag)
            
        filter_train = []
        s = 0
        for i, data in enumerate(poison_train):
            if flags[i] == 0:
                filter_train.append(data)
                if data[-1]==1:
                    s+=1
        print(len(filter_train), s)
        return filter_train
