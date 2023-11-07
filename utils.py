import os
import sys
import errno
import time
import codecs
import numpy as np
import logging 
import torch 
import random 

poisoners = {
        "badnets": {
            "name": "badnets",
            "poison_rate": 0.2,
            "target_label": 1,
            "label_consistency": True,
            "label_dirty": False,
            "triggers": ["cf", "mn", "bb", "tq"],
            "num_triggers": 1,
            "load": True
        },
        "adaptedbadnets": {
            "name": "badnets",
            "poison_rate": 0.2,
            "target_label": 1,
            "label_consistency": True,
            "label_dirty": False,
            "triggers": ["cf", "bb", "sd"],
            "num_triggers": 1,
            "load": True
        },
        "addsent":{
            "name": "addsent",
            "poison_rate": 0.2,
            "target_label": 1,
            "label_consistency": True,
            "label_dirty": False,
            "load": True,
            "triggers": "I watch this 3D movie"
        },
        "synbkd":{
            "name": "synbkd",
            "poison_rate": 0.2,
            "target_label": 1,
            "label_consistency": True,
            "label_dirty": False,
            "load": True,
            "poison_data_basepath": None,
            "poisoned_data_path": "",
            "template_id": -1
        },
         "stylebkd":{
            "name": "stylebkd",
            "poison_rate": 0.2,
            "target_label": 1,
            "label_consistency": True,
            "label_dirty": False,
            "load": True,
            "template_id": 0
        }
}
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from itertools import combinations
from collections import Counter
from sklearn.metrics import accuracy_score
    
def majority(preds, labels=None, C=2):
    if isinstance(preds, list):
        preds = np.array(preds)
    preds = preds.transpose() # N * Group
    final_pred = []
    for i, x in enumerate(preds):
        cnt = np.zeros(C)
        for xx in x:
            cnt[xx]-=1
        idxs = cnt.argsort(kind="stable")
        if labels is None:
            final_pred.append(idxs[0])
        else:
            y = labels[i]
            if idxs[0]!=y:
                final_pred.append(idxs[0])
            else:
                final_pred.append(idxs[1]) # use second largest               

    return np.array(final_pred)

def certified(preds, labels, C=2, target_label=None):
    # C is number of classes
    preds = np.array(preds) # Group * N
    labels = np.array(labels) # N
    if target_label is not None:
        target = labels!=target_label
        print(len(target), sum(target))
        preds = preds[:, target]
        labels = labels[target]
        print(preds.shape)
        
    final_pred = majority(preds, C=C)
    correct = (labels == final_pred)
    n_correct = sum(correct)
    n_wrong = len(labels) - n_correct
    lis_cacc = [n_correct/(n_correct+n_wrong)]
    cpreds = preds[:, correct] # choose correct prediction for certification
    clabels = labels[correct]
    m = len(cpreds)
    n = len(cpreds[0])
    assert n == n_correct
    for i in range(1, m//2+1): # number of backdoored groups
        cacc = 1
        for lis in combinations(range(m), i): # iterate all combinations
            cur = np.copy(cpreds)
            rest = [j for j in range(m) if j not in lis]
            worst_pred = majority(cpreds[rest], clabels, C=C) # find the second common predictions of clean groups
            for j in lis:
                cur[j] = worst_pred
            #print(lis)
            #print(cpreds[:, :10])
            #print(worst_pred[:10])
            hpred = majority(cur, C=C)
            cacc = min(cacc, sum(hpred==clabels)/(n_wrong+n))
        lis_cacc.append(cacc)        
    return lis_cacc

from nltk.tokenize import word_tokenize
import hashlib
def rectify(lis, tot):
    if len(lis)==0:
        return ""
    #if len(lis)<int(tot/2):
    #    return ""
    pre = lis[0]
    text = []
    if pre[-1]!=0:
        text.extend(["[MASK]"]*pre[-1])
    text.append(pre[0])
    for x in lis[1:]:
        if pre[-1]+1<x[-1]:
            text.extend(["[MASK]"]*(x[-1] - pre[-1] - 1))
        pre = x
        text.append(x[0])   
    if tot > pre[-1] + 1:
        text.extend(["[MASK]"]*(tot - pre[-1] - 1))
    return " ".join(text)

def split_group(x, args, allow_empty):
    lis = word_tokenize(x)
    res = [[] for i in range(args.group)]
    for i, x in enumerate(lis):
        h = int(hashlib.md5(x.encode()).hexdigest(), 16) % args.group
        #for k in range(h-2, h+3):
        #    res[(k+args.group)%args.group].append((x, i))
        res[h].append((x, i))
    for i in range(args.group):
        res[i] = rectify(res[i], len(lis))
        if len(res[i])==0 and allow_empty:
            res[i] = " ".join(["[MASK]"]*len(lis))
    return res          
    
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def init_logger(root_dir):
    make_sure_path_exists(root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger