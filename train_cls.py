import torch
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import argparse
import numpy as np
import torch.optim as optim
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import copy
from datasets import load_dataset
import pickle
from utils import init_logger, make_sure_path_exists, setup_seed, poisoners, certified, majority
from sklearn.metrics import accuracy_score
from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler, seq_len_to_mask
import openbackdoor as ob
from openbackdoor import load_dataset
from openbackdoor.attackers.poisoners import load_poisoner
from mapper import *
from hashs import *
from nltk.tokenize import word_tokenize
import shutil
import time 

def model_test(test_batch, model):
    model.eval()
    correct=0
    total=0
    preds = []
    with torch.no_grad():
        for batch,_ in test_batch:
            label = batch["labels"].to(device)
            encoder = process(batch)
            output = model(**encoder)[0]
            # output = out_net(output)
            _, predict = torch.max(output,1)
            preds.extend(predict.cpu().numpy().tolist())
            total+=label.size(0)
            correct += (predict == label).sum().item()
    return correct/total, preds

def cls_process(batch):
    lis = batch["texts"].tolist()
    pretoken = False if isinstance(lis[0], str) else True
    x = tokenizer(lis, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt', is_split_into_words=pretoken)
    for k,v in x.items():
        x[k]=v.to(device)
    return x

def calc_warm_up(epochs, batch_train):
    total_steps = len(batch_train)/ args.gradient_accumulation_steps * epochs
    warm_up_steps = args.warm_up_rate * total_steps
    return total_steps, warm_up_steps

def separate(content, mapper, allow_empty):
    lis = [[] for x in range(args.group)]
    #dic = [set() for x in range(args.group)]
    for x,y,z in content:
        res = split_group(x, mapper, allow_empty)
        for i, cur in enumerate(res):
            #strs = " ".join(cur)
            if args.sort:
                cur = sorted(cur, key=lambda x: (sum(tokenizer.encode(x,add_special_tokens=False)),x))
            if args.tokenize!="same":
                cur = " ".join(cur)
            
            if not allow_empty:
                if len(cur)>0:
                    lis[i].append((cur, y, z))
                    #dic[i].add(strs)
            else:
                if len(cur)==0:
                    if args.tokenize!="same":
                        cur = tokenizer.mask_token
                    else:
                        cur = [tokenizer.mask_token]
                lis[i].append((cur, y, z))
                    
                
    return lis
          
def create_batch(content, evalu=True, allow_empty=False):
    labels = np.array([x[1] for x in content])
    poison = [x[-1] for x in content]
    batch_lis = []
    if args.not_split and evalu == True:
        text_lis = [content.copy() for i in range(mapper.num)]
    else:
        text_lis = separate(content, mapper, allow_empty)
    for cur in text_lis: 
        texts = [x[0] for x in cur]
        dataset = DataSet({"idx": list(range(len(cur))), "texts": texts, "labels":[x[1] for x in cur], "poison": [x[-1] for x in cur]})
        dataset.set_input("idx", "texts","labels", "poison")
        if evalu:
            batch = DataSetIter(dataset=dataset, batch_size=args.batchsize*4, sampler=SequentialSampler())  
        else:
            batch = DataSetIter(dataset=dataset, batch_size=args.batchsize, sampler=RandomSampler()) 
        batch_lis.append((dataset, batch))
    for i in range(args.group):
        logger.info(text_lis[i][-1])
        if args.not_split and evalu == True:
            break
    return batch_lis, labels

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type= str, default="0")
    parser.add_argument("--mapper", type=str, default="mask")
    parser.add_argument("--target_word", type=str, default="empty")
    parser.add_argument("--tokenize", type=str, default="nltk")
    parser.add_argument("--hash", type=str, default="md5")
    parser.add_argument("--embedding", type=str, default="embedding/glove.6B.100d.txt")
    parser.add_argument("--ki_t", type=int , default=10)
    parser.add_argument("--ki_p", type=int , default=5)
    parser.add_argument("--threshold", type=float , default=1e9)
    parser.add_argument("--num_triggers", type=int , default=1)
    parser.add_argument("--attack", type=str, default="")
    parser.add_argument("--setting", type=str, default="mix")
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--train", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--base", type=str, default="bert-base-uncased")
    parser.add_argument("--data", type= str, default= "sst-2")
    parser.add_argument("--lr", type=float, default= 2e-5)
    parser.add_argument("--group", type=int , default=3)
    parser.add_argument("--num_class", type=int , default=2)
    parser.add_argument("--target_label", type=int , default=1)
    parser.add_argument("--batchsize", type=int , default=16)
    parser.add_argument("--max_length", type=int , default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int , default=1)
    parser.add_argument("--warm_up_rate", type=float, default=.1)
    parser.add_argument("--epochs", type=int , default=5)
    parser.add_argument("--save_folder", type=str, default="debug")
    parser.add_argument("--run_seed", type = int, default= 42)
    parser.add_argument("--log_name", type= str, default= "test.log")
    parser.add_argument("--always", default=False, action="store_true")
    parser.add_argument("--not_split", default=False, action="store_true")
    parser.add_argument("--sort", default=False, action="store_true")
    args = parser.parse_args()

    lis_gpu_id = list([int(x) for x in args.device])
    device = torch.device("cuda:"+str(lis_gpu_id[0]))
    
    seed = args.run_seed
    setup_seed(seed)
    base_model = args.base
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    save_folder = f"{args.data}-{args.attack}{args.target_label}-{args.setting}-{args.poison_rate}"+args.save_folder
    if args.attack == "noise":
        final_save_folder = "./certified/" + save_folder + "/" + str(seed) + "/"
    else:
        final_save_folder = "./empirical/" + save_folder + "/" + str(seed) + "/"
    make_sure_path_exists(final_save_folder)
    logger = init_logger(final_save_folder)
    logger.info(args)

    process = cls_process
    dataset = load_dataset(name=args.data)
    if args.attack not in ["", "noise"]:
        poisoner = poisoners[args.attack]
        poisoner["poison_rate"] = args.poison_rate
        poisoner["target_label"] = args.target_label
        if args.setting == "clean":
            poisoner["label_consistency"] = True
            poisoner["label_dirty"] = False
        elif args.setting == "dirty":
            poisoner["label_consistency"] = False
            poisoner["label_dirty"] = True
        elif args.setting == "mix":
            poisoner["label_consistency"] = False
            poisoner["label_dirty"] = False
        poisoner["poison_data_basepath"] = f"./poison/{args.data}-{args.attack}"
        poisoner["poisoned_data_path"] = f"./poison/{args.data}-{args.attack}-{args.setting}-{args.poison_rate}"
        if args.attack.find("badnets")!=-1:
            poisoner["load"] = False
            poisoner["num_triggers"] = args.num_triggers
            if args.attack == "adaptedbadnets":
                if args.num_triggers==-3:
                    poisoner["triggers"] = ["cf", "mm", "mb"]
                else: poisoner["triggers"] = poisoner["triggers"][:args.num_triggers]
        logger.info(poisoner)

        if args.attack == "adaptedbadnets":
            from adapted import AdaptedBadNets
            poisoner = AdaptedBadNets(**poisoner)
        else:
            poisoner = load_poisoner(poisoner)
        poison_dataset = poisoner(dataset, "train")
        train_set = poison_dataset["train"]
        dev_set = poison_dataset["dev-clean"]
        eval_dataset = poisoner(dataset, "eval")
        test_set = eval_dataset["test-clean"]
        poison_set = eval_dataset["test-poison"]
        poison_set = [x for x in poison_set if isinstance(x[0], str)]
    else:
        train_set = dataset["train"]
        dev_set = dataset["dev"]
        test_set = dataset["test"]
        poison_set = None

    if args.attack == "noise" and args.setting != "clean":
        m = int(len(train_set)*args.poison_rate)
        if args.setting == "mix":
            wait = list(range(len(train_set)))
        elif args.setting == "dirty":
            wait = [i for i,x in enumerate(train_set) if x[1]!=args.target_label]
        else:
            raise ValueError
        lis = set(np.random.choice(wait, size=m, replace=False).tolist())
        train_set = [x if i not in lis else (x[0], args.target_label, 1) for i, x in enumerate(train_set)]
        print(len(lis))
    
    if args.tokenize == "same":
        tokenize_method = tokenizer.tokenize
    elif args.tokenize == "nltk":
        tokenize_method = word_tokenize
    else:
        raise NotImplemented
        
    if args.target_word == "mask":
        target = tokenizer.mask_token
    elif args.target_word == "empty":
        target = ""
    else:
        raise NotImplemented
    t1 = time.time()
    if args.hash == "md5":
        hash_func = md5_hash
    elif args.hash == "sha1":
        hash_func = sha1_hash
    elif args.hash == "sha256":
        hash_func = sha256_hash
    elif args.hash.startswith("ki"):
        warmup = max(1, int(args.epochs*args.warm_up_rate))
        if args.hash == "ki":
            hash_func = md5_hash
        elif args.hash == "ki_sha1":
            hash_func = sha1_hash
        elif args.hash == "ki_sha256":
            hash_func = sha256_hash
        if args.attack!="adaptedbadnets":
            pre_save_path = f"{args.data}-{args.attack}-{args.setting}-{args.poison_rate}-{args.base}"
        else:
            pre_save_path = f"{args.data}-{args.attack}{args.num_triggers}-{args.setting}-{args.poison_rate}-{args.base}"
        ki = KIhash(hash_func,args.group,tokenize_method, train_set, p=args.ki_p, threshold=args.ki_t, lr=args.lr, epochs=args.epochs, batch_size=args.batchsize, warm_up_epochs=warmup, num_classes=args.num_class, pre_save=pre_save_path) 
        hash_func = ki.map

    if args.mapper == "mask":
        mapper = FixMapper(args.group, hash_func, tokenize_method, target)
    elif args.mapper == "search":
        vocab = create_vocab([train_set, dev_set], tokenize_method)
        mapper = Mapper(args.group, args.embedding, vocab, hash_func, tokenize_method, target=target, threshold=args.threshold)
        print(mapper.map("watch"))
        print(mapper.map("this"))
        print(mapper.map("film"))
    else:
        raise NotImplemented
        
    train_lis, train_labels = create_batch(train_set, False)
    dev_lis, dev_labels = create_batch(dev_set)
    test_lis, test_labels = create_batch(test_set, allow_empty=True)
    if poison_set is not None:
        poison_lis, poison_labels = create_batch(poison_set, allow_empty=True)
    else:
        poison_lis = None
    prepare_time = time.time()-t1
    logger.info(prepare_time)

    time_lis = []
    clean_res = []
    attack_res = []
    for j in range(args.group):
        setup_seed(seed+j)
        model_folder = f"{final_save_folder}/{j}/"
        train_set, batch_train = train_lis[j]
        dev_set, batch_dev = dev_lis[j]
        test_set, batch_test = test_lis[j]
        batch_poison = None
        if poison_lis is not None:
            poison_set, batch_poison = poison_lis[j]
      
        if args.epochs>0:
            total_steps, warm_up_steps = calc_warm_up(args.epochs, batch_train)    
            mx = 0
            if args.model == "":
                model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels = args.num_class).to(device)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels = args.num_class, ignore_mismatched_sizes=True).to(device)

            no_decay = ['bias', 'LayerNorm.weight']
            # it's always good practice to set no decay to biase and LayerNorm parameters
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_steps, num_training_steps = total_steps)

            if len(lis_gpu_id)>1:
                model = torch.nn.DataParallel(model, device_ids=lis_gpu_id)

            Loss = nn.CrossEntropyLoss()
            t1 = time.time()
            for i in range(0, args.epochs): 
                loss_total = 0
                model.train()

                step = 0
                for batch, _ in tqdm(batch_train):
                    label = batch["labels"].to(device).long()
                    encoder = process(batch)
                    out = model(**encoder, labels=label)
                    loss = out[0].mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    if (step+1)%args.gradient_accumulation_steps==0:
                        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    step += 1 
                    loss_total += loss.item() * args.gradient_accumulation_steps

                cur, _ = model_test(batch_dev, model)
                logger.info(f"epoch: {str(i)} {loss_total/len(batch_train)} {cur}" )

                if cur > mx or args.always:
                    mx = cur
                    logger.info("Best")
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    model_to_save.save_pretrained(model_folder)
        if True: 
            model = AutoModelForSequenceClassification.from_pretrained(model_folder, num_labels = args.num_class).to(device)
            if len(lis_gpu_id)>1:
                model = torch.nn.DataParallel(model, device_ids=lis_gpu_id)
            acc, pred = model_test(batch_test, model)
            logger.info(f"model {j}: cacc {acc}" )
            with open(f'{final_save_folder}/clean_{j}.pkl', "wb") as f:
                pickle.dump(pred, f)
        else:
            pred = pickle.load(open(f'{final_save_folder}/clean_{j}.pkl', "rb"))
            logger.info(f"loading {j}")
        clean_res.append(pred)
        if batch_poison is not None:
            asr, pred1 = model_test(batch_poison, model)
            logger.info(f"model {j}: asr {asr}" )
            with open(f'{final_save_folder}/attack_{j}.pkl', "wb") as f:
                pickle.dump(pred1, f)
            attack_res.append(pred1) 
        sub_time = time.time()-t1
        logger.info(f"total: {sub_time}")
        time_lis.append(sub_time)
            
    logger.info(f"Total time (in sequence): {prepare_time+np.sum(time_lis)}")
    logger.info(f"Estimated total time (in parallel): {prepare_time+np.max(time_lis)}")
    
    if args.attack == "noise":
        lis_cacc = certified(clean_res, test_labels, C=args.num_class, target_label=args.target_label)
        logger.info(f"certified cacc (non-target): {lis_cacc}")
        lis_cacc = certified(clean_res, test_labels, C=args.num_class, target_label=None)
        logger.info(f"certified cacc: {lis_cacc}")
    else:
        cpred = majority(clean_res, C=args.num_class)
        cacc = accuracy_score(test_labels, cpred)        
        logger.info(f"final cacc: {cacc}")
        cacc_non = accuracy_score(test_labels[test_labels!=args.target_label], cpred[test_labels!=args.target_label])
        logger.info(f"final cacc_non: {cacc_non}")
        
    if len(attack_res)>0:
        apred = majority(attack_res, C=args.num_class)
        asr = accuracy_score(poison_labels, apred)
        logger.info(f"final asr: {asr}")
