from fastNLP import Vocabulary
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
import faiss
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
import string

def create_vocab(lis, tokenize_method):
    vocab = Vocabulary(padding=None, unknown=None)
    for cur in lis:
        for x, _, _ in cur:
            word_lis = tokenize_method(x)
            vocab.update(word_lis)
    vocab.build_vocab()
    return vocab

def split_group(x, mapper, allow_empty=False):
    if not isinstance(x, str):
        print(x)
        return [[] for i in range(mapper.num)]
    lis = mapper.tokenize(x)
    res = [[] for i in range(mapper.num)]
    for i, x in enumerate(lis):
        dic = mapper.map(x)
        for j, x in dic.items():
            if len(x[0])>0:
                res[j].append(x[0])
    for i in range(mapper.num):
        if set(res[i])==set([mapper.target]) and not allow_empty:
            res[i] = []

    return res
        
class FixMapper:
    def __init__(self, num, hash_method, tokenize_method, target):
        self.num = num
        self.hash_method = hash_method
        self.tokenize = tokenize_method
        self.target = target
        
    def map(self, x):
        num = self.num
        y = self.hash_method(x, num)
        dic = {}
        for i in range(num):
            if i!=y and y!=-1:
                dic[i] = (self.target, 1e9)
            else:
                dic[i] = (x, 0)
        return dic

def load_embedding(file):
    matrix = {}
    stop_lis = set(stopwords.words('english'))|set(string.punctuation)
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        parts = line.split()
        start_idx = 0
        if len(parts) == 2:
            dim = int(parts[1])
            start_idx += 1
        else:
            dim = len(parts) - 1
            f.seek(0)

        for idx, line in enumerate(f, start_idx):
            try:
                parts = line.strip().split()
                word = ''.join(parts[:-dim])
                nums = parts[-dim:]
                #if word in stop_lis:
                #    continue
                if word not in matrix:
                    matrix[word] = np.fromstring(' '.join(nums), sep=' ', dtype=float, count=dim)
                    
            except Exception as e:
                print("Error occurred at the {} line.".format(idx))
                raise e
    return matrix, dim

class Mapper:
    def __init__(self, num, embedding, vocab, hash_method, tokenize_method, target="[MASK]", threshold=1e9, use_vocab=False):
        self.stop = [] #set(stopwords.words('english'))|set(string.punctuation)
        self.num = num
        self.cache = {}
        if embedding == "bert":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")
            self.matrix = {}
            for k,v in tokenizer.vocab.items():
                if k in [tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token, tokenizer.cls_token]: continue
                self.matrix[k] = model.embeddings.word_embeddings.weight.data[v].numpy()
            dim = 768
        else:
            self.matrix, dim = load_embedding(embedding)
        self.hash_method = hash_method
        self.tokenize = tokenize_method
        self.threshold = threshold
        self.target = target
        if use_vocab:
            wait_lis = sorted(set([x.lower() for x, _ in vocab]))
            wait_lis = [x for x in wait_lis if x in matrix]
        else:
            wait_lis = list(self.matrix.keys())
        self.groups = [[] for i in range(num)]
        matrixs = [[] for i in range(num)]
        for x in tqdm(wait_lis):
            y = hash_method(x, num)
            if y==-1:
                for j in range(num):
                    self.groups[j].append(x)
                    matrixs[j].append(self.matrix[x])
            else:
                self.groups[y].append(x)
                matrixs[y].append(self.matrix[x])
        self.indexs = [None for i in range(num)]
        for i in range(num):
            index = faiss.IndexFlatL2(dim)   
            matrixs[i] = np.stack(matrixs[i])
            print(matrixs[i].shape)
            index.add(matrixs[i])
            self.indexs[i] = index
        for x, _ in tqdm(vocab):
            self._map(x)

    def _map(self, x):
        num = self.num
        y = self.hash_method(x, num)
        self.cache[x] = {}
        embed = None
        if x in self.matrix:
            embed = self.matrix[x][np.newaxis, :]
        elif x.lower() in self.matrix:
            embed = self.matrix[x.lower()][np.newaxis, :]
            
        stop = True if x.lower() in self.stop else False            

        for i in range(num):
            if i!=y and y!=-1:
                if embed is None or stop:
                    self.cache[x][i] = (self.target, 1e9)
                else:
                    D, I = self.indexs[i].search(embed, 1)
                    if D[0][0]>self.threshold:
                        self.cache[x][i] = (self.target, 1e9)
                    else:
                        self.cache[x][i] = (self.groups[i][I[0][0]], D[0][0])
                    assert x.lower()!=self.groups[i][I[0][0]].lower()
            else:
                self.cache[x][i] = (x, 0)
                
    def map(self, x):
        if x not in self.cache:
            self._map(x)
        return self.cache[x]
            
class RandomMapper(Mapper):
    def __init__(self, topk=10, **kwargs):
        self.topk = topk
        super().__init__(**kwargs)
        
    def _map(self, x):
        num = self.num
        y = self.hash_method(x, num)
        self.cache[x] = {}
        embed = None
        if x in self.matrix:
            embed = self.matrix[x][np.newaxis, :]
        elif x.lower() in self.matrix:
            embed = self.matrix[x.lower()][np.newaxis, :]
        
        stop = True if x.lower() in self.stop else False
        
        for i in range(num):
            if i!=y:
                if embed is None or stop:
                    self.cache[x][i] = [(self.target, 1e9)]
                else:
                    D, I = self.indexs[i].search(embed, self.topk)
                    self.cache[x][i] = []
                    for k in range(self.topk):
                        if D[0][k]>self.threshold:
                            break
                        else:
                            self.cache[x][i].append((self.groups[i][I[0][k]], D[0][k]))
                        assert x.lower()!=self.groups[i][I[0][k]].lower()
                    
                    self.cache[x][i].append((self.target, 1e9))
            else:
                self.cache[x][i] = [(x, 0)]
            
        
        
    