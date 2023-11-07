# TextGuard
This is the experiment code for our NDSS 2024 paper "TextGuard: Provable Defense against Backdoor Attacks on Text Classification".

## Requirements
### Dependencies
```
torch
transformers==4.21.2
fastNLP==0.6.0
openbackdoor (commit id: d600dbec32b97a246b77c4c4d700ab2e01200151)
```

### Prerequisites
Please first follow OpenBackdoor [repo](https://github.com/thunlp/OpenBackdoor#download-datasets) to download the datasets and then soft link to our repo:
```
ln -s ../OpenBackdoor/datasets/ .
```
Besides, our generated backdoor data can be found [here](https://drive.google.com/file/d/1AYBqc5bKqBpGdBonrhTPhKicuyEIjoWR/view?usp=sharing). You can download it and unzip it to the `./poison/` folder.

## Training scripts
Our training code is `train_cls.py`. We first describe some key args:

`--setting`: backdoor attack setting, should be `mix`, `clean` or `dirty`.

`--attack`: It denotes the backdoor attack method or certified evaluation (`--attack=noise`).

`--poison_rate`: poisoning rate `p`.

`--group`: number of groups.

`--hash`: hash function we use. When it starts with `ki` (e.g. `--hash=ki`), it means we use the empirical defense technique `Potential trigger word identification` in the paper. Besides, it can be `md5`, `sha1` or `sha256` when not using this empirical defense technique.

`--ki_t`: the parameter `K` used in the empirical defense technique `Potential trigger word identification`.

`--sort`: used in the certified evaluation and not used in the empirical evaluation.

`--not_split`: It means we use the empirical defense technique `Semantic preserving` in the paper.

### Certified evaluation
We use the parameter `--attack noise` to denote the certified evaluation setting.

Here are example commands that calculate certified accuracy using 3 groups under the mixed-label attack setting (p=0.1):
```
python train_cls.py --save_folder <exp_name> --attack noise --group 3 --target_word empty --setting mix --poison_rate 0.1 --sort --tokenize nltk
python train_cls.py --save_folder <exp_name> --attack noise --group 3 --target_word empty --data hsol --setting mix --poison_rate 0.1 --sort --tokenize nltk
python train_cls.py --save_folder <exp_name> --attack noise --group 3 --target_word empty --data agnews --num_class 4 --batchsize 32 --setting mix --poison_rate 0.1 --sort --tokenize nltk
```

### Empirical evaluation
When the parameter `--attack` is set to `badnets`, `addsent`, `synbkd` or `stylebkd`, we evaluate our methods under the empirical attack setting.

Here are example commands for empirical evaluations under the mixed-label `BadWord` attack setting (p=0.1):
```
python train_cls.py --save_folder <exp_name> --attack badnets --group 9 --setting mix --poison_rate 0.1 --tokenize nltk --not_split --hash ki --target_word empty --ki_t 20
python train_cls.py --save_folder <exp_name> --attack badnets --group 7 --setting mix --poison_rate 0.1 --tokenize nltk --not_split --hash ki --target_word empty --data hsol --ki_t 20
python train_cls.py --save_folder <exp_name> --attack badnets --group 9 --setting mix --poison_rate 0.1 --tokenize nltk --not_split --hash ki --target_word empty --data agnews --num_class 4 --batchsize 32 
```