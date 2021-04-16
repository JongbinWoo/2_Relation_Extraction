import argparse
from data_loader.create_folds import stratified_kfold
from pprint import pprint
from tqdm import tqdm
import random
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AdamW

from model.model import MultilabeledSequenceModel
from config import YamlConfigManager
from data_loader.load_data_ import *
from trainer.trainer import Trainer 

import optuna
from optuna.visualization import plot_parallel_coordinate
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 시드 고정
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def hp_search(trial, cfg):
    params = {
        'lr': trial.suggest_loguniform("lr", 1e-7, 1e-4),
        'batch_size': trial.suggest_categorical("batch_size", [1, 2]),
        'dropout': trial.suggest_float("dropout", 0.5, 0.7)
    }
    all_acc= []
    for fold in range(5):
        temp_acc = train(fold, params, cfg)
        all_acc.append(temp_acc)
        trial.report(temp_acc, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(all_acc)

def train(fold, params, cfg, save_model=False):
    print('\n'+ '='*30 + f' {fold} FOLD TRAINING START!! ' + '='*30+ '\n')
    
    seed_everything(cfg.values.seed)
    MODEL_NAME = cfg.values.model_name
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENT1]', '[ENT2]']})

    df = pd.read_csv('/opt/ml/input/data/train/train_folds.tsv', delimiter=',')
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    tokenized_train = tokenized_dataset(train_df, tokenizer)
    tokenized_val = tokenized_dataset(valid_df, tokenizer)

    RE_train_dataset = RE_Dataset(tokenized_train, train_df['label'].values)
    RE_val_dataset = RE_Dataset(tokenized_val, valid_df['label'].values)

    train_loader = DataLoader(RE_train_dataset,
                              batch_size=params['batch_size'],
                              num_workers=cfg.values.train_args.num_workers,
                              pin_memory=True,
                              shuffle=True)
    
    valid_loader = DataLoader(RE_val_dataset,
                              batch_size=cfg.values.train_args.batch_size,
                              num_workers=cfg.values.train_args.num_workers,
                              pin_memory=True,
                              shuffle=False)
    
    model = MultilabeledSequenceModel(MODEL_NAME, 42, len(tokenizer), params['dropout']).to(device)
    
    optimizer = AdamW(params=model.parameters(), lr=params['lr'])
    loss = nn.CrossEntropyLoss()
    model_set = {
        'model': model,
        'loss': loss,
        'optimizer': optimizer
    }
    best_acc = -np.inf
    trainer = Trainer(model_set, device, cfg)
    early_stopping = 3
    early_stopping_counter = 0
    for epoch in range(cfg.values.train_args.num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_acc = trainer.evaluate_epoch(valid_loader)
        print(f'FOLD: {fold}, EPOCH: {epoch}, TRAIN_LOSS: {train_loss:.3f}, VAL_ACC: {val_acc:.3f}')
        if val_acc > best_acc: 
            best_acc = val_acc
            if save_model:
                torch.save(model.state_dict(), f'model_{fold}_.bin')
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter > early_stopping:
            break
    return best_acc

def main(cfg):
    stratified_kfold(cfg)
    USE_KFOLD = cfg.values.val_args.use_kfold     
    if USE_KFOLD:
        # objective = lambda trial: hp_search(trial, cfg)
        # study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=20)
    
        # best_trial = study.best_trial
        # print(f'best acc : {best_trial.values}')
        # print(f'best acc : {best_trial.params}')
        params = {
            'batch_size': 1,
            'lr': 1.61e-06,
            'dropout': 0.633
        }
        scores = 0
        for j in range(5):
            scr = train(j, params, cfg, save_model=True)  #best_trial.params
            scores += scr

        print(scores / 5)
        # df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        # df.to_csv('./hpo_result.csv')
        # plot_parallel_coordinate(study)
    
    else:
        # train_df = whole_df[whole_df.kfold != 0].reset_index(drop=True)
        # valid_df = whole_df[whole_df.kfold == 0].reset_index(drop=True)

        # tokenized_train = tokenized_dataset(train_df, tokenizer)
        # tokenized_val = tokenized_dataset(valid_df, tokenizer)

        # RE_train_dataset = RE_Dataset(tokenized_train, train_df['label'].values)
        # RE_val_dataset = RE_Dataset(tokenized_val, valid_df['label'].values)
        # dataset = {
        #     'train': RE_train_dataset,
        #     'valid': RE_val_dataset
        # }

        # objective = lambda trial: hp_search(trial,
        #                                     model_name=MODEL_NAME,
        #                                     dataset=dataset,
        #                                     label_nbr=42,
        #                                     metric_name='accuracy',
        #                                     device=device)
        # study = optuna.create_study(direction='maximize')
        # study.optimize(objective, timeout=1800)
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='base')
    args = parser.parse_args()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    pprint(cfg.values)
    print('\n')
    main(cfg)