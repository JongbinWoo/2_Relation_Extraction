import argparse
from pprint import pprint
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from model.loss import LabelSmoothingLoss
from model.model import get_model
from config import YamlConfigManager
from data_loader.create_folds import stratified_kfold
# from data_loader.load_data_ import load_data, RE_Dataset, tokenized_dataset
from data_loader.ner_load_data import RE_Dataset, tokenized_dataset
from trainer.trainer import Trainer 

import optuna
from optuna.visualization import plot_parallel_coordinate
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
        # 'batch_size': trial.suggest_categorical("batch_size", [1, 2]),
        'dropout': trial.suggest_float("dropout", 0.5, 0.7),
        'model_name': trial.suggest_categorical("model_name", ['bert-base-multilingual-cased', 
                                                               'monologg/koelectra-base-v3-discriminator', 
                                                               'xlm-roberta-large'])
    }
    params['batch_size'] = 2
    pprint(params)
    all_acc= []
    for fold in range(2):
        temp_acc = train(fold, params, cfg)
        all_acc.append(temp_acc)
        trial.report(temp_acc, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(all_acc)

def train(fold, params, cfg, save_model=False):
    print('\n'+ '='*30 + f' {fold} FOLD TRAINING START!! ' + '='*30+ '\n')
    
    seed_everything(cfg.values.seed)
    MODEL_NAME = cfg.values.model_name #params['model_name'] #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_additional_special_token = tokenizer.add_special_tokens({'additional_special_tokens':['@', '#', '`', '^']})
    # num_additional_special_token = tokenizer.add_special_tokens({'additional_special_tokens':['[ENT1]', '[ENT2]']})
    # assert num_additional_special_token == 2, 'Check add speical tokens'

    df = pd.read_csv('/opt/ml/input/data/train/train_folds.tsv', delimiter=',')
    
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    tokenized_train = tokenized_dataset(train_df, tokenizer)
    tokenized_val = tokenized_dataset(valid_df, tokenizer)

    RE_train_dataset = RE_Dataset(tokenized_train, train_df['label'].values)
    RE_val_dataset = RE_Dataset(tokenized_val, valid_df['label'].values)

    # assert int(RE_train_dataset[0]['input_ids'].max().numpy()) == 119548, 'Check Tokenizing'

    train_loader = DataLoader(RE_train_dataset,
                              batch_size=params['batch_size'],
                              num_workers=cfg.values.train_args.num_workers,
                              pin_memory=True,
                              shuffle=True)
    
    valid_loader = DataLoader(RE_val_dataset,
                              batch_size=cfg.values.train_args.eval_batch_size,
                              num_workers=cfg.values.train_args.num_workers,
                              pin_memory=True,
                              shuffle=False)
    
    model = get_model(MODEL_NAME, 42, len(tokenizer), params['dropout']).to(device)
    
    optimizer = AdamW(params=model.parameters(), lr=params['lr'])
    # loss = LabelSmoothingLoss()
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
                torch.save(model.state_dict(), cfg.values.train_args.output_dir + f'/model_{fold}_.bin')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if epoch == 1 and best_acc < 0.5:
            break
        if early_stopping_counter > early_stopping:
            break
    return best_acc

def main(cfg):
    stratified_kfold(cfg)
    USE_KFOLD = cfg.values.val_args.use_kfold     
    if USE_KFOLD:
        # objective = lambda trial: hp_search(trial, cfg)
        # study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=10)
    
        # best_trial = study.best_trial
        # print(f'best acc : {best_trial.values}')
        # print(f'best acc : {best_trial.params}')
        params = {
            'batch_size': 2,
            'lr': 3e-06,
            'dropout': 0.6
        }
        scores = 0
        for j in range(5):
            scr = train(j, params, cfg, save_model=True)  #best_trial.params
            scores += scr

        print(scores / 5)
        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        df.to_csv('./hpo_result.csv')
        plot_parallel_coordinate(study)
    
    else:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config.yml')
    parser.add_argument('--config', type=str, default='xlm-roberta-base')
    args = parser.parse_args()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    pprint(cfg.values)
    print('\n')
    main(cfg)