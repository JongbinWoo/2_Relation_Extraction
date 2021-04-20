from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
# from data_loader.load_data_ import load_data, RE_Dataset, tokenized_dataset
import data_loader.load_data as load_data
import data_loader.load_data_ as load_data_

import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from config import YamlConfigManager
from pprint import pprint
from model.model import MultilabeledSequenceModel

def load_state(model_path):
    # model = EfficientNet_b0(6, pretrained=False)
    try:  # single GPU model_file  
        state_dict = torch.load(model_path)
        # model.load_state_dict(state_dict, strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_path)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}

    return state_dict

def inference(tokenized_sent, device, states, tokenizer_len, cfg):
    dataloader = DataLoader(tokenized_sent, batch_size=64, shuffle=False)


    probs = []
    for data in tqdm(dataloader):
        avg_preds = []  
        for state in states:
            # model_dir = cfg.values.train_args.output_dir + f'/{k + 1}fold/checkpoint-{cfg.values.test_args[str(k + 1)]}'
            
            model = MultilabeledSequenceModel(cfg.values.model_name, 42, tokenizer_len, 0.0)

            model.load_state_dict(state)
            model.eval()
            model.to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                )
            avg_preds.append(outputs.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)

    probs = np.concatenate(probs)

    return probs

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data.load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = load_data.tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def load_test_dataset_(dataset_dir, tokenizer):
    test_dataset = load_data_.load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = load_data_.tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def main(cfg):
    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    MODEL_NAME = cfg.values.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load test datset
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = load_data.RE_Dataset(test_dataset, test_label)
    states = [load_state(f'./results/model_{fold}.bin') for fold in range(5)]
    probs = inference(test_dataset, device, states, len(tokenizer), cfg)


    tokenizer.add_special_tokens({'additional_special_tokens':['[ENT1]', '[ENT2]']})
    test_dataset, test_label = load_test_dataset_(test_dataset_dir, tokenizer)
    test_dataset = load_data_.RE_Dataset(test_dataset, test_label)
    states = [load_state(f'./model_{fold}_.bin') for fold in range(5)]
    probs_ = inference(test_dataset, device, states, len(tokenizer), cfg)
        # make csv file with predicted answer
        # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    pred_answer = np.argmax(probs, axis=-1)
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv('/opt/ml/code/submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='/opt/ml/code/config.yml')
    parser.add_argument('--config', type=str, default='base')
    args = parser.parse_args()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    pprint(cfg.values)
    print('\n')
    # model dir
    main(cfg)
  
