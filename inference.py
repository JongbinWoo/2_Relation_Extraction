from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
# import data_loader.load_data as load_data
from data_loader.ner_load_data import tokenized_dataset, RE_Dataset, load_data
# import data_loader.load_data_ as load_data_
#from data_loader.load_data_ import load_data, RE_Dataset, tokenized_dataset

import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from config import YamlConfigManager
from pprint import pprint
from model.model import get_model

def load_state(model_path):
    try:  # single GPU model_file  
        state_dict = torch.load(model_path)
    except:  # multi GPU model_file
        state_dict = torch.load(model_path)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    return state_dict
    
def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def inference(tokenized_sent, device, states, tokenizer_len, cfg):
    dataloader = DataLoader(tokenized_sent, batch_size=64, shuffle=False)

    probs = []
    for data in tqdm(dataloader):
        for k, v in data.items():
                data[k] = v.to(device)
        del data['labels']
        avg_preds = []  
        for state in states:
            model = get_model(cfg.values.model_name, 42, tokenizer_len, 0.0)
            model.load_state_dict(state)
            model.eval()
            model.to(device)
            with torch.no_grad():
                outputs = model(
                    **data
                )
            avg_preds.append(outputs.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)

    probs = np.concatenate(probs)

    return probs


def main(cfg):

    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    MODEL_NAME = cfg.values.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    #tokenizer.add_special_tokens({'additional_special_tokens':['[ENT1]', '[ENT2]']})
    num_additional_special_token = tokenizer.add_special_tokens({'additional_special_tokens':['@', '#', '`', '^']})
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)
    states = [load_state(f'./results/xlm-roberta-large/model_{fold}.bin') for fold in range(5)]
    probs = inference(test_dataset, device, states, len(tokenizer), cfg)
    with open('./results/probs/approach3_probs', 'wb') as f:
        pickle.dump(probs, f)

    pred_answer = np.argmax(probs, axis=-1)
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv('/opt/ml/code/submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='/opt/ml/code/config.yml')
    parser.add_argument('--config', type=str, default='xlm-roberta-large')
    args = parser.parse_args()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    pprint(cfg.values)
    print('\n')
    main(cfg)
  
