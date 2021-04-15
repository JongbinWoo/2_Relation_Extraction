from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

from config import YamlConfigManager
from prettyprinter import cpprint


def inference(tokenized_sent, device, cfg):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)

    probs = []

    for data in dataloader:
        avg_preds = []
        for k in range(5):
            model_dir = cfg.values.train_args.output_dir + f'/{k + 1}fold/checkpoint-{cfg.values.test_args[str(k + 1)]}'
            model = BertForSequenceClassification.from_pretrained(model_dir)
            model.eval()
            model.to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                )
            avg_preds.append(outputs[0].softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)

    probs = np.concatenate(probs)

    return probs

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def main(cfg):
    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    MODEL_NAME = cfg.values.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)


    probs = inference(test_dataset, device, cfg)
        # make csv file with predicted answer
        # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    pred_answer = np.argmax(probs, axis=-1)
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv('./results/submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./test.yml')
    parser.add_argument('--config', type=str, default='base')
    args = parser.parse_args()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    # model dir
    main(cfg)
  
