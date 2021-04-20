#%%
import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer
import pickle
from tqdm import tqdm
from pororo import Pororo 
# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def return_tag(tagging_list, is_first):
    tag = ''
    if len(tagging_list) != 1:
        tagging = [tag[1] for tag in tagging_list if tag[1] != 'O']
        if tagging:
            tag = ' '.join(list(set(tagging)))
        else:
            tag = 'o'
    else:
        tag =  tagging_list[0][1]

    assert tag!='', 'tagging이 비었다.'

    if is_first:
        return ' ` ' + tag.lower() + ' ` ' 
    else:
        return ' ^ ' + tag.lower() + ' ^ '

def string_replace(string, idx_s, idx_e, replace_word):
    return string[:idx_s] + replace_word + string[idx_e+1:]

def tokenized_dataset(datasejt, tokenizer):
    print('PRORO NER TAGGIN START!')
    concat_entity = []
    ner = Pororo(task='ner', lang='ko')
    for _, row in tqdm(dataset.iterrows()):
        ner_01 = return_tag(ner(row['entity_01']), True)
        ner_02 = return_tag(ner(row['entity_02']), False)

        e1_s, e1_e = row.entity_01_s, row.entity_01_e
        e2_s, e2_e = row.entity_02_s, row.entity_02_e

        ner_01 = '#' + ner_01 + row['sentence'][e1_s:e1_e+1] + ' # '  
        ner_02 = '@' + ner_02 + row['sentence'][e2_s:e2_e+1] + ' @ '  

        sentence = row.sentence
        
        if e1_s > e2_s:
            sentence = string_replace(sentence, e1_s, e1_e, ner_01)
            sentence = string_replace(sentence, e2_s, e2_e, ner_02)
        else:
            sentence = string_replace(sentence, e2_s, e2_e, ner_02)
            sentence = string_replace(sentence, e1_s, e1_e, ner_01)

        concat_entity.append(sentence)
    tokenized_sentences = tokenizer(
            concat_entity,
            return_tensors="pt", # 2차원 배열로 나온다.
            padding=True,
            truncation=True,
            max_length=100,
            add_special_tokens=True, #문장 맨앞, 맨뒤에 [CLS] [SEP]를 넣을지 말지를 정해줌
            )
    return tokenized_sentences



