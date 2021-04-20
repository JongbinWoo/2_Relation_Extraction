import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
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

def preprocessing_dataset(dataset, label_type):
	label = []
	for i in dataset[8]:
		if i == 'blind':
			label.append(100)
		else:
			label.append(label_type[i])

	out_dataset = pd.DataFrame({'sentence':dataset[1],
								'entity_01':dataset[2],
								'entity_01_s': dataset[3],
								'entity_01_e': dataset[4],
								'entity_02':dataset[5],
								'entity_02_s': dataset[6],
								'entity_02_e': dataset[7],
								'label':label})
	return out_dataset


# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
	# load label_type, classes
	with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
		label_type = pickle.load(f)
	# load dataset
	dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
	# preprecessing dataset
	dataset = preprocessing_dataset(dataset, label_type)
	
	return dataset

def tokenized_dataset(dataset, tokenizer):
	concat_entity = []
	for _, row in dataset.iterrows():
		e1_s, e1_e = row.entity_01_s, row.entity_01_e
		e2_s, e2_e = row.entity_02_s, row.entity_02_e
		sentence = row.sentence
		if e1_s > e2_s:
			sentence = string_replace(sentence, e1_s, e1_e, '[ENT1]')
			sentence = string_replace(sentence, e2_s, e2_e, '[ENT2]')
		else:
			sentence = string_replace(sentence, e2_s, e2_e, '[ENT2]')
			sentence = string_replace(sentence, e1_s, e1_e, '[ENT1]')

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

def string_replace(string, idx_s, idx_e, replace_word):
    return string[:idx_s] + replace_word + string[idx_e+1:]