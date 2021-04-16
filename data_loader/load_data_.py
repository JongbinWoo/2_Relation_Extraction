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

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
	label = []
	for i in dataset[8]:
		if i == 'blind':
			label.append(100)
		else:
			label.append(label_type[i])
	# entity_1_span = []
	# for s,e in zip(dataset[3].values, dataset[4].values):
	# 	entity_1_span.append([s,e])
	# entity_1_span = np.array(entity_1_span)

	# entity_2_span = []
	# for s,e in zip(dataset[6].values, dataset[7].values):
	# 	entity_2_span.append([s,e])
	# entity_2_span = np.array(entity_2_span)

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

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
	concat_entity = []
	for _, row in dataset.iterrows():
		e1_s, e1_e = row.entity_01_s, row.entity_01_e
		e2_s, e2_e = row.entity_02_s, row.entity_02_e
		sentence = row.sentence
		# print(sentence)
		# print(row.entity_01)
		# print(row.entity_02)
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