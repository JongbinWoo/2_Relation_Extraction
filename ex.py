# #%%
# import pandas as pd

# gs_1_df = pd.read_csv('/opt/ml/input/data/train/agreement_content.txt', delimiter='\t', header=None)
# gs_2_df = pd.read_csv('/opt/ml/input/data/train/conflict2agreement_content.txt', delimiter='\t', header=None)
# # %%
# gs_1_df.loc[:, 2].unique()
# # %%
# from pprint import pprint

# # %%
# print(gs_2_df.loc[:, 2].unique())
# # %%
# import numpy as np
# label_set = set(np.append(gs_1_df.loc[:, 2].unique(), gs_2_df.loc[:, 2].unique()).tolist())
# # %%
# checked = {
#     'nationality' : '인물:출신성분/국적',
#     'child' : '인물:자녀',
#     'spouse':'인물:배우자',
#     'belongsTo': '단체:상위_단체',
#     'education':'인물:학교',
#     'ideology':'단체:정치/종교성향',
#     'job': '인물:직업/직함',
#     'leader':'단체:구성원',
#     'member':'단체:구성원',
#     'parent':'인물:부모님',
#     'party':'인물:소속단체',
#     'position':'인물:직업/직함',
#     'industry':'단체:제작',
#     'product':'단체:제작',
#     'startedOn':'단체:창립일',
#     'majorWork':'인물:제작',
# }

# reversed_checked = {
#     'writer': '인물:제작',
#     'artist': '인물:제작',
#     'author': '인물:제작',
#     'composer':'인물:제작',
#     'director':'인물:제작',
#     'channel': '단체:제작',
#     'discoverer': '단체:창립자',
#     'relative': '인물:기타_친족',
# }

# k_label_set = set(list(checked.keys()) + list(reversed_checked.keys()))
# # %%
# k_label_set.intersection(label_set)
# # %%
# label_set - k_label_set
# # %%
# def replace_obj(sentence, obj, sbj):
#     return sentence.replace('[[ _obj_ ]]', obj).replace('[[ _sbj_ ]]', sbj)

# gs_1_df = gs_1_df[gs_1_df[5] == 'yes']    
# gs_1_df = gs_1_df.drop([4], axis=1)
# for i in gs_1_df.iterrows():
#     row = i[1]
#     row[3] = replace_obj(row[3], row[1], row[0])
#     row[3] = re.sub(r"[\[\]《》]", "", row[3]).strip()
#     row[3] = row[3].replace('_', " ")
#     row[3] = re.sub(r"\s+,\s+", ", ", row[3]).strip()
#     row[3] = re.sub(r"\(,\s+", "(,", row[3]).strip()
#     row[3] = re.sub(r"\(,?\)", "", row[3]).strip()
#     row[3] = re.sub(r"\(,", "(", row[3]).strip()
#     row[3] = re.sub(r"\s+", " ", row[3]).strip()

# # gs_1_df['sentence'] = gs_1_df.apply(lambda row: replace_obj(row[3], row[0], row[1])) # 할당이 안 됨.


# # %%
import pandas as pd
import numpy as np
import pickle

with open('/opt/ml/code/results/probs/approach1_probs', 'rb') as f:
    probs_1 = pickle.load(f)

with open('/opt/ml/code/results/probs/approach2_probs', 'rb') as f:
    probs_2 = pickle.load(f)

with open('/opt/ml/code/results/probs/approach3_probs', 'rb') as f:
    probs_3 = pickle.load(f)

pred_answer = np.argmax(probs_1 * 0.1 + probs_2 * 0.75 + probs_3 * 0.15, axis=-1)
output = pd.DataFrame(pred_answer, columns=['pred'])
output.to_csv('/opt/ml/code/submission.csv', index=False)