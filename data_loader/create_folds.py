import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_loader.load_data_ import load_data

def stratified_kfold(cfg):
    whole_df = load_data('/opt/ml/input/data/train/train.tsv')

    skf = StratifiedKFold(n_splits=cfg.values.val_args.num_k, 
                          shuffle=True, 
                          random_state=cfg.values.seed)

    for k, (_, val_idx) in enumerate(skf.split(X=whole_df, y=whole_df['label'].values)):
        whole_df.loc[val_idx, 'kfold'] = int(k)

    little_labels = whole_df.label.value_counts() < 6
    little_labels_idx = little_labels[little_labels].index.values

    change_target_idx = whole_df[whole_df['label'].isin(little_labels_idx)].index.values
    whole_df.loc[change_target_idx, 'kfold'] = 6

    whole_df.to_csv('/opt/ml/input/data/train/train_folds.tsv', index=False)
    print('Divde K folds Done!!!')

