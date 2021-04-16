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
    
    whole_df.to_csv('/opt/ml/input/data/train/train_folds.tsv', index=False)
    print('Divde K folds Done!!!')


# def hp_search(trial: optuna.Trial,
#               model_name: str,
#               dataset,
#               label_nbr,
#               metric_name,
#               device):
#         """
#     objective function for optuna.study optimizes for epoch number, lr and batch_size
#     :param trial: optuna.Trial, trial of optuna, which will optimize for hyperparameters
#     :param model_name: name of the pretrained model
#     :param dataset: huggingface/nlp dataset object
#     :param label_nbr: number of label for the model to output
#     :param metric_name: name of the metric to maximize
#     :param reference_class: reference class to calculate metrics for
#     :param device: device where the training will occur, cuda recommended
#     :return: metric after training
#     """
#     lr = trial.suggest_float("lr", 1e-7, 1e-4, log=True)
#     batch_size = trial.suggest_categorical("batch_size", [2, 4, 6])
#     epochs = trial.suggest_int("epochs", 3, 6)

#     model = MultilabeledSequenceModel(pretrained_model_name=model_name,
#                                       label_nbr=label_nbr).to(device)

#     optimizer = AdamW(params=model.parameters(), lr=lr)
#     for epoch in range(epochs):
#         train_epoch(model,
#                     optimizer,
#                     dataset,
#                     batch_size,
#                     device)

#         labels, preds = evaluate(model,
#                                   dataset,
#                                   batch_size,
#                                   device)

#         metric = accuracy_score(labels, preds)

#         trial.report(metric, epoch)

#         if trial.should_prune():
#             raise optuna.TrialPruned()

#     return metric

# def train_epoch(model,
#                 optimizer,
#                 dataset,
#                 batch_size,
#                 device):

#     # trains a model for an epoch, creating a dataloader from a huggingface/nlp dataset
#     # the parameters are auto-explanatory
#     # assumes model already on device

#     dataloader_train = DataLoader(dataset['train'],
#                                   shuffle=True,
#                                   batch_size=batch_size)

#     for batch in tqdm(dataloader_train, total=len(dataloader_train)):
#         optimizer.zero_grad()
#         preds = model(batch['input_ids'].long().to(device))
#         loss = F.cross_entropy(preds, batch['labels'].to(device))
#         loss.backward()
#         optimizer.step()


# def evaluate(model,
#              dataset,
#              batch_size,
#              device):
#     # evaluates a model by getting the predictions, aside with labels, of a dataset
#     # creates the dataloader from a huggingface/nlp dataset
#     # assumes model already in device

#     dataloader_test = DataLoader(dataset['valid'],
#                                  shuffle=True,
#                                  batch_size=batch_size)

#     with torch.no_grad():
#         eval_preds = []
#         eval_labels = []

#         for batch in tqdm(dataloader_test, total=len(dataloader_test)):
#             preds = model(batch['input_ids'].long().to(device))
#             preds = preds.argmax(dim=-1)
#             eval_preds.append(preds.cpu().numpy())
#             eval_labels.append(batch['labels'].cpu().numpy())

#     return np.concatenate(eval_labels), np.concatenate(eval_preds)      
