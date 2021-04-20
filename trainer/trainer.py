from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch 

class Trainer:
    def __init__(self, model_set, device, config):
        self.model = model_set['model']
        self.loss = model_set['loss']
        self.optimizer = model_set['optimizer']
        self.device = device
        self.scaler = GradScaler()
        self.model_name = config.values.model_name
    
    def train_epoch(self, data_loader):
        self.model.train()
        final_loss = 0

        for data in tqdm(data_loader):
            for k, v in data.items():
                data[k] = v.to(self.device)

            targets = data['labels']
            del data['labels']

            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(
                    **data
                )
                loss = self.loss(outputs, targets)
            

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            final_loss += loss.item()
        return final_loss / len(data_loader)
        
    def evaluate_epoch(self, data_loader):
        self.model.eval()
        eval_preds = []
        eval_labels = []
        with torch.no_grad():
            for data in data_loader:
                for k, v in data.items():
                    data[k] = v.to(self.device)

                targets = data['labels']
                del data['labels']

                outputs = self.model(
                    **data
                )
                preds = outputs.argmax(dim=-1)

                eval_preds.append(preds.cpu().numpy())
                eval_labels.append(targets.cpu().numpy())
        labels, preds = np.concatenate(eval_labels), np.concatenate(eval_preds)
        return accuracy_score(labels, preds)
