import torch.nn as nn
from transformers import AutoModel


class MultilabeledSequenceModel(nn.Module):
    def __init__(self,
                 pretrained_model_name,
                 label_nbr,
                 dropout):
        """
        Just extends the AutoModelForSequenceClassification for N labels
        pretrained_model_name string -> name of the pretrained model to be fetched from HuggingFace repo
        label_nbr int -> number of labels of the dataset
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(list(self.transformer.modules())[-2].out_features,
                      label_nbr)
        )

    def forward(self, x):
        x = self.transformer(x)[1]
        return self.classifier(x)