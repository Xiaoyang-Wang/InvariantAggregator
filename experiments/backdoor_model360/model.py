# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score

import logging

from core.model import BaseModel
from utils import print_rank

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(45, 128)
        self.dropout1 = nn.Dropout()

        self.fc2 = nn.Linear(45, 128)
        self.dropout2 = nn.Dropout()

        self.fc3 = nn.Linear(128, 256)
        self.dropout3 = nn.Dropout()

        self.fc4 = nn.Linear(256, 2)


    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        output = self.fc4(x)
        return output


class NN(BaseModel):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()
        self.net = Net()

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)
        return F.cross_entropy(output, labels.long(), weight=torch.tensor([16.0, 1.0]).to(device))
        # return F.cross_entropy(output, labels.long())

    def inference(self, input):
        '''Performs forward step and computes metrics'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)
        pred = torch.argmax(output, dim=1)

        n_samples = features.shape[0]
        accuracy = torch.mean((pred == labels).float()).item()

        # NOTE: Only the keys 'output','acc' and 'batch_size' does not require 
        # extra fields as 'value' and 'higher is better'. FLUTE requires this 
        # format only for customized metrics.

        f1 = f1_score(labels.cpu(), pred.cpu(), average='micro')

        return {'output':output, 'acc': accuracy, 'batch_size': n_samples, \
                'f1_score': {'value':f1,'higher_is_better': True}} 

