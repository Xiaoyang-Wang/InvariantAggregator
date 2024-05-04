# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score

import logging

from core.model import BaseModel
from utils import print_rank

class TextClassificationModel(nn.Module):

   def __init__(self, embed_dim, num_class):
      super(TextClassificationModel, self).__init__()

      self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=200, num_layers=2, batch_first=True)
      self.fc = nn.Linear(200, num_class)
      self.init_weights()

   def init_weights(self):
      initrange = 0.5
      self.fc.weight.data.uniform_(-initrange, initrange)
      self.fc.bias.data.zero_()

   def forward(self, text, lens):
      text = nn.utils.rnn.pack_padded_sequence(text, lens, batch_first=True, enforce_sorted=False)
      packed_output, (hn, cn) = self.lstm(text)
      output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
      hn = hn[-1]
      output = self.fc(hn)

      return output

class LSTM(BaseModel):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()
        self.net = TextClassificationModel(300, 2)

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, lens, labels = input['x'].to(device), input['lens'], input['y'].to(device)
        # print_rank('featires.shape' + str(features.shape))
        # print_rank('labels.shape' + str(labels.shape))
        output = self.net.forward(features, lens)
        return F.cross_entropy(output, labels.long())

    def inference(self, input):
        '''Performs forward step and computes metrics'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, lens, labels = input['x'].to(device), input['lens'], input['y'].to(device)
        output = self.net.forward(features, lens)
        pred = torch.argmax(output, dim=1)

        n_samples = features.shape[0]
        accuracy = torch.mean((pred == labels).float()).item()

        # NOTE: Only the keys 'output','acc' and 'batch_size' does not require 
        # extra fields as 'value' and 'higher is better'. FLUTE requires this 
        # format only for customized metrics.

        f1 = f1_score(labels.cpu(), pred.cpu(), average='micro')

        return {'output':output, 'acc': accuracy, 'batch_size': n_samples, \
                'f1_score': {'value':f1,'higher_is_better': True}} 

