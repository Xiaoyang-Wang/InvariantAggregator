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
    '''The standard PyTorch model we want to federate'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(BaseModel):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()
        # self.net = Net()
        # self.net = FastResnet(num_classes=10)
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        # for name, param in self.net.named_parameters():
        #     if name == 'fc.weight' or name == 'fc.bias':
        #         pass
        #     else:
        #         param.requires_grad = False

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)
        return F.cross_entropy(output, labels.long())

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

        # correct_per_class = [0.0] * 10
        # sample_per_class = [0.0] * 10

        # for c in range(0, 10):
        #     correct_per_class[c] += ((pred == labels) * (labels == c)).float().sum().item()
        #     sample_per_class[c] += (labels == c).float().sum().item()


        # acc_per_class = []
        # for c in range(0, 10):
        #     if sample_per_class[c] == 0:
        #         acc_per_class.append(-1.0)
        #     else:
        #         acc_per_class.append(correct_per_class[c] / sample_per_class[c])

        # print_rank('acc_per_class: ' + str(acc_per_class))

        f1 = f1_score(labels.cpu(), pred.cpu(), average='micro')

        return {'output':output, 'acc': accuracy, 'batch_size': n_samples, \
                'f1_score': {'value':f1,'higher_is_better': True}} 



def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m


def seq_conv_bn(in_channels, out_channels, conv_kwargs, bn_kwargs):
    if "padding" not in conv_kwargs:
        conv_kwargs["padding"] = 1
    if "stride" not in conv_kwargs:
        conv_kwargs["stride"] = 1
    if "bias" not in conv_kwargs:
        conv_kwargs["bias"] = False
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, **conv_kwargs),
        # batch_norm(out_channels, **bn_kwargs),
        nn.ReLU(inplace=True)
    )


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), x.size(1))


class FastResnet(nn.Module):

    def __init__(self, num_classes, conv_kwargs=None, bn_kwargs=None,
                 conv_bn_fn=seq_conv_bn,
                 final_weight=0.125):
        super(FastResnet, self).__init__()

        conv_kwargs = {} if conv_kwargs is None else conv_kwargs
        bn_kwargs = {} if bn_kwargs is None else bn_kwargs

        self.prep = conv_bn_fn(3, 64, conv_kwargs, bn_kwargs)

        self.layer1 = nn.Sequential(
            conv_bn_fn(64, 128, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualBlock(128, 128, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn)
        )

        self.layer2 = nn.Sequential(
            conv_bn_fn(128, 256, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            conv_bn_fn(256, 512, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualBlock(512, 512, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn)
        )

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )

        self.final_weight = final_weight

        self.features = nn.Sequential(
            self.prep,
            self.layer1,
            self.layer2,
            self.layer3,
            self.head
        )

        self.classifier = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.

        f = self.features(x)

        y = self.classifier(f)
        y = y * self.final_weight
        return y


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kwargs, bn_kwargs,
                 conv_bn_fn=seq_conv_bn):
        super(IdentityResidualBlock, self).__init__()
        self.conv1 = conv_bn_fn(in_channels, out_channels, conv_kwargs, bn_kwargs)
        self.conv2 = conv_bn_fn(out_channels, out_channels, conv_kwargs, bn_kwargs)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual
