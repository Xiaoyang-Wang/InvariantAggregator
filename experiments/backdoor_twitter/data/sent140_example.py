import torch
from torch import nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe

import json

with open('path_to_leaf/leaf/data/sent140/data/test/all_data_niid_0_keep_0_test_9.json') as f:
   data = json.load(f)

print(data.keys())

# print(data['users'][0])
# print(data['num_samples'][0])

print(data['user_data'][data['users'][0]]['x'][0][4])
print(data['user_data'][data['users'][0]]['y'])
print(len(data['users']))

text = [data['user_data'][data['users'][i]]['x'][0][4] for i in range(0, 20000)]
label = [data['user_data'][data['users'][i]]['y'][0] for i in range(0, 20000)]
print('text: ', text[0:10])
print('label: ', label[0:10])
label = torch.Tensor(label).long()

tokenizer = get_tokenizer('basic_english')
text = [tokenizer(s) for s in text]

# print('text: ', text)

vec = GloVe(name='6B', dim=300)
ret = [vec.get_vecs_by_tokens(s) for s in text]

# print('ret: ', ret)
length = [len(r) for r in ret]
ret = nn.utils.rnn.pad_sequence(ret, batch_first=True)

ret = ret[0:64]
length = length[0:64]
ret = nn.utils.rnn.pack_padded_sequence(ret, length, batch_first=True, enforce_sorted=False)
# print('ret: ', ret.data.shape)

label = label[0:64]

class TextClassificationModel(nn.Module):

   def __init__(self, embed_dim, num_class):
      super(TextClassificationModel, self).__init__()

      self.lstm = nn.LSTM(input_size=300, hidden_size=300, batch_first=True)
      self.fc = nn.Linear(300, 2)
      self.init_weights()

   def init_weights(self):
      initrange = 0.5
      self.fc.weight.data.uniform_(-initrange, initrange)
      self.fc.bias.data.zero_()

   def forward(self, text):
      packed_output, (hn, cn) = self.lstm(text)
      output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
      print('output: ', output.shape)
      print('output_length: ', output_length.shape)
      print('hn: ', hn.shape)
      print('cn: ', cn.shape)

      hn = hn[-1]
      print('hn: ', hn.shape)
      output = self.fc(hn)

      return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LSTMModel = TextClassificationModel(300, 2)
criterion = nn.CrossEntropyLoss()

LSTMModel = LSTMModel.to(device)
criterion = criterion.to(device)

lr = 5e-4

optimizer = optim.Adam(LSTMModel.parameters(), lr=lr)

ret = ret.to(device)
label = label.to(device)

output = LSTMModel(ret)

loss = criterion(output, label)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print('output.shape: ', output.shape)
