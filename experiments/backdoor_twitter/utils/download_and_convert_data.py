import h5py
import json
import time

import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe

import tqdm
import pickle

def _dump_dict_to_hdf5(data_dict: dict, hdf5_file: h5py.File):
    '''Dump dict with expected structure to HDF5 file'''

    hdf5_file.create_dataset('users', data=data_dict['users'])
    hdf5_file.create_dataset('num_samples', data=data_dict['num_samples'])

    # Store actual data in groups
    user_data_group = hdf5_file.create_group('user_data')
    for user, user_data in tqdm.tqdm(data_dict['user_data'].items()):
        user_subgroup = user_data_group.create_group(user)
        user_subgroup.create_dataset('x', data=user_data) 

    user_data_lens_group = hdf5_file.create_group('user_data_lens')
    for user, user_data_lens in tqdm.tqdm(data_dict['user_data_lens'].items()):
        user_data_lens_group.create_dataset(user, data=user_data_lens) 

    user_data_label_group = hdf5_file.create_group('user_data_label')
    for user, user_data_label in tqdm.tqdm(data_dict['user_data_label'].items()):
        user_data_label_group.create_dataset(user, data=user_data_label) 

def _process_and_save_to_disk(data, lens, targets, n_users, file_format, output):
    '''Process a Torchvision dataset to expected format and save to disk'''

    # Split training data equally among all users
    total_samples = data.shape[0]
    samples_per_user = total_samples // n_users
    assert total_samples % n_users == 0

    # Function for getting a given user's data indices
    user_idxs = lambda user_id: slice(user_id * samples_per_user, (user_id + 1) * samples_per_user)

    # Convert training data to expected format
    print('Converting data to expected format...')
    start_time = time.time()

    data_dict = {  # the data is expected to have this format
        'users' : [f'{user_id:04d}' for user_id in range(n_users)],
        'num_samples' : 5000 * [samples_per_user],
        'user_data' : {f'{user_id:04d}': data[user_idxs(user_id)].tolist() for user_id in range(n_users)},
        'user_data_lens' : {f'{user_id:04d}': lens[user_idxs(user_id)] for user_id in range(n_users)},
        'user_data_label': {f'{user_id:04d}': targets[user_idxs(user_id)] for user_id in range(n_users)},
    } 

    print(f'Finished converting data in {time.time() - start_time:.2f}s.')

    # Save training data to disk
    print('Saving data to disk...')
    start_time = time.time()

    if file_format == 'json':
        with open(output + '.json', 'w') as json_file:
            json.dump(data_dict, json_file)
    elif file_format == 'hdf5':
        with h5py.File(output + '.hdf5', 'w') as hdf5_file:
            _dump_dict_to_hdf5(data_dict=data_dict, hdf5_file=hdf5_file)
    else:
        raise ValueError('unknown format.')

    print(f'Finished saving data in {time.time() - start_time:.2f}s.')


# make sent140 dataset, tensordataset
with open('path_to_leaf/leaf/data/sent140/data/test/all_data_niid_0_keep_0_test_9.json') as f:
   data = json.load(f)

text = [data['user_data'][data['users'][i]]['x'][0][4] for i in range(0, 50000)]
label = [data['user_data'][data['users'][i]]['y'][0] for i in range(0, 50000)]

train_label = torch.Tensor(label).long()

tokenizer = get_tokenizer('basic_english')
text = [tokenizer(s) for s in text]

vec = GloVe(name='6B', dim=300)
ret = [vec.get_vecs_by_tokens(s) for s in text]

train_lens = [len(r) for r in ret]
train_data = nn.utils.rnn.pad_sequence(ret, batch_first=True)

text = [data['user_data'][data['users'][i]]['x'][0][4] for i in range(50000, 55000)]
label = [data['user_data'][data['users'][i]]['y'][0] for i in range(50000, 55000)]

test_label = torch.Tensor(label).long()

text = [tokenizer(s) for s in text]

ret = [vec.get_vecs_by_tokens(s) for s in text]

test_lens = [len(r) for r in ret]
test_data = nn.utils.rnn.pad_sequence(ret, batch_first=True)

# print('Processing training set...')
# _process_and_save_to_disk(train_data, train_lens, train_label, n_users=100, file_format='hdf5', output='./data/train_data')

# print('Processing test set...')
# _process_and_save_to_disk(test_data, test_lens, test_label, n_users=1, file_format='hdf5', output='./data/test_data')

file = open("data/backdoor_train.txt","r")

text = file.readlines()

text = [tokenizer(s) for s in text]
ret = [vec.get_vecs_by_tokens(s) for s in text]
train_lens = [len(r) for r in ret]
train_data = nn.utils.rnn.pad_sequence(ret, batch_first=True)

print('train_lens: ', train_lens)
print('train_data.shape: ', train_data.shape)

train_target = torch.zeros((train_data.shape[0],), dtype=int)

torch.save(train_data, 'data/backdoor_train_data.pt')
torch.save(train_target, 'data/backdoor_train_target.pt')
with open('data/backdoor_train_data_lens.txt', 'wb') as file:
    pickle.dump(train_lens, file)

file = open("data/backdoor_test.txt","r")

text = file.readlines()

text = [tokenizer(s) for s in text]
ret = [vec.get_vecs_by_tokens(s) for s in text]
test_lens = [len(r) for r in ret]
test_data = nn.utils.rnn.pad_sequence(ret, batch_first=True)

print('test_lens: ', test_lens)
print('test_data.shape: ', test_data.shape)

test_target = torch.zeros((test_data.shape[0],), dtype=int)

torch.save(test_data, 'data/backdoor_test_data.pt')
torch.save(test_target, 'data/backdoor_test_target.pt')
with open('data/backdoor_test_data_lens.txt', 'wb') as file:
    pickle.dump(test_lens, file)
