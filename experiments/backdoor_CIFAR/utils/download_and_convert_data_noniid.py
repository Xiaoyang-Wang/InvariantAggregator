import h5py
import json
import time
import os
from random import Random
import random


import torch
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn

import numpy as np
from numpy.random import RandomState

import tqdm

# Getting this function from FedML -- 02-17-22
def __getDirichletData__(dataset, n_nets, alpha, K, seed):
    labelList = np.array(dataset.targets)
    min_size = 0
    N = len(labelList)
    np.random.seed(seed)

    net_dataidx_map = {}
    while min_size < K:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(labelList == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    local_sizes = []
    for i in range(n_nets):
        local_sizes.append(len(net_dataidx_map[i]))
    local_sizes = np.array(local_sizes)
    weights = local_sizes / np.sum(local_sizes)

    print('Data statistics: %s' % str(net_cls_counts))
    print('Data ratio: %s' % str(weights))

    return net_dataidx_map


def _dump_dict_to_hdf5(data_dict: dict, hdf5_file: h5py.File):
    '''Dump dict with expected structure to HDF5 file'''

    hdf5_file.create_dataset('users', data=data_dict['users'])
    hdf5_file.create_dataset('num_samples', data=data_dict['num_samples'])

    # Store actual data in groups
    user_data_group = hdf5_file.create_group('user_data')
    for user, user_data in tqdm.tqdm(data_dict['user_data'].items()):
        user_subgroup = user_data_group.create_group(user)
        user_subgroup.create_dataset('x', data=user_data) 

    user_data_label_group = hdf5_file.create_group('user_data_label')
    for user, user_data_label in tqdm.tqdm(data_dict['user_data_label'].items()):
        user_data_label_group.create_dataset(user, data=user_data_label) 

def _process_and_save_to_disk(dataset, n_users, file_format, output):
    '''Process a Torchvision dataset to expected format and save to disk'''

    # Split training data equally among all users
    total_samples = len(dataset)
    samples_per_user = total_samples // n_users
    assert total_samples % n_users == 0

    # Function for getting a given user's data indices
    user_idxs = lambda user_id: slice(user_id * samples_per_user, (user_id + 1) * samples_per_user)

    # Convert training data to expected format
    print('Converting data to expected format...')
    start_time = time.time()

    data_dict = {  # the data is expected to have this format
        'users' : [f'{user_id:04d}' for user_id in range(n_users)],
        'num_samples' : 10000 * [samples_per_user],
        'user_data' : {f'{user_id:04d}': dataset.data[user_idxs(user_id)].tolist() for user_id in range(n_users)},
        'user_data_label': {f'{user_id:04d}': dataset.targets[user_idxs(user_id)] for user_id in range(n_users)},
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

def _noniid_process_and_save_to_disk(dataset, index_map, n_users, file_format, output):
    '''Process a Torchvision dataset to expected format and save to disk'''

    # Split training data equally among all users
    total_samples = len(dataset)
    samples_per_user = total_samples // n_users
    assert total_samples % n_users == 0

    # Function for getting a given user's data indices
    # user_idxs = lambda user_id: slice(user_id * samples_per_user, (user_id + 1) * samples_per_user)

    # Convert training data to expected format
    print('Converting data to expected format...')
    start_time = time.time()
    
    data_dict = {  # the data is expected to have this format
        'users' : [f'{user_id:04d}' for user_id in range(n_users)],
        'num_samples' : [len(index_map[user_id]) for user_id in range(n_users)],
        'user_data' : {f'{user_id:04d}': dataset.data[index_map[user_id]].tolist() for user_id in range(n_users)},
        'user_data_label': {f'{user_id:04d}': [dataset.targets[idx] for idx in index_map[user_id]] for user_id in range(n_users)},
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


# Get training and testing data from torchvision
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)

index_map = __getDirichletData__(trainset, 100, alpha=0.5, K=10, seed=0)

print('Processing training set...')
_noniid_process_and_save_to_disk(trainset, index_map, n_users=100, file_format='hdf5', output='./noniid_data/train_data')

print('Processing test set...')
_process_and_save_to_disk(testset, n_users=1, file_format='hdf5', output='./noniid_data/test_data')