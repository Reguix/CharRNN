# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:46:27 2018

@author: ZhangXin
"""

import torch
import os
import re
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SequenceDataset(Dataset):
    def __init__(self, is_train_set=True, root='', data_pickle=''):
        self.char2idx = {}
        self.idx2char = {}
        self.sequences = []
        self.seq_lengths = []
        self.seq_max_length = 0
        
        if os.path.isfile(data_pickle):
            print('Note : data load from {}!'.format(data_pickle))
            with open(data_pickle, 'rb') as f:
                data = torch.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = data['idx2char']
            self.sequences = data['sequences']
            self.seq_lengths = data['seq_lengths']
            self.seq_max_length = data['seq_max_length']
        else:
            self._parse_raw_data(root)
            pickle = {'char2idx': self.char2idx,
                      'idx2char': self.idx2char,
                      'sequences': self.sequences,
                      'seq_lengths': self.seq_lengths,
                      'seq_max_length': self.seq_max_length
                      }
            pickle_filename = re.sub('[\\/]', ' ', root)
            pickle_filename = re.split('\s+', pickle_filename)[-1]
            pickle_filename += '_' + time.strftime("%Y%m%d%H%M", time.localtime()) + '.pickle'
            with open(pickle_filename, 'wb') as f:
                torch.save(pickle, f)
            
    def _parse_raw_data(self, root):
        seq_max_length = 0
        sequences = []
        seq_lengths = []
        if not os.path.isdir(root):
            raise OSError('Directory {} does not exist.'.format(root))
        file_list = os.listdir(root)
        if not len(file_list):
            raise FileNotFoundError('Directory {} is empty.'.format(root))
        
        # set_lines = set()
        for filename in file_list:
            filename = os.path.join(root, filename)
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = re.sub('[\[\]]', ' ', line.strip())
                    sequence = re.split('\s+', line.strip())
                    # add <START> <EOP> tag
                    sequence = ['<START>'] + sequence + ['<EOP>']
                    
                    sequences.append(sequence)
                    seq_lengths.append(len(sequence))
                    seq_max_length = max(len(sequence), seq_max_length)
            self.seq_max_length = seq_max_length
                    
        self.sequences = self._transform_sequences(sequences, '</s>', seq_max_length)
        self.seq_lengths = torch.LongTensor(seq_lengths)
        
    
    def _transform_sequences(self, sequences, padding_char, seq_max_length):
        characters = {char for sequence in sequences for char in sequence}
        self.char2idx = {char: index for index, char in enumerate(characters)}
        self.char2idx[padding_char] = len(self.char2idx)
        self.idx2char = {index: char for char, index in list(self.char2idx.items())}
        
        index_sequences = [[self.char2idx[char] for char in sequence] for sequence in sequences]
        
        return self._pad_sequences(index_sequences, self.char2idx[padding_char], seq_max_length)
        
    def _pad_sequences(self, sequences, padding_value, seq_length):
        padding_sequences = torch.ones((len(sequences), seq_length)).long() * padding_value
        for i, sequence in enumerate(sequences):
            padding_sequences[i, :len(sequence)] = torch.LongTensor(sequence)
        
        return padding_sequences
                    
    def __getitem__(self, index):
        return self.sequences[index], self.seq_lengths[index]
    
    def __len__(self):
        return len(self.sequences)
    
    def get_num_characters(self):
        return len(self.char2idx)
    
    def get_sequence_max_length(self):
        return self.seq_max_length
        
        
if __name__== '__main__':
    train_dataset = SequenceDataset(data_pickle='data_pickle_test.pickle')
    # train_dataset = SequenceDataset(root='./data/raw_data')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=5, 
                              shuffle=True)
    print(train_dataset.char2idx)
    print(len(train_loader.dataset))
    print(len(train_loader))
    print(train_dataset.get_num_characters())
    print(train_dataset.get_sequence_max_length())
    for epoch in range(1):
        for i, (data, seq_lengths) in enumerate(train_loader):
            print('epoch {0} | batch {1} | sequence_lengths {2} \ndata {3}'.format(
                    epoch, i, seq_lengths, data))
#            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#            for i in range(len(seq_lengths)):
#                if seq_lengths[i] == train_dataset.get_sequence_max_length():
#                    seq_lengths[i] -= 1
#            data = data[perm_idx]
#            input = data[:, :-1]
#            target = data[:, 1:]
#            print('after pack_padded_sequence:')
#            print('sequence_lengths {0} \ndata {1} \ninput {2} \ntarget {3}'.format(
#                    seq_lengths, data, input, target))
#            input = input.transpose(0, 1)
#            print(pack_padded_sequence(input, seq_lengths.data.cpu().numpy()))
            
            
    
    