# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:13:17 2018

@author: ZhangXin
"""
import os
import re
import time
import torch
import argparse
from model import CharRNN, load_model
from torch.autograd import Variable
from sequence_dataset import SequenceDataset


def generate(root, model_file):
    pass
    
   
def predicate(sequence, model_file, data_pickle):
    train_dataset = SequenceDataset(data_pickle=data_pickle)
    idx2char , char2idx = train_dataset.idx2char, train_dataset.char2idx
    
    char_rnn = load_model(model_file)
    input = torch.Tensor([char2idx['<START>']]).view(1, 1).long()
    hidden = char_rnn.init_hidden(input.size()[0])
    if torch.cuda.is_available():
        input, hidden = input.cuda(), hidden.cuda()
    input, hidden = Variable(input), Variable(hidden)
        
    output, hidden = char_rnn(input, hidden)
    for i, char in enumerate(sequence[:]):
        print(char, end=' ')
        if char not in char2idx.keys():
            return None
        input = input.data.new([char2idx[char]]).view(1, 1)
        output, hidden = char_rnn(input, hidden)
    # row_max row_max_index = torch.max(tensor, 1)
    pred_char = idx2char[int(torch.max(output, 1)[1].data.numpy().squeeze())] 
    return pred_char
        

# Run as standalone script    
if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_pickle', type=str, default='./data_pickle_test.pickle')
    argparser.add_argument('--model_file', type=str, default='./model_file_test.ckpt')
    args = argparser.parse_args()
    # print(vars(args))
    test = [['好', '为', '庐', '山'],
            ['寒', '雨', '连', '江', '夜', '入', '吴', '平', '明', '送', '客', '楚', '山'],
            ['五', '岳', '寻', '仙', '不', '辞', '远', '一', '生', '好', '入', '名', '山'],
            ['庐', '山'],
            ['黄', '云', '万', '里', '动', '风', '色', '白', '波', '九', '道', '流', '雪', '山'],
            ['好', '为', '庐', '山', '谣', '兴', '因', '庐', '山']]
    for t in test:
        print(predicate(t, args.model_file, args.data_pickle))
    