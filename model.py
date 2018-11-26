#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:07:17 2018

@author: ZhangXin
"""
import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 n_layers=2, drop_prob=0.1):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        # self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden):
        # input shape: B(batch_size) x S(sequence_length)
        batch_size, seq_len = input.size()
        # Embedding B x S -> B x S x E (embedding size)
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = self.linear(output.contiguous().view(-1, self.hidden_size))
        # output = self.dropout(output)
        return output, hidden
        
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden
        
        
def save_model(model, filename='char_rnn.ckpt'):
    checkpoint = {'input_size': model.input_size,
                  'hidden_size': model.hidden_size,
                  'output_size': model.output_size,
                  'n_layers': model.n_layers,
                  'drop_prob': model.drop_prob,
                  'state_dict': model.state_dict()}
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f)
    model = CharRNN(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['output_size'],
                    checkpoint['n_layers'], checkpoint['drop_prob'])
    model.load_state_dict(checkpoint['state_dict'])

    return model


if __name__ == '__main__':
    print('Model example :')
    model = CharRNN(1000, 256, 1000)
    print(model)
#    save_model(model, 'test_save_model.ckpt')
#    model = load_model('test_save_model.ckpt')
    