#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:07:17 2018

@author: ZhangXin
"""
import torch
import argparse
import time
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet import meter


from model import CharRNN, save_model, load_model
from sequence_dataset import SequenceDataset

def train(root, data_pickle, n_epochs, print_every, hidden_size, n_layers, lr, batch_size):
    train_dataset = SequenceDataset(root=root)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    
    n_characters = train_dataset.get_num_characters()
    
    char_rnn = CharRNN(n_characters, hidden_size, n_characters, n_layers)
    print(char_rnn)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        char_rnn = nn.DataParallel(char_rnn)
    if torch.cuda.is_available():
        char_rnn.cuda()
    
    optimizer = torch.optim.Adam(char_rnn.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    print(optimizer)
    print(criterion)

    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10
        
    try:
        print("Training for %d epochs..." % n_epochs)
        for epoch in range(n_epochs):
            loss_meter.reset()
            for i, (data, _) in enumerate(train_loader):
                data = data.contiguous()
                hidden = char_rnn.init_hidden(data.size()[0])
                if torch.cuda.is_available():
                    data, hidden = data.cuda(), hidden.cuda()
                data, hidden = Variable(data), Variable(hidden)
                input = data[:, :-1]
                target = data[:, 1:]
                
                optimizer.zero_grad()
                
                output, _ = char_rnn(input, hidden)

                loss = criterion(output, target.contiguous().view(-1))
                loss.backward()
                optimizer.step()
                loss_meter.add(loss.item())
                
                if (i + 1) % print_every == 0:
                    print('Epoch: %s | percentage : %.2f%% | train loss: %.4f'
                          % (epoch, (100. * i / len(train_loader)), loss_meter.value()[0]))
            
            save_model(char_rnn, '%sepoch_%s_loss_%.4f_time_%s.ckpt' % 
                       ('./checkpoints/', epoch, loss_meter.value()[0], 
                        time.strftime("%Y%m%d%H%M", time.localtime())))
            
            if loss_meter.value()[0] > previous_loss:
                lr = lr * 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            previous_loss = loss_meter.value()[0]
    except KeyboardInterrupt:
        print("Saving before quit...")
        save_model(char_rnn, '%sepoch_%s_loss_%.4f_time_%s.ckpt' %
                   ('./checkpoints/KeyboardInterrupt_', epoch, loss_meter.value()[0], 
                    time.strftime("%Y%m%d%H%M", time.localtime())))

# Run as standalone script    
if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root', type=str, default='./data/raw_data')
    argparser.add_argument('--data_pickle', type=str, default='')
    argparser.add_argument('--n_epochs', type=int, default=100)
    argparser.add_argument('--print_every', type=int, default=3)
    argparser.add_argument('--hidden_size', type=int, default=256)
    argparser.add_argument('--n_layers', type=int, default=2)
    argparser.add_argument('-lr', type=float, default=0.001)
    argparser.add_argument('--batch_size', type=int, default=4)
    args = argparser.parse_args()
    print(vars(args))
    
    train(**vars(args))
    
    