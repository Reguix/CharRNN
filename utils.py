#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)