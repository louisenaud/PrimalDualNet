"""
Project:    
File:       differential_operators.py
Created by: louise
On:         10/13/17
At:         3:10 PM
"""

from torch.autograd import Variable
import torch
import torch.nn as nn


class ForwardGradient(nn.Module):
    def __init__(self):
        super(ForwardGradient, self).__init__()

    def forward(self, x, dtype=torch.FloatTensor):
        im_size = x.size()
        gradient = Variable(torch.zeros((2, im_size[1], im_size[2])).type(dtype))  # Allocate gradient array
        # Horizontal direction
        gradient[0, :, :-1] = x[0, :, 1:] - x[0, :, :-1]
        # Vertical direction
        gradient[1, :-1, :] = x[0, 1:, :] - x[0, :-1, :]
        return gradient


class BackwardDivergence(nn.Module):
    def __init__(self):
        super(BackwardDivergence, self).__init__()

    def forward(self, y, dtype=torch.FloatTensor):
        im_size = y.size()
        # Horizontal direction
        d_h = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_h[0, :, 0] = y[0, :, 0]
        d_h[0, :, 1:-1] = y[0, :, 1:-1] - y[0, :, :-2]
        d_h[0, :, -1] = -y[0, :, -2:-1]

        # Vertical direction
        d_v = Variable(torch.zeros((1, im_size[1], im_size[2])).type(dtype))
        d_v[0, 0, :] = y[1, 0, :]
        d_v[0, 1:-1, :] = y[1, 1:-1, :] - y[1, :-2, :]
        d_v[0, -1, :] = -y[1, -2:-1, :]

        # Divergence
        div = d_h + d_v
        return div