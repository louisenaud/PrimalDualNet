"""
Project:    
File:       proximal_operators.py
Created by: louise
On:         10/13/17
At:         3:11 PM
"""

from torch.autograd import Variable
import torch
import torch.nn as nn


class ProximalLinfBall(nn.Module):
    def __init__(self):
        super(ProximalLinfBall, self).__init__()

    def forward(self, p, r):
        if p.is_cuda:
            m1 = torch.max(torch.add(p.data, - r), torch.zeros(p.size()).cuda())
            m2 = torch.max(torch.add(torch.neg(p.data), - r), torch.zeros(p.size()).cuda())
        else:
            m1 = torch.max(torch.add(p.data, - r), torch.zeros(p.size()))
            m2 = torch.max(torch.add(torch.neg(p.data), - r), torch.zeros(p.size()))
        return p - Variable(m1 - m2)


class ProximalL1(nn.Module):
    def __init__(self):
        super(ProximalL1, self).__init__()

    def forward(self, x, f, clambda):
        if x.is_cuda:
            res = x + torch.clamp(f - x, -clambda, clambda).cuda()
        else:
            res = x + torch.clamp(f - x, -clambda, clambda)
        return res


class ProximalL2(nn.Module):
    def __init__(self, x, f, clambda):
        super(ProximalL2, self).__init__()
        self.x = x
        self.f = f
        self.clambda = clambda

    def forward(self):
        return
