"""
Project:    
File:       energies.py
Created by: louise
On:         10/13/17
At:         3:13 PM
"""

from torch.autograd import Variable
import torch
import torch.nn as nn

from differential_operators import BackwardDivergence, ForwardGradient

class PrimalEnergyROF(nn.Module):
    def __init__(self):
        super(PrimalEnergyROF, self).__init__()

    def forward(self, x, img_obs, clambda):
        fwdgr = ForwardGradient()
        energy_reg = torch.sum(torch.norm(fwdgr.forward(x, torch.cuda.FloatTensor), 1))
        energy_data_term = torch.sum(0.5 * clambda * torch.norm(x - img_obs, 2))
        return energy_reg + energy_data_term


class DualEnergyROF(nn.Module):
    def __init__(self):
        super(DualEnergyROF, self).__init__()

    def forward(self, y, img_obs):
        bwd = BackwardDivergence()
        nrg = -0.5 * (img_obs - bwd.forward(y, torch.cuda.FloatTensor)) ** 2
        nrg = torch.sum(nrg)
        return nrg


def dual_energy_tvl1(y, im_obs):
    """
    Compute the dual energy of TV-L1 problem.
    :param y: pytorch Variable, [MxNx2]
    :param im_obs: pytorch Variable, observed image
    :return: float, dual energy
    """
    bwd = BackwardDivergence()
    nrg = -0.5 * (im_obs - bwd.forward(y, torch.cuda.FloatTensor))**2
    nrg = torch.sum(nrg)
    return nrg


def dual_energy_rof(y, im_obs):
    """
    Compute the dual energy of ROF problem.
    :param y: pytorch Variable, [MxNx2]
    :param im_obs: pytorch Variables [MxN], observed image
    :return: float, dual energy
    """
    nrg = -0.5 * (im_obs - backward_divergence(y, torch.cuda.FloatTensor))**2
    nrg = torch.sum(nrg)
    return nrg


def primal_energy_rof(x, img_obs, clambda):
    """

    :param x: pytorch Variables, [MxN]
    :param img_obs: pytorch Variable [MxN], observed image
    :param clambda: float, lambda parameter
    :return: float, primal ROF energy
    """
    energy_reg = torch.sum(torch.norm(forward_gradient(x, torch.cuda.FloatTensor), 1))
    energy_data_term = torch.sum(0.5*clambda * torch.norm(x - img_obs, 2))
    return energy_reg + energy_data_term


def primal_energy_tvl1(x, img_obs, clambda):
    """

    :param x: pytorch Variables, [MxN]
    :param img_obs: pytorch Variables [MxN], observed image
    :param clambda: float, lambda parameter
    :return: float, primal ROF energy
    """
    energy_reg = torch.sum(torch.norm(forward_gradient(x,torch.cuda.FloatTensor), 1))
    energy_data_term = torch.sum(clambda * torch.abs(x - img_obs))
    return energy_reg + energy_data_term
