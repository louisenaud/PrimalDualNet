"""
Project:    
File:       primal_dual_model.py
Created by: louise
On:         10/13/17
At:         3:45 PM
"""
from torch.autograd import Variable
import torch
import torch.nn as nn

from proximal_operators import ProximalLinfBall
from differential_operators import ForwardGradient, BackwardDivergence
from energies import PrimalEnergyROF, DualEnergyROF

class PrimalUpdate(nn.Module):
    """
    Class to compute the primal update for the ROF model in the Primal Dual setup.
    """
    def __init__(self, lambda_rof, tau):
        super(PrimalUpdate, self).__init__()
        self.backward_div = BackwardDivergence()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, img_obs):
        x = (x + self.tau * self.backward_div.forward(y, dtype=torch.cuda.FloatTensor) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class PrimalWeightedUpdate(nn.Module):
    """
    Class to compute the Primal update for the weighted version of the ROF model i.e. :
    ||x-f||^2 + sum(wi |xi - xj|)
    """
    def __init__(self, lambda_rof, tau):
        super(PrimalWeightedUpdate, self).__init__()
        self.backward_div = BackwardDivergence()
        self.tau = tau
        self.lambda_rof = lambda_rof

    def forward(self, x, y, img_obs):
        x = (x + self.tau * self.backward_div.forward(y, dtype=torch.cuda.FloatTensor) +
             self.lambda_rof * self.tau * img_obs) / (1.0 + self.lambda_rof * self.tau)
        return x


class PrimalRegularization(nn.Module):
    """
    Class to compute the regularization in the Primal. 
    """
    def __init__(self, theta):
        super(PrimalRegularization, self).__init__()
        self.theta = theta

    def forward(self, x, x_tilde, x_old):
        x_tilde = x + self.theta * (x - x_old)
        return x_tilde


class DualUpdate(nn.Module):
    """
    Class to compute the dual update of the ROF model. 
    """
    def __init__(self, sigma):
        super(DualUpdate, self).__init__()
        self.forward_grad = ForwardGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y):
        if y.is_cuda:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, dtype=torch.cuda.FloatTensor)
        else:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, dtype=torch.FloatTensor)
        return y


class DualWeightedUpdate(nn.Module):
    """
    Class to compute the dual update of the ROF model. 
    """
    def __init__(self, sigma):
        super(DualWeightedUpdate, self).__init__()
        self.forward_grad = ForwardGradient()
        self.sigma = sigma

    def forward(self, x_tilde, y):
        if y.is_cuda:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, dtype=torch.cuda.FloatTensor)
        else:
            y = y + self.sigma * self.forward_grad.forward(x_tilde, dtype=torch.FloatTensor)
        return y


class PrimalDualNetwork(nn.Module):
    """
    Primal Dual algorithm implemented in the PyTorch framework.
    """
    def __init__(self, max_it=20, lambda_rof=7.0, sigma=1. / (7.0 * 0.01), tau=0.01, theta=0.5):
        super(PrimalDualNetwork, self).__init__()
        self.max_it = max_it
        self.dual_update = DualUpdate(sigma)
        self.prox_l_inf = ProximalLinfBall()
        self.primal_update = PrimalUpdate(lambda_rof, tau)
        self.primal_reg = PrimalRegularization(theta)

        self.energy_primal = PrimalEnergyROF()
        self.energy_dual = DualEnergyROF()
        self.gap = 0.0

    def forward(self, img_obs):
        x = img_obs.clone().cuda()
        x_tilde = img_obs.clone().cuda()
        img_size = img_obs.size()
        x_old = x.clone().cuda()
        y = Variable(torch.zeros((img_size[0] + 1, img_size[1], img_size[2]))).cuda()
        for it in range(self.max_it):
            # Dual update
            y = self.dual_update.forward(x_tilde, y)
            y = self.prox_l_inf.forward(y, 1.0)
            # Primal update
            x_old = x
            x = self.primal_update.forward(x, y, img_obs)
            # Smoothing
            x_tilde = self.primal_reg.forward(x, x_tilde, x_old)

            # Compute energies
            pnrg = self.energy_primal.forward(x, img_obs, self.primal_update.lambda_rof)
            dnrg = self.energy_dual.forward(y, img_obs)
            self.gap = pnrg - dnrg
            print self.gap

        return x_tilde
