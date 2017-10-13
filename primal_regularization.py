"""
Project:    
File:       primal_regularization.py
Created by: louise
On:         10/13/17
At:         3:45 PM
"""

class PrimalRegularization(nn.Module):
    def __init__(self, theta):
        super(PrimalRegularization, self).__init__()
        self.theta = theta

    def forward(self, x, x_old):
        x_tilde = x + self.theta * (x - x_old)
        return x_tilde