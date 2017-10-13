"""
Project:    
File:       main.py
Created by: louise
On:         10/13/17
At:         3:09 PM
"""
from __future__ import print_function
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms

from scipy.misc import face

from primal_dual_model import PrimalDualNetwork
from energies import PrimalEnergyROF, DualEnergyROF
from differential_operators import ForwardGradient, BackwardDivergence

if __name__ == '__main__':
    # cuda
    use_cuda = torch.cuda.is_available()
    print("Cuda = ", use_cuda)
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()
    t0 = time.time()
    # Create image to noise and denoise
    img_ = face(True)
    h, w = img_.shape
    img_.resize((h, w, 1))
    img_tensor = pil2tensor(img_.transpose(1, 0, 2)).cuda()
    img_ref = Variable(img_tensor)
    img_obs = img_ref + Variable(torch.randn(img_ref.size()).cuda() * 0.1)
    # Parameters
    norm_l = 7.0
    max_it = 200
    theta = 1.0
    tau = 0.01
    sigma = 1.0 / (norm_l * tau)
    #lambda_TVL1 = 1.0
    lambda_rof = 7.0

    x = Variable(img_obs.data.clone()).cuda()
    x_tilde = Variable(img_obs.data.clone()).cuda()
    img_size = img_ref.size()
    y = Variable(torch.zeros((img_size[0]+1, img_size[1], img_size[2]))).cuda()

    perof = PrimalEnergyROF()
    p_nrg = perof.forward(x, img_obs, lambda_rof)
    print("Primal Energy = ", p_nrg.data)
    derof = DualEnergyROF()
    d_nrg = derof.forward(y, img_obs)
    print("Dual Energy = ", d_nrg)

    # Solve ROF
    primal = np.zeros((max_it,))
    dual = np.zeros((max_it,))
    gap = np.zeros((max_it,))
    primal[0] = p_nrg.data[0]
    dual[0] = d_nrg.data[0]
    fwdgd = ForwardGradient()
    y = fwdgd.forward(x, dtype=torch.cuda.FloatTensor)

    # Plot reference, observed and denoised image
    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    # ax1.imshow(tensor2pil(img_ref.data.cpu()))
    # ax1.set_title("Reference image")
    # ax2.imshow(tensor2pil(img_obs.data.cpu()))
    # ax2.set_title("Observed image")
    # ax3.imshow(tensor2pil(x_tilde.data.cpu()))
    # ax3.set_title("Denoised image")

    # Net approach
    t0 = time.time()
    net = PrimalDualNetwork()
    dn_image = net.forward(x)
    print("Elapsed time = ", time.time() - t0, "s")
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    # Display results
    ax1.imshow(tensor2pil(img_ref.data.cpu()))
    ax1.set_title("Reference image")
    ax2.imshow(tensor2pil(img_obs.data.cpu()))
    ax2.set_title("Observed image")
    ax3.imshow(tensor2pil(dn_image.data.cpu()))
    ax3.set_title("Denoised image")
    plt.show()



