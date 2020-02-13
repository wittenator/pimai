import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

import numpy as np

from .base import Autoencoder



class SBVAE(Autoencoder):
    def __init__(self, device, save_dir, k=50, dist="km"):
        super(SBVAE, self).__init__(device, save_dir)
        self.k = k
        self.device = device
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.k)
        self.fc22 = nn.Linear(400, self.k)

        self.fc3 = nn.Linear(self.k, 400)
        self.fc4 = nn.Linear(400, 784)

        self.prior_alpha = torch.Tensor([1]).to(device)
        self.prior_beta = torch.Tensor([5]).to(device)

        self.dist = dist
    def encode(self, x):
        x = torch.flatten(x, 1)
        h1 = F.leaky_relu(self.fc1(x))
        return F.softplus(self.fc21(h1)), F.softplus(self.fc22(h1))

    def reparameterize(self, a, b):
        eps = 10 * torch.finfo(torch.float).eps  # smalles representable number such that 1.0 + eps != 1.0
        batch_size = a.size()[0]

        uniform_samples = Uniform(torch.tensor([eps]), torch.tensor([1.0 - eps])).rsample(a.size()).squeeze().to(
            self.device) if self.device.type == 'cpu' else torch.cuda.FloatTensor(a.size(0), a.size(1)).uniform_().clamp(eps, 1.0 - eps)
        exp_a = torch.reciprocal(
            a)  # returns a new tensor with the reciprocal of the elements of input: out_i = 1/input_i
        exp_b = torch.reciprocal(b)
        # value for Kumaraswamy distribution
        if (dist == "km"):
            km = (1 - uniform_samples.pow(exp_b) + eps).pow(exp_a)
        elif (dist == "gamma"):
            gamma_func_a = lgamma(a) #lgamma(a) == (a-1)! https://discuss.pytorch.org/t/is-there-a-gamma-function-in-pytorch/17122/2
            km = ((uniform_samples*a*gamma_func).pow(exp_a))/b
        else:
            km = (1 - uniform_samples.pow(exp_b) + eps).pow(exp_a)

        # no Nans are allowed in the matrix
        # assert not torch.isnan(km).any().item()

        # concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        cumprods = torch.cat((torch.ones([batch_size, 1], device=self.device), torch.cumprod(1 - km, axis=1)), dim=1)
        sticks = cumprods[:, :-1] * km  # cumulative product of elements along a given axis
        sticks[:, -1] = 1 - sticks[:, :-1].sum(axis=1)
        return sticks

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        a, b = self.encode(x)
        z = self.reparameterize(a, b)
        return self.decode(z), a, b

    def Beta(self, a, b):
        return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))

    def KLD(self, a, b, prior_alpha, prior_beta):
        ab = (a * b)
        kl = 1 / (1 + ab) * self.Beta(1 / a, b)
        kl += 1 / (2 + ab) * self.Beta(2 / a, b)
        kl += 1 / (3 + ab) * self.Beta(3 / a, b)
        kl += 1 / (4 + ab) * self.Beta(4 / a, b)
        kl += 1 / (5 + ab) * self.Beta(5 / a, b)
        kl += 1 / (6 + ab) * self.Beta(6 / a, b)
        kl += 1 / (7 + ab) * self.Beta(7 / a, b)
        kl += 1 / (8 + ab) * self.Beta(8 / a, b)
        kl += 1 / (9 + ab) * self.Beta(9 / a, b)
        kl += 1 / (10 + ab) * self.Beta(10 / a, b)
        kl *= (prior_beta - 1) * b

        kl += (a - prior_alpha) / a * (-np.euler_gamma - torch.digamma(
            b) - 1 / b)  # T.psi(self.posterior_b)

        # add normalization constants
        kl += torch.log(ab) + torch.log(self.Beta(prior_alpha, prior_beta))

        # final term
        kl += -(b - 1) / b

        return kl

    def loss_function(self, recon_x, x, a, b, prior_alpha, prior_beta, epoch, epochs):
        self.counter += 1
        period = 20
        BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none')
        KLD = self.KLD(a, b, prior_alpha, prior_beta)
        self.writer.add_scalar('KLD/train', KLD.sum(), self.counter)
        self.writer.add_scalar('BCE/train', BCE.sum(), self.counter)

        return 60000 / a.size(0) * torch.mean(
            1 / period * (epoch % period) * KLD.sum(axis=1) + BCE.sum(axis=1))

    def compute_loss_train(self, data, target, epoch, epochs):
        recon_batch, a, b = self(data)
        return self.loss_function(recon_batch, data, a, b, self.prior_alpha, self.prior_beta, epoch, epochs)

    def compute_loss_test(self, data, target, epoch, epochs):
        recon_batch, a, b = self(data)
        return self.loss_function(recon_batch, data, a, b, self.prior_alpha, self.prior_beta, epoch, epochs).item(), recon_batch