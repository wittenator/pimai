import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Autoencoder


class VAE(Autoencoder):
    def __init__(self, device, save_dir, warmup_method, warmup_period, k=20):
        super(VAE, self).__init__(device, save_dir, warmup_method, warmup_period)

        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, k)
        self.fc22 = nn.Linear(500, k)
        self.fc3 = nn.Linear(k, 500)
        self.fc4 = nn.Linear(500, 784)

    def encode(self, x):
        x = torch.flatten(x, 1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        self.name = 'Gauss VAE'
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, epoch, epochs):
        self.counter += 1
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = self.kld_warmup(epoch, epochs)*(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        self.writer.add_scalar('KLD/train', KLD.sum(), self.counter)
        self.writer.add_scalar('BCE/train', BCE.sum(), self.counter)

        return BCE + KLD

    def compute_loss_train(self, data, target, epoch, epochs):
        recon_batch, mu, logvar = self(data)
        return self.loss_function(recon_batch, data, mu, logvar, epoch, epochs)

    def compute_loss_test(self, data, target, epoch, epochs):
        recon_batch, mu, logvar = self(data)
        return self.loss_function(recon_batch, data, mu, logvar, epoch, epochs).item(), recon_batch  # sum up batch loss
