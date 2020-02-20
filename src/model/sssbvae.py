from .sbvae import SBVAE

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from src.util.Distributions import Distributions

class SSSBVAE(SBVAE):
    def __init__(self, device, save_dir, warmup_method, warmup_period, k=50, dist=Distributions.KUMARASWAMY):
        super(SSSBVAE, self).__init__(device, save_dir, warmup_method, warmup_period, k=k)

        self.fc23 = nn.Linear(500, 10)
        self.fc3 = nn.Linear(self.k + 10, 500)

    def encode(self, x):
        x = torch.flatten(x, 1)
        h1 = F.relu(self.fc1(x))
        return F.softplus(self.fc21(h1)), F.softplus(self.fc22(h1)), F.softplus(self.fc23(h1))

    def forward(self, x):
        a, b, y = self.encode(x)
        z = self.reparameterize(a, b)
        z = torch.cat((z, y / torch.norm(y, p=1)), dim=1)
        return self.decode(z), a, b, y / torch.norm(y, p=1)

    def loss_function(self, recon_x, x, a, b, y, y_true, prior_alpha, prior_beta, epoch, epochs):
        period = 20
        BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='none')
        KLD = self.KLD(a, b, prior_alpha, prior_beta)

        log_y = F.binary_cross_entropy(y, torch.eye(10, device=self.device)[y_true], reduction='none').sum(axis=1)
        ent_y = Categorical(probs=y).entropy()

        y_recon = torch.where(y_true != -1, log_y, ent_y)

        eye_batch = torch.eye(10, device=self.device).unsqueeze(0).expand(a.size(0), -1, -1)
        y_batch = y.unsqueeze(2).expand(-1, -1, 10)

        factor = torch.where(y_true != -1, torch.ones(a.size(0), device=self.device),
                             F.binary_cross_entropy(y_batch, eye_batch, reduction='none').sum(axis=(1, 2)))

        term = kld_warmup(epoch, epochs)* KLD.sum(axis=1) + BCE.sum(axis=1) * factor + y_recon

        return 60000 / a.size(0) * (torch.mean(term))

    def compute_loss_train(self, data, target, epoch, epochs):
        recon_batch, a, b, y = self(data)
        return self.loss_function(recon_batch, data, a, b, y, target, self.prior_alpha, self.prior_beta, epoch, epochs)

    def compute_loss_test(self, data, target, epoch, epochs):
        recon_batch, a, b, y = self(data)
        return self.loss_function(recon_batch, data, a, b, y, target, self.prior_alpha, self.prior_beta, epoch,
                                  epochs).item(), recon_batch
