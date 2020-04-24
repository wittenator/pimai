from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, device, save_dir, warmup_method, warmup_period):
        super(Autoencoder, self).__init__()
        self.device = device
        self.save_dir = save_dir
        now = datetime.now()
        current_time = now.strftime("%Y%m%d-%H%M%S")
        self.warmup_method = warmup_method
        self.warmup_period = warmup_period
        self.writer = SummaryWriter(log_dir=self.save_dir + current_time)
        self.embeddings = []
        self.embedding_labels = []
        self.counter = 0

    def trains(self, device, train_loader, optimizer, epoch, epochs):
        self.train()
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = self.compute_loss_train(data, target, epoch, epochs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            self.writer.add_scalar('Loss/train', loss.item(), self.counter)

    def tests(self, device, test_loader, epoch, epochs):
        self.eval()
        test_loss = 0
        recon = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                loss, output = self.compute_loss_test(data, target, epoch, epochs)
                test_loss += loss
                recon += F.binary_cross_entropy_with_logits(output, data.view(-1, 784), reduction='none').sum(
                    axis=1).mean()

        test_loss /= len(test_loader.dataset)
        recon /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Reconstruction error: {}\n'.format(
            test_loss, recon))

    def kld_warmup(self, epoch, epochs):
        if self.warmup_method == 'none':
            return 1
        elif self.warmup_method == 'tanh':
            return np.tanh(epoch / (self.warmup_period*0.3))
        else:
            return np.tanh((epoch % self.warmup_period) / (self.warmup_period*0.3))

    def add_embedding(self, loader):
        with torch.no_grad():
            labels = []
            embs = []
            for data, label in loader:
                data, label = data.to(self.device), label.to(self.device)
                labels.append(label)
                _, a, b, *rest = self(data)
                emb = self.reparameterize(a,b)
                embs.append(emb)
            self.embeddings.append(torch.cat(tuple(embs), dim=0).cpu().numpy())
            self.embedding_labels = torch.cat(tuple(labels), dim=0).cpu().numpy()
            
    def visualize_embeddings(self, epoch, path):
        import hypertools as hyp
        hyp.plot(self.embeddings[epoch], '.', hue=self.embedding_labels, reduce='TSNE', ndims=2, save_path=f'{path}/{self.__class__.__name__}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.svg')

