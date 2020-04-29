from src.model.sbvae import SBVAE
from src.model.dataset import build_dataset

import itertools
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.distributions import Beta, Gamma
import numpy as np
import hypertools as hyp


import plotly.graph_objects as go 
import matplotlib.pyplot as plt

device, train_loader, test_loader, train_loader_occluded, test_loader_occluded = build_dataset("./assets/")


paths = [x[0] for x in os.walk("./assets/data/") if x[0].startswith("./assets/data/sbvae")]

def draw_weights(path, name, model, test_loader, train_loader):
    norms = torch.norm(model.fc3.weight, p=2, dim=0).detach().numpy()
    norms = norms[::-1]/norms.sum()
    with torch.no_grad():
        embs = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            _, a, b, *rest = sbvae(data)
            emb = torch.cumsum(sbvae.reparameterize(a,b), dim=1)
            embs.append(emb)
        emb = (torch.cat(tuple(embs), dim=0).cpu().numpy())
        boundary = np.mean(np.argmax(emb > 0.99, axis=1))

    fig = go.Figure([go.Bar(x=list(range(0, norms.shape[0])), y=norms)])
    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=boundary,
            y0=0,
            x1=boundary,
            y1=np.max(norms),
            line=dict(
                color="Red",
                width=3
            )))
    fig.write_image(path + ".weights.pdf")

def draw_latent_space(path, name, model, test_loader, train_loader):
    with torch.no_grad():
        labels = []
        embs = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            labels.append(label)
            _, a, b, *rest = sbvae(data)
            emb = sbvae.reparameterize(a,b)
            embs.append(emb)
        emb = (torch.cat(tuple(embs), dim=0).cpu().numpy())
        labels = torch.cat(tuple(labels), dim=0).cpu().numpy()
        hyp.plot(emb, '.', hue=labels, reduce='TSNE', ndims=2, save_path=path + ".latent.pdf")

def draw_samples(path, name, model, test_loader, train_loader):
    sample = Gamma(torch.tensor([1.0]), torch.tensor([5.0])).rsample([64,50]).squeeze().to(device)
    cumprods = torch.cat((torch.ones([64, 1], device=device), torch.cumprod(1-sample, axis=1)), dim=1)
    sample = cumprods[:,:-1]*sample
    sample[:, -1] = 1- sample[:, :-1].sum(axis=1)
    sample = torch.sigmoid(model.decode(sample)).reshape(64, 28, 28).cpu().detach().numpy()
    f, axarr = plt.subplots(8,8)
    for i in range(64):
      axarr[i//8,i%8].imshow(sample[i])
    plt.savefig(path + ".samples.pdf")

files = [(f"./assets/data/sbvae-500-50--{warmup}--50--{method}/sbvae-500-50--{warmup}--50--{method}.pth", f"sbvae-500-50--{warmup}--50--{method}", method)  for method, warmup in itertools.product(['gamma'], ['none', 'tanh', 'cycle'])]

for path, name, method in files:
    sbvae = SBVAE(device, './assets/data/runs/', "none", 20,k=50, dist=method).to(device)
    sbvae.load_state_dict(torch.load(path, map_location=device))
    #draw_weights(path, name, sbvae, test_loader, train_loader)
    #draw_latent_space(path, name, sbvae, test_loader, train_loader)
    draw_samples(path, name, sbvae, test_loader, train_loader)




