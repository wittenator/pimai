from src.model.sbvae import SBVAE
from src.model.dataset import build_dataset

import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import numpy as np

import plotly.graph_objects as go 

device, train_loader, test_loader, train_loader_occluded, test_loader_occluded = build_dataset("./assets/")


def test_vae_acc(weights_path, name, train_loader, test_loader):
    sbvae = SBVAE(device, './assets/data/runs/', "none", 20,k=50, dist=name.split('+')[0]).to(device)
    sbvae.load_state_dict(torch.load(weights_path, map_location=device))

    with torch.no_grad():
        labels = []
        embs = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            labels.append(label)
            _, a, b, *rest = sbvae(data)
            emb = sbvae.reparameterize(a, b)
            embs.append(emb)
        emb_train = (torch.cat(tuple(embs), dim=0).cpu().numpy())
        labels_train = torch.cat(tuple(labels), dim=0).cpu().numpy()

    with torch.no_grad():
        labels = []
        embs = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            labels.append(label)
            _, a, b, *rest = sbvae(data)
            emb = sbvae.reparameterize(a, b)
            embs.append(emb)
        emb_test = (torch.cat(tuple(embs), dim=0).cpu().numpy())
        labels_test = torch.cat(tuple(labels), dim=0).cpu().numpy()

    accs = []
    for n in [3, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(emb_train, labels_train)
        accs.append(accuracy_score(labels_test, knn.predict(emb_test)))

    return go.Bar(name=name, x=['n=3', 'n=5', 'n=10'], y=accs, text=np.around(accs, 3), textposition='auto')


traces = [test_vae_acc(f"./assets/data/sbvae-500-50--{warmup}--50--{method}/sbvae-500-50--{warmup}--50--{method}.pth", f"{method}+{warmup}", train_loader, test_loader) for method, warmup in itertools.product(['km', 'gamma', 'gl'], ['none', 'tanh', 'cycle'])]
traces.append(go.Bar(name='km + none (paper)', x=['n=3', 'n=5', 'n=10'], y=[1-0.0934, 1-0.0865, 1-0.0890], text=np.around(np.array([1-0.0934, 1-0.0865, 1-0.0890]), 3), textposition='auto'))
traces.append(go.Bar(name='Gauss VAE (paper)', x=['n=3', 'n=5', 'n=10'], y=[1-0.284, 1-0.2096, 1-0.1533], text=np.around(np.array([1-0.284, 1-0.2096, 1-0.1533]), 3), textposition='auto'))

fig = go.Figure(data=traces)
# Change the bar mode
fig.update_layout(
    barmode='group',
    title="MNIST:  Accuracy  for  kNN  on  latentspace",
    xaxis_title="Number of neighbours for KNN",
    yaxis_title="Accuracy"
)
fig.show()
fig.write_image("./knn.pdf")
