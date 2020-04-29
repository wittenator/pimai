from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.model.sssbvae import SSSBVAE
from src.model.dataset import build_dataset

import torch

import plotly.graph_objects as go 

device, train_loader, test_loader, train_loader_occluded, test_loader_occluded = build_dataset("./assets/")

def test_ssvae_acc(weights_path, name, train_loader, test_loader):

    sssbvae = SSSBVAE(device, './assets/data/runs/', "none", 20, k=50).to(device)
    sssbvae.load_state_dict(torch.load(weights_path, map_location=device))

    with torch.no_grad():
        labels = []
        embs = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            labels.append(label)
            data = torch.flatten(data,1)
            embs.append(data)
        emb_train = (torch.cat(tuple(embs), dim=0).cpu().numpy())
        labels_train = torch.cat(tuple(labels), dim=0).cpu().numpy()

    with torch.no_grad():
        labels = []
        embs = []
        embs2 = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            labels.append(label)
            _, a, b, y, *rest = sssbvae(data)
            emb = y.argmax(axis=1)
            embs.append(emb)
            data = torch.flatten(data,1)
            embs2.append(data)
        emb2_test = (torch.cat(tuple(embs2), dim=0).cpu().numpy())
        emb_test = (torch.cat(tuple(embs), dim=0).cpu().numpy())
        labels_test = torch.cat(tuple(labels), dim=0).cpu().numpy()





    for n in [5]:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(emb_train, labels_train)
        accs = accuracy_score(knn.predict(emb2_test), labels_test)

    accs_sssbvae = accuracy_score(emb_test, labels_test)

    re = [accs, accs_sssbvae]

    return re

weights = [
("./assets/data/sssbvae-50-50--cycle--50--km/sssbvae-50-50--cycle--50--km.pth", "km + cycle")
]
re = [test_ssvae_acc(*weight, train_loader, test_loader) for weight in weights]
traces = []

traces.append(go.Bar(name='Gauss-DGM (paper)', x=['0.01'], y=[1-0.0474]))
traces.append(go.Bar(name='Semi Supervised SB-VAE', x=['0.01'], y=[re[0][1]]))
traces.append(go.Bar(name='KNN = 5', x=['0.01'], y=[re[0][0]]))

fig = go.Figure(data=traces)
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(
    title="MNIST: Accuracy Semi-Supervised SB-VAE",
    xaxis_title="Percentage of non-occluded labels",
    yaxis_title="Accuracy"
)

fig.show()
fig.write_image("./sssbvae.pdf")
