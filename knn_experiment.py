from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.model.sbvae import SBVAE
from src.model.dataset import build_dataset

import torch

device, train_loader, test_loader, train_loader_occluded, test_loader_occluded = build_dataset("./assets/")


def test_vae_acc(weights_path, train_loader, test_loader):
    sbvae = SBVAE(device, './assets/data/runs/', "none", 20,k=50).to(device)
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
        for data, label in train_loader:
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
        accs.append(1 - accuracy_score(labels_test, knn.predict(emb_test)))

    return accs

weights = [
"./assets/data/sbvae-1000-50--cycle--50--gamma-.pth",
"./assets/data/sbvae-1000-50--cycle--50--gl-.pth",
"./assets/data/sbvae-1000-50--cycle--50--km-.pth",
"./assets/data/sbvae-1000-50--none--50--gamma-.pth",
"./assets/data/sbvae-1000-50--none--50--gl-.pth",
"./assets/data/sbvae-1000-50--none--50--km-.pth",
"./assets/data/sbvae-1000-50--tanh--50--gamma-.pth",
"./assets/data/sbvae-1000-50--tanh--50--gl-.pth",
"./assets/data/sbvae-1000-50--tanh--50--km-.pth"
]
[print(test_vae_acc(weight, train_loader, test_loader)) for weight in weights]