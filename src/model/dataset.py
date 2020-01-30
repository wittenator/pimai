import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

from multiprocessing import cpu_count
import numpy as np

def build_dataset():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if use_cuda else {}

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_test = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor()]))

    prob = 0.05
    train_occluded = np.array([1] * int(len(mnist_train) * prob) + [0] * int((1 - prob) * len(mnist_train)))
    test_occluded = np.array([1] * int(len(mnist_test) * prob) + [0] * int((1 - prob) * len(mnist_test)))
    np.random.shuffle(train_occluded)
    np.random.shuffle(test_occluded)

    mnist_train_occluded = datasets.MNIST('./data', train=True, download=True,
                                          transform=transforms.Compose([transforms.ToTensor()]))
    mnist_train_occluded.targets[train_occluded] = -1
    mnist_test_occluded = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_test_occluded.targets[test_occluded] = -1

    train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True, **kwargs)
    test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=True, **kwargs)

    train_loader_occluded = DataLoader(mnist_train_occluded, batch_size=128, shuffle=True, **kwargs)
    test_loader_occluded = DataLoader(mnist_test_occluded, batch_size=1000, shuffle=True, **kwargs)

    return device, train_loader, test_loader, train_loader_occluded, test_loader_occluded