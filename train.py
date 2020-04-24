import argparse
import os

from src.model.dataset import build_dataset

from src.model.vae import VAE
from src.model.sbvae import SBVAE
from src.model.sssbvae import SSSBVAE

import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim

def train(save_dir='./assets/', model='sbvae', learning_rate=0.0003, max_epoch=1000, warmup_method='cycle', warmup_period=50, latent_size=50, reparametrization="km", **kwargs):
    learning_rate = learning_rate if reparametrization != 'gl' else learning_rate*50
    device, train_loader, test_loader, train_loader_occluded, test_loader_occluded = build_dataset(save_dir)
    modeltype = model
    name = f'{modeltype}-{max_epoch}-{latent_size}--{warmup_method}--{warmup_period}--{reparametrization}'
    os.makedirs(f'{save_dir}data/{name}/', exist_ok=True)
    if model == 'vae':
        model = VAE(device, save_dir + 'data/runs/' + name, warmup_method, warmup_period, k=latent_size).to(device)
    elif model == 'sbvae':
        model = SBVAE(device, save_dir + 'data/runs/' + name, warmup_method, warmup_period, k=latent_size, dist=reparametrization).to(device)
    elif model == 'sssbvae':
        model = SSSBVAE(device, save_dir + 'data/runs/' + name, warmup_method, warmup_period, k=latent_size, dist=reparametrization).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.writer.add_graph(model, next(iter(train_loader))[0].to(device))



    #scheduler = MultiStepLR(optimizer, milestones=[1], gamma=0.1)
    epochs = max_epoch
    for epoch in range(1, epochs + 1):
        name_epoch = f'{modeltype}-{epoch}-{latent_size}--{warmup_method}--{warmup_period}--{reparametrization}'
        model.trains(device, train_loader if model != 'sssbvae' else train_loader_occluded, optimizer, epoch, epochs)
        model.tests(device, test_loader if model != 'sssbvae' else test_loader_occluded, epoch, epochs)
        #scheduler.step()
        if epoch in [1, 10, 100, 250, 500, 1000, max_epoch]:
            torch.save(model.state_dict(), f'{save_dir}data/{name}/{name_epoch}.pth')
    model.add_embedding(test_loader)
    model.visualize_embeddings(-1, f'{save_dir}data/{name}')



if __name__ == "__main__":
    DESCRIPTION = ("Train a VAE, SBVAE or a SSSBVAE using Pytorch.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    dataset = p.add_argument_group("Experiment options")
    dataset.add_argument('--label-drop-percentage', type=float, default=.99,
                         help='percentage of labels to drop from training data. Default:%(default)s')

    model = p.add_argument_group("Model options")
    model.add_argument('--model', choices=['sbvae', 'vae', 'sssbvae'], default='sbvae',
                       help='Model type to train. Default:%(default)s')
    model.add_argument("--reparametrization", choices=['km', 'gl', 'gamma'], default='km',
                       help="Desired Modelreparametrization (km, gamma or gl")
    model.add_argument('--latent-size', type=int, default=50,
                       help='dimensionality of latent variable. Default:%(default)s')
    model.add_argument('--prior-concentration-param', type=float, default=1.,
                       help="the Beta prior's concentration parameter: v ~ Beta(1, alpha0). The larger the alpha0, the deeper the net. Default:%(default)s")
    model.add_argument('--warmup-method', choices=['cycle', 'tanh', 'none'], default='cycle',
                       help='dimensionality of latent variable. Default:%(default)s')
    model.add_argument('--warmup-period', type=int, default=50,
                       help='dimensionality of latent variable. Default:%(default)s')

    training = p.add_argument_group("Training options")
    training.add_argument('--batch-size', type=int, default=100,
                          help='size of the batch to use when training the model. Default: %(default)s.')
    training.add_argument('--max-epoch', type=int, metavar='N', default=2000,
                          help='train for a maximum of N epochs. Default: %(default)s')

    optimizer = p.add_argument_group("AdaM Options")
    optimizer.add_argument('--learning-rate', type=float, default=.0003,
                           help="the AdaM learning rate (alpha) parameter. Default:%(default)s")

    general = p.add_argument_group("General arguments")
    general.add_argument('--save-dir', default="./assets/",
                         help='name of the folder where to save the experiment. Default: %(default)s.')

    args= p.parse_args()
    torch.autograd.set_detect_anomaly(True)
    train(**vars(args))



