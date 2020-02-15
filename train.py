import argparse

from src.model.dataset import build_dataset

from src.model.vae import VAE
from src.model.sbvae import SBVAE
from src.model.sssbvae import SSSBVAE

import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim

DESCRIPTION = ("Train a VAE, SBVAE or a SSSBVAE using Pytorch.")
p = argparse.ArgumentParser(description=DESCRIPTION)

dataset = p.add_argument_group("Experiment options")
dataset.add_argument('--label-drop-percentage', type=float, default=.99,
                     help='percentage of labels to drop from training data. Default:%(default)s')

model = p.add_argument_group("Model options")
model.add_argument('--model', type=str, default='sbvae',
                   help='Model type to train. Default:%(default)s')
model.add_argument("--reparametrization", type=str, default="km",
                   help="Desired Modelreparametrization (km, gamma or gl")
model.add_argument('--latent-size', type=int, default=50,
                   help='dimensionality of latent variable. Default:%(default)s')
model.add_argument('--prior-concentration-param', type=float, default=1.,
                   help="the Beta prior's concentration parameter: v ~ Beta(1, alpha0). The larger the alpha0, the deeper the net. Default:%(default)s")
model.add_argument('--warmup-method', type=str, default='cycle',
                   help='dimensionality of latent variable. Default:%(default)s')
model.add_argument('--warmup-period', type=int, default=20,
                   help='dimensionality of latent variable. Default:%(default)s')

training = p.add_argument_group("Training options")
training.add_argument('--batch-size', type=int, default=100,
                      help='size of the batch to use when training the model. Default: %(default)s.')
training.add_argument('--max-epoch', type=int, metavar='N', default=2000,
                      help='train for a maximum of N epochs. Default: %(default)s')

optimizer = p.add_argument_group("AdaM Options")
optimizer.add_argument('--learning-rate', type=float, default=.003,
                       help="the AdaM learning rate (alpha) parameter. Default:%(default)s")

general = p.add_argument_group("General arguments")
general.add_argument('--save-dir', default="./assets/",
                     help='name of the folder where to save the experiment. Default: %(default)s.')

args = p.parse_args()
torch.autograd.set_detect_anomaly(True)

device, train_loader, test_loader, train_loader_occluded, test_loader_occluded = build_dataset(args.save_dir)

if args.model == 'vae':
    model = VAE(device, args.save_dir, args.warmup_method, args.warmup_period, k=args.latent_size).to(device)
elif args.model == 'sbvae':
    model = SBVAE(device, args.save_dir, args.warmup_method, args.warmup_period, k=args.latent_size).to(device)
elif args.model == 'sbvae':
    model = SSSBVAE(device, args.save_dir, args.warmup_method, args.warmup_period, k=args.latent_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate*10)
model.writer.add_graph(model, next(iter(train_loader))[0].to(device))



scheduler = MultiStepLR(optimizer, milestones=[1,2], gamma=0.1)
epochs = args.max_epoch
for epoch in range(1, epochs + 1):
    model.trains(device, train_loader if args.model != 'sssbvae' else train_loader_occluded, optimizer, epoch, epochs)
    model.tests(device, test_loader if args.model != 'sssbvae' else test_loader_occluded, epoch, epochs)
    scheduler.step()
    model.add_embedding(test_loader)
torch.save(model.state_dict(), f'{args.save_dir}data/{args.model}-{args.max_epoch}-{args.latent_size}--{args.warmup_method}--{args.warmup_period}.pth')


