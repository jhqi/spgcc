import argparse
from Preprocess import Preprocess
from torchvision import transforms
import numpy as np
from pretrain_models import Enc_VAE, Dec_VAE, CNN_Enc
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler
from utils import AEDataset, latent_loss
import os
import sys
run_dir=os.path.dirname(sys.argv[0])
if run_dir=='':
    run_dir='.'
os.chdir(run_dir)

device = torch.device('cuda:1')

parser = argparse.ArgumentParser(description="VAE-based 3D-CNN pretrain")
parser.add_argument('--dataset', type=str, default='IP')
parser.add_argument('--encoded_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--windowsize', type=int, default=27)
parser.add_argument('--Patch_channel', type=int, default=15)
args = parser.parse_args()

if args.dataset == 'IP':
    args.Patch_channel = 30
    pretrain_epochs=35
elif args.dataset == 'SA':
    pretrain_epochs=80
elif args.dataset == 'PU':
    pretrain_epochs=60

XPath = f'DataArray/{args.dataset}_X.npy'

if not os.path.exists(XPath):
    Preprocess(XPath, args.dataset, args.windowsize, Patch_channel=args.Patch_channel)

trans = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(np.zeros(args.Patch_channel), np.ones(args.Patch_channel))])

Patch_dataset = AEDataset(XPath, trans)

Patch_loader = DataLoader(dataset=Patch_dataset, batch_size=args.batch_size, shuffle=True)
Enc_patch = Enc_VAE(channel=args.Patch_channel, output_dim=args.encoded_dim, windowSize=args.windowsize).to(device)
Dec_patch = Dec_VAE(channel=args.Patch_channel, windowSize=args.windowsize, input_dim=args.encoded_dim).to(device)
optim_enc = Adam(Enc_patch.parameters(), lr=1e-3, weight_decay=5e-4)
optim_dec = Adam(Dec_patch.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.MSELoss()

for epoch in range(1, pretrain_epochs + 1):
    Enc_patch.train()
    Dec_patch.train()
    epoch_loss = 0
    with torch.set_grad_enabled(True):
        for data in Patch_loader:
            data = data.float().to(device)
            optim_dec.zero_grad()
            optim_enc.zero_grad()
            _, mu, sigma = Enc_patch(data)
            std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
            code = mu + sigma * torch.tensor(std_z, requires_grad=False).to(device)
            recon = Dec_patch(code)
            recon_loss = criterion(data, recon)
            distribution_loss = latent_loss(mu, sigma)
            loss = recon_loss + distribution_loss
            loss.backward()
            optim_dec.step()
            optim_enc.step()
            epoch_loss += loss.item() * data.shape[0]
    print(f'Epoch:{epoch}, Loss:{epoch_loss/len(Patch_dataset)}')
state_dict = Enc_patch.state_dict()
state_dict.pop('projector.0.weight')
state_dict.pop('projector.0.bias')
state_dict.pop('mu.weight')
state_dict.pop('mu.bias')
state_dict.pop('log_sigma.weight')
state_dict.pop('log_sigma.bias')
torch.save(state_dict, f'pretrain_checkpoints/{args.dataset}_pretrain_weight.pth')

eval_loader = DataLoader(dataset=Patch_dataset, batch_size=4096, shuffle=False)
Enc_patch = CNN_Enc(channel=args.Patch_channel, output_dim=args.encoded_dim, windowSize=args.windowsize)
Enc_patch.load_state_dict(torch.load(f'pretrain_checkpoints/{args.dataset}_pretrain_weight.pth', map_location=torch.device('cpu')))
Enc_patch.to(device)
Enc_patch.eval()
CNN_encoding = torch.FloatTensor([])
with torch.set_grad_enabled(False):
    for data in eval_loader:
        data = data.float().to(device)
        encoding = Enc_patch(data)
        CNN_encoding = torch.concat([CNN_encoding, encoding.detach().cpu()], dim=0)

CNN_encoding = CNN_encoding.numpy()
scaler = StandardScaler()
CNN_encoding = scaler.fit_transform(CNN_encoding)
np.save(f'pretrained_emb/{args.dataset}_pretrained_emb.npy', CNN_encoding)