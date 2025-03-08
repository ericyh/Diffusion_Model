# %%
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle

# %%
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, time):
        i = torch.arange(self.d)
        even = 1/2*(1-(-1)**i)
        odd = 1/2*(1+(-1)**i)
        x = 1/torch.exp(math.log(10000) * ((i - i%2) / self.d))
        x = time[:, None] * x[None, :]
        x = torch.sin(even*x) + torch.cos(odd*x)
        return x

# %%
class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding = 1),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(),
        )
    def forward(self, x):
        x = self.layers(x)
        return x

# %%
class ResnetBlock(nn.Module):    
    def __init__(self, dim_in, dim_out, time_dim):
        super().__init__()
        self.time = nn.Linear(time_dim, dim_out)
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)

    def forward(self, x, t):
        x = self.block1(x)
        x = torch.unsqueeze(torch.unsqueeze(self.time(t),-1),-1) + x
        x = self.block2(x)
        return x

# %%
class Unet(nn.Module):
    def __init__ (self, time_dim, dims, channels=3):
        super().__init__()

        self.time_embed = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim, 4*time_dim),
                nn.GELU(),
                nn.Linear(4*time_dim, 4*time_dim),
            )
        
        self.init_block = Block(channels, dims[0][0])
        self.encoder = nn.ModuleList([])
        self.mid_block = Block(dims[-1][-1], dims[-1][-1])
        self.decoder = nn.ModuleList([])
        self.final_block = Block(dims[0][0], channels)

        for (dim_in, dim_out) in dims:
            self.encoder.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, 4*time_dim),
                ResnetBlock(dim_out, dim_out, 4*time_dim),
                nn.Conv2d(dim_out,dim_out, 4, 2, 1)
            ]))
        for (dim_in, dim_out) in reversed(dims):
            self.decoder.append(nn.ModuleList([
                ResnetBlock(2*dim_out, dim_in, 4*time_dim),
                ResnetBlock(dim_in, dim_in, 4*time_dim),
                nn.ConvTranspose2d(dim_out, dim_out, 4, 2, 1)
            ]))
        
    def forward(self, x, t):
        x = self.init_block(x)
        t = self.time_embed(t)
        res = []
        for B1, B2, downsample in self.encoder:
            print(x.shape)
            x = B1(x, t)
            x = B2(x, t)
            res.append(x)
            x = downsample(x)
            print(x.shape)
        x = self.mid_block(x)
        for B1, B2, upsample in self.decoder:
            print(x.shape)
            x = upsample(x)
            x = torch.cat((x, res.pop()), dim=1)
            x = B1(x, t)
            x = B2(x, t)
            print(x.shape)
        x = self.final_block(x)

# %%
def beta_schedule(timesteps, s=0.008):
    # from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=5d751df2
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

timesteps = 200
betas = beta_schedule(timesteps=timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# %%
def add_noise(x,t):
    noise = torch.randn_like(x)
    return torch.index_select(sqrt_alphas_cumprod, 0, t) * x + torch.index_select(sqrt_one_minus_alphas_cumprod, 0, t) * noise

# %%
class ImageDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.l = len(self.data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

# %%
def get_data_loader(batch_size=200):
    file = open('C:/VSCode/Datasets/Faces/train_dataset_small.pkl', 'rb')
    dataset = pickle.load(file)
    file.close()
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader


