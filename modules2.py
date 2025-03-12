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
from torchvision.utils import save_image
import random
import numpy as np
import torch.nn.functional as F

# %%
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, d, device="cuda"):
        super().__init__()
        self.d = d
        self.device = device

    def forward(self, time):
        i = torch.arange(self.d, device = self.device)
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
            nn.GroupNorm(1, out_dim),
            nn.SiLU(),
        )
    def forward(self, x):
        x = self.layers(x)
        return x

# %%
class ResnetBlock(nn.Module):    
    def __init__(self, dim_in, dim_out, time_dim):
        super().__init__()
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out))
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()


    def forward(self, x, t):
        y = self.block1(x)
        t = self.time(t)
        y = torch.unsqueeze(torch.unsqueeze(t,-1),-1) + y
        y = self.block2(y)
        return y + self.res_conv(x)


# %%
class Unet(nn.Module):
    def __init__(
        self,
        dims,
        time_dim = 200,
        channels = 1,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim, 4*time_dim),
                nn.GELU(),
                nn.Linear(4*time_dim, 4*time_dim),
            )

        self.init_conv = nn.Conv2d(channels, dims[0][0], 7, padding=3)

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        for i, (dim_in, dim_out) in enumerate(dims):
            self.encoder.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, 4*time_dim),
                ResnetBlock(dim_out, dim_out, 4*time_dim),
                nn.Conv2d(dim_out,dim_out, 4, 2, 1) if not i == len(dims) - 1 else nn.Identity(),
            ]))
    
        for i, (dim_in, dim_out) in enumerate(reversed(dims)):
            self.decoder.append(nn.ModuleList([
                ResnetBlock(2*dim_out, dim_in, 4*time_dim),
                ResnetBlock(dim_in, dim_in, 4*time_dim),
                nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not i == len(dims) - 1 else nn.Identity(),
            ]))

        mid_dim = dims[-1][-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim=4*time_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim=4*time_dim)

        self.final_block = ResnetBlock(dims[0][0], channels, time_dim=4*time_dim)
        self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time)
        res = []
        for B1, B2, downsample in self.encoder:
            x = B1(x, t)
            x = B2(x, t)
            res.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        for B1, B2, upsample in self.decoder:
            x = torch.cat((x, res.pop()), dim=1)
            x = B1(x, t)
            x = B2(x, t)
            x = upsample(x)
        x = self.final_block(x,t)
        return self.final_conv(x)
# %%
def beta_schedule(timesteps, s=0.008, device="cuda"):
    # from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=5d751df2
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

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

# %%
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# %%
class schedule():
    def __init__(self, timesteps = 200, device="cuda"):
        self.device = device
        betas = beta_schedule(timesteps=timesteps, device = device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.betas = betas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    def add_noise(self, x,t):
        noise = torch.randn_like(x, device = self.device)
        A = torch.index_select(self.sqrt_alphas_cumprod, 0, t)
        B = torch.index_select(self.sqrt_one_minus_alphas_cumprod, 0, t)
        x = x.permute(1,2,3,0)
        noise_permuted = noise.permute(1,2,3,0)
        C = (A * x + B * noise_permuted).permute(3,0,1,2)

        return C, noise
    def loss(self, model, x0, t):
        xt, noise = self.add_noise(x0,t)
        pred = model(xt, t)
        loss = F.smooth_l1_loss(noise, pred)
        return loss
    
    def sample(self, model, device, time_steps):
        M = np.zeros((time_steps, 28, 28, 1))
        img = torch.randn((1, 1, 28, 28), device=device)*0.5
        for t in range(0,time_steps)[::-1]:
            model_mean = self.sqrt_recip_alphas[t] * (img - self.betas[t] * model(img, torch.unsqueeze(torch.tensor(t,device = device), dim=0)) / self.sqrt_one_minus_alphas_cumprod[t])
            noise = torch.randn_like(model_mean)
            img = model_mean + torch.sqrt(self.posterior_variance[t]) * noise
            out = (img[0].detach().cpu().permute(1,2,0)*0.5 + 0.5)*255
            M[t] = out.numpy().astype(np.uint8)
        return np.flip(M, axis = 0)
