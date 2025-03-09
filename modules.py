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

# %%
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, d, device):
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
    def __init__ (self, time_dim, dims, channels=3, device = "cuda"):
        super().__init__()

        self.time_embed = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim, device),
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
            x = B1(x, t)
            x = B2(x, t)
            res.append(x)
            x = downsample(x)
        x = self.mid_block(x)
        for B1, B2, upsample in self.decoder:
            x = upsample(x)
            x = torch.cat((x, res.pop()), dim=1)
            x = B1(x, t)
            x = B2(x, t)
        x = self.final_block(x)
        return x

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
    file = open('C:/VSCode/Datasets/Faces/test_dataset_small.pkl', 'rb')
    dataset = pickle.load(file)
    file.close()
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader

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
        noise = noise.permute(1,2,3,0)
        C = (A * x + B * noise).permute(3,0,1,2)
        return C
    def loss(self, model, x0, t):
        noise = torch.randn_like(x0, device = self.device)
        xt = self.add_noise(x0,t)
        pred = model(xt, t)
        #save_image(x0[0], str("results/" + str(random.randint(0,100))+".png"), nrow = 6)
        loss = torch.mean((noise - pred)**2, dim=(1,2,3))
        return loss
    
    @torch.no_grad()
    def sample(self, model, device, time_steps):
        img = torch.randn((1, 3, 112, 112), device=device)*0.5
        img = torch.clamp(img, -1.0, 1.0)
        for t in range(0,time_steps)[::-1]:
            model_mean = self.sqrt_recip_alphas[t] * (img - self.betas[t] * model(img, torch.unsqueeze(torch.tensor(t,device = device), dim=0)) / self.sqrt_one_minus_alphas_cumprod[t])
            noise = torch.randn_like(model_mean)
            img = model_mean + torch.sqrt(self.posterior_variance[t]) * noise
            img = torch.clamp(img, -1.0, 1.0)
            plt.imshow(img[0].detach().cpu())
            plt.show()



