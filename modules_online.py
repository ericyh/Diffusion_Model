# %% [markdown]
# <h1>
# 	The Annotated Diffusion Model
# </h1>
# 
# 
# <div class="author-card">
#     <a href="/nielsr">
#         <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/48327001?v=4" width="100" title="Gravatar">
#         <div class="bfc">
#             <code>nielsr</code>
#             <span class="fullname">Niels Rogge</span>
#         </div>
#     </a>
#     <a href="/kashif">
#         <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/8100?v=4" width="100" title="Gravatar">
#         <div class="bfc">
#             <code>kashif</code>
#             <span class="fullname">Kashif Rasul</span>
#         </div>
#     </a>
#     
# </div>
# 
# <script async defer src="https://unpkg.com/medium-zoom-element@0/dist/medium-zoom-element.min.js"></script>
# 
# 

# %% [markdown]
# In this blog post, we'll take a deeper look into **Denoising Diffusion Probabilistic Models** (also known as DDPMs, diffusion models, score-based generative models or simply [autoencoders](https://benanne.github.io/2022/01/31/diffusion.html)) as researchers have been able to achieve remarkable results with them for (un)conditional image/audio/video generation. Popular examples (at the time of writing) include [GLIDE](https://arxiv.org/abs/2112.10741) and [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI, [Latent Diffusion](https://github.com/CompVis/latent-diffusion) by the University of Heidelberg and [ImageGen](https://imagen.research.google/) by Google Brain.
# 
# We'll go over the original DDPM paper by ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)), implementing it step-by-step in PyTorch, based on Phil Wang's [implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) - which itself is based on the [original TensorFlow implementation](https://github.com/hojonathanho/diffusion). Note that the idea of diffusion for generative modeling was actually already introduced in ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)). However, it took until ([Song et al., 2019](https://arxiv.org/abs/1907.05600)) (at Stanford University), and then ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) (at Google Brain) who independently improved the approach.
# 
# Note that there are [several perspectives](https://twitter.com/sedielem/status/1530894256168222722?s=20&t=mfv4afx1GcNQU5fZklpACw) on diffusion models. Here, we employ the discrete-time (latent variable model) perspective, but be sure to check out the other perspectives as well.

# %% [markdown]
# Alright, let's dive in!

# %% [markdown]
# <p align="center">
# <img src='https://drive.google.com/uc?id=11C3cBUfz7_vrkj_4CWCyePaQyr-0m85_' width=500>
# </p>

import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

# %% [markdown]
# 
# ## What is a diffusion model?
# 
# A (denoising) diffusion model isn't that complex if you compare it to other generative models such as Normalizing Flows, GANs or VAEs: they all convert noise from some simple distribution to a data sample. This is also the case here where **a neural network learns to gradually denoise data** starting from pure noise.
# 
# In a bit more detail for images, the set-up consists of 2 processes:
# * a fixed (or predefined) forward diffusion process $q$ of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise
# * a learned reverse denoising diffusion process $p_\theta$, where a neural network is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.
# 
# <p align="center">
#     <img src="https://drive.google.com/uc?id=1t5dUyJwgy2ZpDAqHXw7GhUAp2FE5BWHA" width="600" />
# </p>
# 
# Both the forward and reverse process indexed by \\(t\\) happen for some number of finite time steps \\(T\\) (the DDPM authors use \\(T=1000\\)). You start with \\(t=0\\) where you sample a real image \\(\mathbf{x}_0\\) from your data distribution (let's say an image of a cat from ImageNet), and the forward process samples some noise from a Gaussian distribution at each time step \\(t\\), which is added to the image of the previous time step. Given a sufficiently large \\(T\\) and a well behaved schedule for adding noise at each time step, you end up with what is called an [isotropic Gaussian distribution](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) at \\(t=T\\) via a gradual process.
# 
# ## In more mathematical form
# 
# Let's write this down more formally, as ultimately we need a tractable loss function which our neural network needs to optimize.
# 
# Let \\(q(\mathbf{x}_0)\\) be the real data distribution, say of "real images". We can sample from this distribution to get an image, \\(\mathbf{x}_0 \sim q(\mathbf{x}_0)\\). We define the forward diffusion process \\(q(\mathbf{x}_t | \mathbf{x}_{t-1})\\) which adds Gaussian noise at each time step \\(t\\), according to a known variance schedule \\(0 < \beta_1 < \beta_2 < ... < \beta_T < 1\\) as
# $$
# q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}).
# $$
# 
# Recall that a normal distribution (also called Gaussian distribution) is defined by 2 parameters: a mean \\(\mu\\) and a variance \\(\sigma^2 \geq 0\\). Basically, each new (slightly noiser) image at time step \\(t\\) is drawn from a **conditional Gaussian distribution** with \\(\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}\\) and \\(\sigma^2_t = \beta_t\\), which we can do by sampling \\(\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\\) and then setting \\(\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\epsilon}\\).
# 
# Note that the \\(\beta_t\\) aren't constant at each time step \\(t\\) (hence the subscript) --- in fact one defines a so-called **"variance schedule"**, which can be linear, quadratic, cosine, etc. as we will see further (a bit like a learning rate schedule).
# 
# So starting from \\(\mathbf{x}_0\\), we end up with \\(\mathbf{x}_1,  ..., \mathbf{x}_t, ..., \mathbf{x}_T\\), where \\(\mathbf{x}_T\\) is pure Gaussian noise if we set the schedule appropriately.
# 
# Now, if we knew the conditional distribution \\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\), then we could run the process in reverse: by sampling some random Gaussian noise \\(\mathbf{x}_T\\), and then gradually "denoise" it so that we end up with a sample from the real distribution \\(\mathbf{x}_0\\).
# 
# However, we don't know \\(p(\mathbf{x}_{t-1} | \mathbf{x}_t)\\). It's intractable since it requires knowing the distribution of all possible images in order to calculate this conditional probability. Hence, we're going to leverage a neural network to **approximate (learn) this conditional probability distribution**, let's call it \\(p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t)\\), with \\(\theta\\) being the parameters of the neural network, updated by gradient descent.
# 
# Ok, so we need a neural network to represent a (conditional) probability distribution of the backward process. If we assume this reverse process is Gaussian as well, then recall that any Gaussian distribution is defined by 2 parameters:
# * a mean parametrized by \\(\mu_\theta\\);
# * a variance parametrized by \\(\Sigma_\theta\\);
# 
# so we can parametrize the process as
# $$ p_\theta (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))$$
# where the mean and variance are also conditioned on the noise level \\(t\\).

# %% [markdown]
# Hence, our neural network needs to learn/represent the mean and variance. However, the DDPM authors decided to **keep the variance fixed, and let the neural network only learn (represent) the mean \\(\mu_\theta\\) of this conditional probability distribution**. From the paper:
# 
# > First, we set \\(\Sigma_\theta ( \mathbf{x}_t, t) = \sigma^2_t \mathbf{I}\\) to untrained time dependent constants. Experimentally, both \\(\sigma^2_t = \beta_t\\) and \\(\sigma^2_t  = \tilde{\beta}_t\\) (see paper) had similar results.
# 
# This was then later improved in the [Improved diffusion models](https://openreview.net/pdf?id=-NEXDKk8gZ) paper, where a neural network also learns the variance of this backwards process, besides the mean.
# 
# So we continue, assuming that our neural network only needs to learn/represent the mean of this conditional probability distribution.
# 
# ## Defining an objective function (by reparametrizing the mean)
# 
# To derive an objective function to learn the mean of the backward process, the authors observe that the combination of \\(q\\) and \\(p_\theta\\) can be seen as a variational auto-encoder (VAE) [(Kingma et al., 2013)](https://arxiv.org/abs/1312.6114). Hence, the **variational lower bound** (also called ELBO) can be used to minimize the negative log-likelihood with respect to ground truth data sample \\(\mathbf{x}_0\\) (we refer to the VAE paper for details regarding ELBO). It turns out that the ELBO for this process is a sum of losses at each time step \\(t\\), \\(L = L_0 + L_1 + ... + L_T\\). By construction of the forward \\(q\\) process and backward process, each term (except for \\(L_0\\)) of the loss is actually the **KL divergence between 2 Gaussian distributions** which can be written explicitly as an L2-loss with respect to the means!
# 
# A direct consequence of the constructed forward process \\(q\\), as shown by Sohl-Dickstein et al., is that we can sample \\(\mathbf{x}_t\\) at any arbitrary noise level conditioned on \\(\mathbf{x}_0\\) (since sums of Gaussians is also Gaussian). This is very convenient:  we don't need to apply \\(q\\) repeatedly in order to sample \\(\mathbf{x}_t\\).
# We have that
# $$q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})$$
# 
# with \\(\alpha_t := 1 - \beta_t\\) and \\(\bar{\alpha}t := \Pi_{s=1}^{t} \alpha_s\\). Let's refer to this equation as the "nice property". This means we can sample Gaussian noise and scale it appropriatly and add it to \\(\mathbf{x}_0\\) to get \\(\mathbf{x}_t\\) directly. Note that the \\(\bar{\alpha}_t\\) are functions of the known \\(\beta_t\\) variance schedule and thus are also known and can be precomputed. This then allows us, during training, to **optimize random terms of the loss function \\(L\\)** (or in other words, to randomly sample \\(t\\) during training and optimize \\(L_t\\).

# %% [markdown]
# Another beauty of this property, as shown in Ho et al. is that one can (after some math, for which we refer the reader to [this excellent blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)) instead **reparametrize the mean to make the neural network learn (predict) the added noise (via a network \\(\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)\\) for noise level \\(t\\)** in the KL terms which constitute the losses. This means that our neural network becomes a noise predictor, rather than a (direct) mean predictor. The mean can be computed as follows:
# 
# $$ \mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(  \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$
# 
# The final objective function \\(L_t\\) then looks as follows (for a random time step \\(t\\) given \\(\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\\) ):
# 
# $$ \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 = \| \mathbf{\epsilon} - \mathbf{\epsilon}_\theta( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\epsilon}, t) \|^2.$$

# %% [markdown]
# Here, \\(\mathbf{x}_0\\) is the initial (real, uncorruped) image, and we see the direct noise level \\(t\\) sample given by the fixed forward process. \\(\mathbf{\epsilon}\\) is the pure noise sampled at time step \\(t\\), and \\(\mathbf{\epsilon}_\theta (\mathbf{x}_t, t)\\) is our neural network. The neural network is optimized using a simple mean squared error (MSE) between the true and the predicted Gaussian noise.
# 
# The training algorithm now looks as follows:
# 
# 
# <p align="center">
#     <img src="https://drive.google.com/uc?id=1LJsdkZ3i1J32lmi9ONMqKFg5LMtpSfT4" width="400" />
# </p>
# 
# In other words:
# * we take a random sample $\mathbf{x}_0$ from the real unknown and possibily complex data distribution $q(\mathbf{x}_0)$
# * we sample a noise level $t$ uniformally between $1$ and $T$ (i.e., a random time step)
# * we sample some noise from a Gaussian distribution and corrupt the input by this noise at level $t$ using the nice property defined above
# * the neural network is trained to predict this noise based on the corruped image $\mathbf{x}_t$, i.e. noise applied on $\mathbf{x}_0$ based on known schedule $\beta_t$
# 
# In reality, all of this is done on batches of data as one uses stochastic gradient descent to optimize neural networks.
# 
# ## The neural network
# 
# The neural network needs to take in a noised image at a particular time step and return the predicted noise. Note that the predicted noise is a tensor that has the same size/resolution as the input image. So technically, the network takes in and outputs tensors of the same shape. What type of neural network can we use for this?
# 
# What is typically used here is very similar to that of an [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder), which you may remember from typical "intro to deep learning" tutorials. Autoencoders have a so-called "bottleneck" layer in between the encoder and decoder. The encoder first encodes an image into a smaller hidden representation called the "bottleneck", and the decoder then decodes that hidden representation back into an actual image. This forces the network to only keep the most important information in the bottleneck layer.
# 
# In terms of architecture, the DDPM authors went for a **U-Net**, introduced by ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) (which, at the time, achieved state-of-the-art results for medical image segmentation). This network, like any autoencoder, consists of a bottleneck in the middle that makes sure the network learns only the most important information. Importantly, it introduced residual connections between the encoder and decoder, greatly improving gradient flow (inspired by ResNet in [He et al., 2015](https://arxiv.org/abs/1512.03385)).
# 
# <p align="center">
#     <img src="https://drive.google.com/uc?id=1_Hej_VTgdUWGsxxIuyZACCGjpbCGIUi6" width="400" />
# </p>
# 
# As can be seen, a U-Net model first downsamples the input (i.e. makes the input smaller in terms of spatial resolution), after which upsampling is performed.
# 
# Below, we implement this network, step-by-step.
# 
# ### Network helpers
# 
# First, we define some helper functions and classes which will be used when implementing the neural network. Importantly, we define a `Residual` module, which simply adds the input to the output of a particular function (in other words, adds a residual connection to a particular function).
# 
# We also define aliases for the up- and downsampling operations.

# %%
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

# %%
from modules2 import SinusoidalPositionEmbeddings
# %%
from modules2 import ResnetBlock

# %%
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# %% [markdown]
# ### Conditional U-Net
# 
# Now that we've defined all building blocks (position embeddings, ResNet/ConvNeXT blocks, attention and group normalization), it's time to define the entire neural network. Recall that the job of the network \\(\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)\\) is to take in a batch of noisy images + noise levels, and output the noise added to the input. More formally:
# 
# - the network takes a batch of noisy images of shape `(batch_size, num_channels, height, width)` and a batch of noise levels of shape `(batch_size, 1)` as input, and returns a tensor of shape `(batch_size, num_channels, height, width)`
# 
# The network is built up as follows:
# * first, a convolutional layer is applied on the batch of noisy images, and position embeddings are computed for the noise levels
# * next, a sequence of downsampling stages are applied. Each downsampling stage consists of 2 ResNet/ConvNeXT blocks + groupnorm + attention + residual connection + a downsample operation
# * at the middle of the network, again ResNet or ConvNeXT blocks are applied, interleaved with attention
# * next, a sequence of upsampling stages are applied. Each upsampling stage consists of 2 ResNet/ConvNeXT blocks + groupnorm + attention + residual connection + an upsample operation
# * finally, a ResNet/ConvNeXT block followed by a convolutional layer is applied.
# 
# Ultimately, neural networks stack up layers as if they were lego blocks (but it's important to [understand how they work](http://karpathy.github.io/2019/04/25/recipe/)).
# 

from modules2 import Block
# %%
class Unet(nn.Module):
    def __init__(
        self,
        dims,
        time_dim = 200,
        channels=3,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim, 4*time_dim),
                nn.GELU(),
                nn.Linear(4*time_dim, 4*time_dim),
            )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i, (dim_in, dim_out) in enumerate(dims):
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, 4*time_dim),
                ResnetBlock(dim_out, dim_out, 4*time_dim),
                nn.Conv2d(dim_out,dim_out, 4, 2, 1) if not i == len(dims) - 1 else nn.Identity(),
            ]))
    
        for i, (dim_in, dim_out) in enumerate(reversed(dims)):
            self.ups.append(nn.ModuleList([
                ResnetBlock(2*dim_out, dim_in, 4*time_dim),
                ResnetBlock(dim_in, dim_in, 4*time_dim),
                nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not i == len(dims) - 1 else nn.Identity(),
            ]))


        mid_dim = dims[-1][-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim=4*time_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim=4*time_dim)


        out_dim = default(out_dim, channels)
        self.final_block = ResnetBlock(dims[0][0], out_dim, time_dim=4*time_dim)
        self.final_conv = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            #print(x.shape, "a")
            x = downsample(x)
            #print(x.shape, "b")
        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, upsample in self.ups:
            #print(x.shape, "c")
            x = torch.cat((x, h.pop()), dim=1)
            #print(x.shape, "d")
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)
            #print(x.shape, "e")

        x = self.final_block(x,t)

        return self.final_conv(x)

# %% [markdown]
# ## Defining the forward diffusion process
# 
# The forward diffusion process gradually adds noise to an image from the real distribution, in a number of time steps $T$. This happens according to a **variance schedule**. The original DDPM authors employed a linear schedule:
# 
# > We set the forward process variances to constants
# increasing linearly from $\beta_1 = 10^{−4}$
# to $\beta_T = 0.02$.
# 
# However, it was shown in ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672)) that better results can be achieved when employing a cosine schedule.
# 
# Below, we define various schedules for the $T$ timesteps, as well as corresponding variables which we'll need, such as cumulative variances.

# %%
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# %% [markdown]
# To start with, let's use the linear schedule for \\(T=200\\) time steps and define the various variables from the \\(\beta_t\\) which we will need, such as the cumulative product of the variances \\(\bar{\alpha}_t\\). Each of the variables below are just 1-dimensional tensors, storing values from \\(t\\) to \\(T\\). Importantly, we also define an `extract` function, which will allow us to extract the appropriate \\(t\\) index for a batch of indices.
# 

# %%
timesteps = 200

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# %% [markdown]
# We'll illustrate with a cats image how noise is added at each time step of the diffusion process.