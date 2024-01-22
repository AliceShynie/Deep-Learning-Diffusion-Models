import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torch import nn
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

#====================================================================================================
# Resources
# https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=i7AZkYjKgQTm
#====================================================================================================


#====================================================================================================
# Loading Images into Torchvision and Dataloader
#====================================================================================================

IMG_SIZE = 64

BATCH_SIZE = 25

#path_1 = "c:/Users/selva/OneDrive - Durham University/Year 3/Deep Learning/Coursework/STL10/img/"

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    # data_transforms = [
    #     transforms.ToTensor(), # Scales data into [0,1] 
    #     transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    #     transforms.Lambda(lambda x: x.repeat(3,1,1)), 
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    # ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.ImageFolder(root=path_1, 
                                         transform=data_transform)

    return train

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

# data = load_transformed_dataset()

# dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

#============================================================================================================
# UNET Architecture
#============================================================================================================

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

#============================================================================================================
# Functions
#============================================================================================================

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)



#============================================================================================================
# Functions for sampling
#============================================================================================================

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(num):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    #plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 1
    stepsize = int(T/num_images)
    
    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            #plt.subplot(1, num_images, int(i/stepsize+1))
            show_tensor_image(img.detach().cpu())
    plt.savefig(f'DDPM_youtube_SimpleUNET_{str(num)}.png',bbox_inches='tight',pad_inches = 0)
               
@torch.no_grad()
def sample_plot_image_64():
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    num_images = 1
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            show_tensor_image(img.detach().cpu())

def interpolation_tensor():
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    num_images = 1
    stepsize = int(T/num_images)
    
    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            return img
            
#============================================================================================================
# Caculations based on DDPM paper
#============================================================================================================

# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

#============================================================================================================
# Calling the model
#============================================================================================================

model = SimpleUnet()

#============================================================================================================
# Loading weights into model 
#============================================================================================================
import logging

logging.basicConfig(filename = "DDPM_youtube_SimpleUNET.log", format='%(asctime)s %(message)s', filemode='w', level=logging.INFO)

PATH = './DDPM_youtube_SimpleUNET.pth'

model.to(device)

model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

#============================================================================================================
# Training
#============================================================================================================

# print("Training now....")
# optimizer = Adam(model.parameters(), lr=0.001)
# epochs = 100000 # Try more!
# for epoch in range(epochs):
#     for step, batch in enumerate(dataloader):
#       optimizer.zero_grad()

#       t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
#       loss = get_loss(model, batch[0], t)
#       loss.backward()
#       optimizer.step()

#       if epoch % 5 == 0 and step == 0:
#         print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
#         logging.info(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
#         torch.save(model.state_dict(), PATH)
#         sample_plot_image()

#============================================================================================================
# Evaluation and obtaining samples
#============================================================================================================

model.eval()
# w = 10
# h = 10
# fig = plt.figure(figsize=(8, 8))
# columns = 2
# rows = 2
# print("Obtaining samples now....")
# for i in range(1, columns*rows +1):
#     fig.subplots_adjust(wspace=0, hspace=0)
#     fig.add_subplot(rows, columns, i).axis('off')
#     sample_plot_image_64()
# plt.savefig(f'DDPM_youtube_SimpleUNET_64_1.png',bbox_inches='tight',pad_inches = 0)
# plt.show()


#============================================================================================================
# Interpolation between two images
# https://nn.labml.ai/diffusion/ddpm/evaluate.html
#============================================================================================================

from labml import  monit
from labml_nn.diffusion.ddpm import  gather

q_sample = sample_timestep
sigma2 = betas


def p_sample(xt: torch.Tensor, t: torch.Tensor):
    alpha_bar = alphas_cumprod
    alpha = alphas
    x_noisy, noise = forward_diffusion_sample(xt, t, device)
    eps_theta = noise
    alpha_bar = gather(alpha_bar, t)
    alpha = gather(alpha, t)
    eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
    mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
    var = gather(sigma2, t)
    eps = torch.randn(xt.shape, device=xt.device)
    return mean + (var ** .5) * eps

def _sample_x0( xt: torch.Tensor, n_steps: int):
    n_samples = xt.shape[0]
    for t_ in monit.iterate('Denoise', n_steps):
        t = n_steps - t_ - 1
        xt = p_sample(xt, xt.new_full((n_samples,), t, dtype=torch.long))
    return xt

def interpolate(x1: torch.Tensor, x2: torch.Tensor, lambda_: float, t_: int = 100):
    n_samples = x1.shape[0]
    t = torch.full((n_samples,), t_, device=device)
    xt = (1 - lambda_) * q_sample(x1, t) + lambda_ * q_sample(x2, t)
    return _sample_x0(xt, t_)

x1 = interpolation_tensor()
x2 = interpolation_tensor()


output1 = interpolate(x1,x2,0.25,1)
output = interpolate(x1,x2,0.5,1)
output2 = interpolate(x1,x2,0.75,1)

fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(wspace=0, hspace=0)

fig.add_subplot(1, 5, 1).axis('off')
show_tensor_image(x1.detach().cpu())

fig.add_subplot(1, 5, 2).axis('off')
show_tensor_image(output1.detach().cpu())

fig.add_subplot(1, 5, 3).axis('off')
show_tensor_image(output.detach().cpu())

fig.add_subplot(1, 5, 4).axis('off')
show_tensor_image(output2.detach().cpu())

fig.add_subplot(1, 5, 5).axis('off')
show_tensor_image(x2.detach().cpu())
plt.show()