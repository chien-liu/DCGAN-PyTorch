from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch import nn
from torchvision import transforms

from models import Generator

# Set random seed for reproducibility
manualSeed = 777
# manualSeed = random.randint(1, 10000) # use if you want new results
print('Random Seed: ', manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)  # Needed for reproducible results

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device(
    'cuda:0' if (
        torch.cuda.is_available() and ngpu > 0
    ) else 'cpu',
)

# Create the generator
netG = Generator(ngpu, nc, nz, ngf).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Print the model
print(netG)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Lists to save outputs from Generator netG
img_list = []

# Load checkpoints
checkpointRoot = Path('checkpoints')
checkpointPath = checkpointRoot / 'checkpoint.tar'
if checkpointPath.exists():
    checkpoint = torch.load(checkpointPath)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netG.to(device)
    netG.eval()

else:
    sys.exit(0)

print('Starting Inference...')

with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

# Plot the fake images from the last epoch
# Plot the real images
# Create the dataset
dataroot = Path('data/digiface')
image_size = 64
dataset = dset.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
        ),
    ]),
)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64,
    shuffle=True, num_workers=1,
)
real_batch = next(iter(dataloader))

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Real Images')
plt.imshow(
    np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[
                :64
            ], padding=5, normalize=True,
        ).cpu(), (1, 2, 0),
    ),
)

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Fake Images')
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()

# Plot results of noise arithmetic
noise1 = fixed_noise[25, :, :, :]
noise2 = fixed_noise[27, :, :, :]
noise3 = fixed_noise[39, :, :, :]
noise4 = noise1 - noise2 + noise3
noise = torch.stack([noise1, noise2, noise3, noise4], axis=0)
with torch.no_grad():
    fake = netG(noise).detach().cpu()
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.title(
    'Face right profile w/ glasses - Face right profile + \
          Face left profile = Face left profile w/ glasses', fontsize=16,
)
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
