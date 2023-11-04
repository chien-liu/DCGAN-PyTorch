# %matplotlib inline
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils

from .models import Generator


def parse_args():
    def valid_path(s: str):
        p = Path(s)
        if not p.parent.exists():
            print(f'Parent directory {p.parent} doesn\'t exist.')
            exit(1)
        if p.suffix not in ['.jpg', '.png', '.svg', '.pdf']:
            print(
                f'Suffix must be either .jpg, .png, .svg, or .pdf. (Given {p.name})',
            )
            exit(1)
        return p

    parser = argparse.ArgumentParser(
        prog='GenerateRandomFace',
        description='Generate a random face created by DCGAN trained with DigiFace1M database',
    )
    parser.add_argument(
        '-s', '--save-file', type=valid_path,
        help='the path to store output image. If not set, the image is shown via matplotlib.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Generate the image without loading pre-train weight, and then neither\
            show or save the output image.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Random initialize random seed
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

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

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)

    # Lists to save outputs from Generator netG
    img_list = []

    # Load checkpoints if arg '--dry-run' is not set
    if not args.dry_run:
        checkpointRoot = Path(__file__).parent
        checkpointPath = checkpointRoot / 'weights.tar'
        if checkpointPath.exists():
            checkpoint = torch.load(checkpointPath)
            netG.load_state_dict(checkpoint['netG_state_dict'])
            netG.to(device)
            netG.eval()
        else:
            sys.exit(1)

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # Plot the fake images from the last epoch

    plt.axis('off')
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))

    if args.dry_run:
        exit(0)

    filename = args.save_file
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == '__main__':
    main()
