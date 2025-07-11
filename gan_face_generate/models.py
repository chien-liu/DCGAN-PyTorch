"""
DCGAN Models
============

The models are adapted from PyTorch Tutorial
<https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py>
"""

from torch import Tensor, nn


class Generator(nn.Module):
    """Generator of GAN
    Args
    ----
    ngpu (int): Number of GPUs available. Use 0 for CPU mode.

    nc (int): Number of channels in the training images.
              For color images this is 3, nc = 3.

    nz (int): Size of z latent vector (i.e. size of generator input)

    ngf (int): Size of feature maps in generator.
    """

    def __init__(self, ngpu: int, nc: int, nz: int, ngf: int) -> None:
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.main(input)


class Discriminator(nn.Module):
    """Discriminator of GAN
    Args
    ----
    ngpu (int): Number of GPUs available. Use 0 for CPU mode.

    nc (int): Number of channels in the training images.
              For color images this is 3, nc = 3.

    ndf (int): Size of feature maps in discriminator.
    """

    def __init__(self, ngpu: int, nc: int, ndf: int) -> None:
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.main(input)
