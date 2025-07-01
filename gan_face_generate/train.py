import argparse
import random
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
from gan_face_generate.dataset import DigiFace1MDataset
from gan_face_generate.models import Discriminator, Generator


def parse_args() -> Namespace:
    def valid_dir(s: str | None) -> Path | None:
        if s is None:
            return None

        p = Path(s)
        if not p.is_dir():
            print(f"Invalid directory. {p}")
            exit(1)
        return p

    parser = argparse.ArgumentParser(
        prog="Train gan_face_generate",
        description="Train DCGAN model with given dataset",
    )
    parser.add_argument(
        "-d",
        "--data-root",
        type=valid_dir,
        default=None,
        help="the root directory to training dataset.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="checkpoints",
        help="Directory to read/write checkpoints.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="gan_face_generate",
        help="Directory to write weights of generator.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the training process by setting epoch to zero",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    # Set random seed for reproducibility
    manualSeed = random.randint(1, 10000)  # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Root directory for dataset
    dataroot = args.data_root

    # Number of workers for dataloader
    workers = 6

    # Batch size during training
    batch_size = 512

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 0 if args.dry_run else 40

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = DigiFace1MDataset(
        root=dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        ),
    )

    # Create the dataloader
    dataloader: DataLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )

    # Decide which device we want to run on
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu",
    )

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64],
                padding=2,
                normalize=True,
            ).cpu(),
            (1, 2, 0),
        ),
    )

    # custom weights initialization called on ``netG`` and ``netD``

    def weights_init(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # type: ignore [arg-type]
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)  # type: ignore [arg-type]
            nn.init.constant_(m.bias.data, 0)  # type: ignore [arg-type]

    # Create the generator
    netG: Generator | nn.DataParallel[Generator]
    netG = Generator(ngpu, nc, nz, ngf).to(device)

    # Handle multi-GPU if desired
    if (device.type == "cuda") and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD: Discriminator | nn.DataParallel[Discriminator]
    netD = Discriminator(ngpu, nc, ndf).to(device)

    # Handle multi-GPU if desired
    if (device.type == "cuda") and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # W&B init
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="chienliu6001-personal",
        # Set the wandb project where this run will be logged.
        project="train-DCGAN",
        # Track hyperparameters and run metadata.
        config={
            "model-generator": netG.__class__.__name__,
            "model-discriminator": netD.__class__.__name__,
            "dataset": dataset.__class__.__name__,
            "ngpu": ngpu,
            "epochs": 40,
            "learning_rate": lr,
            "size_latent_vector": nz,
            "size_feature_map_generator": ngf,
            "size_feature_map_discriminator": ndf,
        },
    )

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    epoch_st = 0

    # Create directory to save checkpoints
    checkpointRoot = Path(args.checkpoints)
    if not checkpointRoot.exists():
        checkpointRoot.mkdir()

    # Create directory to save weights of Generator
    weightsRoot = Path(args.weights)
    if not weightsRoot.exists():
        weightsRoot.mkdir()
    weightPath = weightsRoot / "weights.tar"

    # Load checkpoints
    checkpointPath = checkpointRoot / "checkpoint.tar"
    if checkpointPath.exists():
        print("Loading chechpoint... ", end="")
        checkpoint = torch.load(checkpointPath)
        netD.load_state_dict(checkpoint["netD_state_dict"])
        netG.load_state_dict(checkpoint["netG_state_dict"])
        optimizerD.load_state_dict(checkpoint["optimizerD_state_dict"])
        optimizerG.load_state_dict(checkpoint["optimizerG_state_dict"])
        img_list = checkpoint["img_list"]
        G_losses = checkpoint["G_losses"]
        D_losses = checkpoint["D_losses"]
        iters = checkpoint["iters"]
        epoch_st = checkpoint["epoch"]

        netG.to(device)
        netD.to(device)

        netG.train()
        netD.train()
        print("Done!")

    print("Starting Training Loop...")

    # For each epoch
    for epoch in range(epoch_st, num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full(
                (b_size,),
                real_label,
                dtype=torch.float,
                device=device,
            )
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    ),
                )
                run.log(
                    {
                        "epoch": epoch,
                        "training_steps": epoch * len(dataloader) + i,
                        "loss_generator": errG.item(),
                        "loss_discriminator": errD.item(),
                        "D_x": D_x,
                        "D_G_z1": D_G_z1,
                        "D_G_z2": D_G_z2,
                    }
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_grid = vutils.make_grid(
                    fake,
                    padding=2,
                    normalize=True,
                )
                wandb.log({"Generated Images": wandb.Image(img_grid)})

                img_list.append(img_grid)

                torch.save(
                    {
                        "netD_state_dict": netD.state_dict(),
                        "netG_state_dict": netG.state_dict(),
                        "optimizerD_state_dict": optimizerD.state_dict(),
                        "optimizerG_state_dict": optimizerG.state_dict(),
                        "img_list": img_list,
                        "G_losses": G_losses,
                        "D_losses": D_losses,
                        "iters": iters,
                        "epoch": epoch,
                    },
                    checkpointPath,
                )
                torch.save(
                    {
                        "netG_state_dict": netG.state_dict(),
                    },
                    weightPath,
                )

                # Upload with W&B Artifacts
                artifact = wandb.Artifact("gan-checkpoint", type="model")
                artifact.add_file(str(checkpointPath))
                artifact.add_file(str(weightPath))
                wandb.log_artifact(artifact)

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.axis("off")

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64],
                padding=5,
                normalize=True,
            ).cpu(),
            (1, 2, 0),
        ),
    )

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    train()
