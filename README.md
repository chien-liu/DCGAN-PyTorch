# DCGAN
Deep Convolutional Generative Adversarial Networks (DCGAN) is a class of generative adversarial networks (GAN) introduced by Radford et. al. in the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf). The generator and discriminator of DCGAN are contructed of convolutional and convolutional-transpose layers.


## Dataset
[DigiFace-1M](https://github.com/microsoft/DigiFace1M) is a generated dataset for training face recognition models. The face images are high quality and thus are qualified to train a GAN network. There are two additional advantages of the dataset:

* Ethical considerations: The use of existing datasets that were collected from web images without explicit consent. In contrast, digital faces in DigiFace-1M are generated using a generative model constructed from high-quality head scans of a limited number of individuals obtained with consent.

* Data bias - DigiFace-1M is generated in a controlled pipeline, so that the racial distribution is guaranteed to be balance.

## Results
### Loss in Training Process
![](assets/training/loss.png)

### Generated Images compared with Dataset Images
![](assets/results/real_and_fake.png)

### Noise Arithmetic
![](assets/results/noise_arithmetic.png)



## Reference
* [PyTorch DCGAN TUTORIAL](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
* [DigiFace-1M: 1 Million Digital Face Images for Face Recognition](https://github.com/microsoft/DigiFace1M)
