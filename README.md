# CAD-VAE
CAD-VAE: Leveraging Correlation-Aware Latents for Comprehensive Fair Disentanglement, Accepted by AAAI 2026 main track

## Description of the repo
Here is the code of CAD-VAE. This repository contains the following files:

## CAD-VAE in CelebA dataset
- train.py: This Python file is to train CAD-VAE on the CelebA dataset.
- module.py: This contains modules presented in CAD-VAE.
- dataloader.py: This is the dataloader to load the dataset.
- loss.py: This contains the loss function and utility function.
- grid_interpolation_annote.py: generate fine-grained image editing result using trained model.

## CAD-VAE in VLM tasks
- CLIP_VAE.py: This Python file is to train CLIP+CAD-VAE on the CelebA dataset to address fairness issues.
- CLIP_cls_zeroshot.py, CLIP_cls_linear.py: This Python file is to train CLIP on the CelebA dataset.

## datasets
To download the dataset, please visit the following websites and follow the instructions:
- CelebA dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- C-MNIST dataset: https://github.com/alinlab/LfF/tree/master
