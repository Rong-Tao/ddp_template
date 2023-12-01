# DistributedDataParallel PyTorch Template

## Introduction

This template provides a ready-to-use implementation of PyTorch's DistributedDataParallel (DDP) feature. It is designed to facilitate easy and efficient parallelism across multiple GPUs, minimizing the need for extensive code modifications.

## Prerequisites

- PyTorch
- Tensorboard

## Structure

- `main.py`: The main script for DDP implementation. It is recommended not to modify this file.
- `model.py`: Place your model architecture here.
- `util.py`: Define parameters, optimizers, and other utilities.
- `dataloader.py`: Contains the DataLoader and Dataset definitions.

## Usage

To use this template, follow these steps:

1. **Model Setup:** Update your model architecture in `model.py`.
2. **Parameter Configuration:** Specify your training parameters, optimizer, scheduler, etc., in `util.py`.
3. **Data Preparation:** Modify `dataloader.py` to suit your dataset and data loading strategy.
4. **Run the Script:** Use the following command to run the training script:

   ```sbatch main.slurm```
5. **Tensor broad:** Use the following command to run tensorborad

   ```./bd.sh```