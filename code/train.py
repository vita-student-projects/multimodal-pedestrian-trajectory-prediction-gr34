import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm.notebook import tqdm as tqdm
from model.modules import MLP, CGBlock, MCGBlock, HistoryEncoder
from model.multipathpp import MultiPathPP
from model.data import *
from model.losses import pytorch_neg_multi_log_likelihood_batch, nll_with_covariances
from prerender.utils.utils import data_to_numpy, get_config
from torch.utils.tensorboard import SummaryWriter
import subprocess
import os
import glob
import sys
import random
from datetime import datetime
import argparse

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR = os.path.join(ROOT_PATH, f"checkpoint/checkpoint_{TIMESTAMP}")
LOG_DIR = os.path.join(ROOT_PATH, "runs", f"run_{TIMESTAMP}")
WRITER = SummaryWriter(log_dir=LOG_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)


def load_checkpoint(model, optimizer, scheduler, checkpoint=None):
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        num_steps = checkpoint.get('num_steps', 0)
        start_epoch = checkpoint.get('epoch', 0)
    else:
        start_epoch = 0
        num_steps = 0
    return model, optimizer, scheduler, num_steps, start_epoch


def train_epoch(model, dataloader, norm_coeffs, optimizer, scheduler, config, num_steps):
    train_losses = []
    # Progress bar for the current epoch
    pbar = tqdm(dataloader, desc="Iterations", dynamic_ncols=True)

    for data in pbar:
        model.train()
        optimizer.zero_grad()

        if config["train"]["normalize"]:
            data = normalize(data, norm_coeffs)

        dict_to_cuda(data)  # Move data to GPU
        probas, coordinates, covariance_matrices, loss_coeff = model(data, num_steps)

        # Check if the outputs are finite
        assert torch.isfinite(coordinates).all()
        assert torch.isfinite(probas).all()
        assert torch.isfinite(covariance_matrices).all()

        # Get the ground truth future coordinates
        xy_future_gt = data["target/future/xy"]

        # Normalise the output if the configuration requires it
        if config["train"]["normalize_output"]:
            coordinates = denormalize_future_xy(coordinates, norm_coeffs)
            assert torch.isfinite(coordinates).all()

        # Compute loss
        loss = nll_with_covariances(
            xy_future_gt, coordinates, probas, data["target/future/valid"].squeeze(-1),
            covariance_matrices) * loss_coeff
        train_losses.append(loss.item())
        loss.backward()

        # Clip the gradients if the configuration requires it
        if "clip_grad_norm" in config["train"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_grad_norm"])

        optimizer.step()
        num_steps += 1

        # Clear GPU memory
        if num_steps % (len(dataloader) // 10) == 0 and num_steps > 0:
            torch.cuda.empty_cache()

        # Update progress bar with iteration number and training loss
        pbar.set_description(f"Iteration {num_steps}")
        pbar.set_postfix(train_loss=loss.item())

        # Log the training loss
        WRITER.add_scalar(f"{config['alias']}/train_loss", loss.item(), num_steps)

        if "max_iterations" in config["train"] and num_steps > config["train"]["max_iterations"]:
            break

    return train_losses, num_steps


def validate_epoch(model, val_dataloader, norm_coeffs, config, epoch, num_steps):
    model.eval()
    val_losses = []
    num_val_samples = len(val_dataloader.dataset)
    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}", dynamic_ncols=True, total=len(val_dataloader))
        for val_data in pbar:
            # Normalise the validation data if the configuration requires it
            if config["train"]["normalize"]:
                val_data = normalize(val_data, norm_coeffs)

            dict_to_cuda(val_data)  # Move data to GPU
            probas, coordinates, covariance_matrices, loss_coeff = model(val_data, num_steps)

            # Get the ground truth future coordinates for validation data
            xy_future_gt = val_data["target/future/xy"]

            # Reverse the normalisation process
            if config["train"]["normalize_output"]:
                coordinates = denormalize_future_xy(coordinates, norm_coeffs)
                assert torch.isfinite(coordinates).all()

            # Compute the validation loss using the model outputs and ground truth
            val_loss = nll_with_covariances(
                xy_future_gt, coordinates, probas, val_data["target/future/valid"].squeeze(-1),
                covariance_matrices) * loss_coeff

            # Log the validation loss
            WRITER.add_scalar(f"{config['alias']}/val_loss", val_loss.item(), epoch)

            # Append the current validation loss to the list of validation losses
            val_losses.append(val_loss.item())

            # Delete unnecessary variables
            del probas, coordinates, covariance_matrices, loss_coeff, xy_future_gt, val_loss
            torch.cuda.empty_cache()  # Clear memory cache

            pbar.set_postfix(val_loss=val_losses[-1])  # Update progress bar with current validation loss

    return val_losses


def train(model, dataloader, val_dataloader, norm_coeffs, optimizer, scheduler, config, num_steps=0, checkpoint=None):
    best_val_loss = float("inf")

    # Load checkpoint if provided
    model, optimizer, scheduler, num_steps, start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint)

    for ep in tqdm(range(start_epoch, config["train"]["n_epochs"]), desc="Epochs"):
        # Train for one epoch and get the training losses and the updated number of steps
        train_losses, num_steps = train_epoch(model, dataloader, norm_coeffs, optimizer, scheduler, config, num_steps)

        # Validate for one epoch and get the validation losses
        val_losses = validate_epoch(model, val_dataloader, norm_coeffs, config, ep, num_steps)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f" Epoch {ep + 1} - Average training loss: {avg_train_loss}, Average validation loss: {avg_val_loss}")

        # If a scheduler is used, step it here
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Save the model if the validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            saving_data = {
                "num_steps": num_steps,
                "epoch": ep + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            if scheduler is not None:
                saving_data["scheduler_state_dict"] = scheduler.state_dict()

            torch.save(saving_data, os.path.join(MODEL_DIR, f"best_{config['alias']}_model.pth"))


def main():
    parser = argparse.ArgumentParser(description='Train MultiPath++ Single')
    parser.add_argument('--train-data-folder', type=str, help='Path to the training data folder')
    parser.add_argument('--val-data-folder', type=str, help='Path to the validation data folder')
    parser.add_argument('--norm-coeffs', type=str, help='Path to the normalization coefficient (.npy) file')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epoch', type=float, default=40, help='Number of epochs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the saved checkpoint')
    args = parser.parse_args()

    # Load normalisation coefficients and move them to GPU
    coeff = np.load(args.norm_coeffs, allow_pickle=True)[()]
    to_cuda(coeff)  # to GPU

    # Load the configuration file
    config = get_config(f"{ROOT_PATH}/code/configs/final_RoP_Cov_Single.yaml")

    # Set the training data path in the configuration
    config["train"]["data_config"]["dataset_config"]["data_path"] = args.train_data_folder
    # Create dataloader for training data
    dataloader = get_dataloader(config["train"]["data_config"])

    # Set the validation data path in the configuration
    config["val"]["data_config"]["dataset_config"]["data_path"] = args.val_data_folder
    # Create dataloader for validation data
    val_dataloader = get_dataloader(config["val"]["data_config"])

    # Set the number of epochs and learning rate in the configuration
    config["train"]["n_epochs"] = args.epoch
    config["train"]["optimizer"]["lr"] = args.lr

    # Init the model and move it to GPU
    model = MultiPathPP(config["model"])
    model = model.cuda()

    # Init the optimizer
    optimizer = Adam(model.parameters(), **config["train"]["optimizer"])

    # Init the scheduler if it's defined in the configuration
    scheduler = None
    if config["train"]["scheduler"]:
        scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

    # Train the model
    train(model, dataloader, val_dataloader, coeff, optimizer, scheduler, config, checkpoint=args.checkpoint)



if __name__ == '__main__':
    print(ROOT_PATH)
