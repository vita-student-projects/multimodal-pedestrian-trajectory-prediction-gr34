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
import subprocess
import os
import glob
import sys
from eval import WaymoMetrics
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import config_util_py as config_util
import argparse

METRIC_COMPUTATION = WaymoMetrics()
METRIC_NAMES = config_util.get_breakdown_names_from_motion_config(METRIC_COMPUTATION._metrics_config)
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def load_inference_model(config, checkpoint_path):
    """Load pre-trained model for inference"""
    model = MultiPathPP(config["model"])
    model = model.cuda()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def inference(model, inference_dataloader, norm_coeffs, config, num_steps):
    METRIC_COMPUTATION.reset()
    model.eval()
    with torch.no_grad():
        pbar = tqdm(inference_dataloader, desc=f"Inference", dynamic_ncols=True,
                    total=len(inference_dataloader))
        for test_data in pbar:

            # Normalise the test data if the configuration requires it
            if config["train"]["normalize"]:
                test_data = normalize(test_data, norm_coeffs)

            dict_to_cuda(test_data)  # Move data to GPU
            probas, coordinates, _, _ = model(test_data, num_steps)

            # Reverse the normalisation process
            if config["train"]["normalize_output"]:
                # xy_future_gt = normalize_future_xy(xy_future_gt, norm_coeffs)
                coordinates = denormalize_future_xy(coordinates, norm_coeffs)
                assert torch.isfinite(coordinates).all()

            ##################################################
            METRIC_COMPUTATION.update(test_data, coordinates, probas)
            ##################################################

            # Delete unnecessary variables
            del probas, coordinates
            torch.cuda.empty_cache()  # Clear memory cache


def main():
    parser = argparse.ArgumentParser(description='Inference MultiPath++ Single')
    parser.add_argument('--data-folder', type=str, help='Path to the data folder')
    parser.add_argument('--norm-coeffs', type=str, help='Path to the normalization coefficient (.npy) file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the saved checkpoint')
    parser.add_argument('--config', type=str, help='Path to the config file')
    args = parser.parse_args()

    # Load normalisation coefficients and move them to GPU
    coeff = np.load(args.norm_coeffs, allow_pickle=True)[()]
    to_cuda(coeff)  # to GPU

    # Load the configuration file
    config = get_config(args.config)

    # Set the inference data path in the configuration
    config["val"]["data_config"]["dataset_config"]["data_path"] = args.data_folder

    # Create dataloader for validation data
    inf_dataloader = get_dataloader(config["val"]["data_config"])

    model = load_inference_model(config, args.checkpoint)

    inference(model, inf_dataloader, coeff, config, 0)

    ##################################################
    test_metric_values = METRIC_COMPUTATION.result()

    # Open the file in append mode so that new metrics are added to the end of the file
    with open('output.txt', 'w') as f:
        for i, metric in enumerate(
                ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
            for j, name in enumerate(METRIC_NAMES):
                if 'PEDESTRIAN' in name:
                    f.write('{}/{}: {}\n'.format(metric, name, test_metric_values[i][j]))
    ##################################################

if __name__ == '__main__':
    print(ROOT_PATH)
    main()
