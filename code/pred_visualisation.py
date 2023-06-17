import numpy as np
from prerender.utils.visualize import plot_roadlines
import matplotlib.pyplot as plt
import argparse
from model.data import *
from prerender.utils.utils import get_config
from inference import *

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def plot_arrowbox(center, yaw, length, width, color, alpha=1):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array(((c, -s), (s, c))).reshape(2, 2)
    box = np.array([
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
        [length / 2, width / 2],
        [length * 1.3 / 2, 0],
        [length / 2, -width / 2],
        [-length / 2, -width / 2]])
    box = box @ R.T + center.clone().cpu().numpy()
    plt.plot(box[:, 0], box[:, 1], color=color, alpha=alpha)


def plot_scene_pred(scene_data, coordinates):
    for timezone, color in [('history', 'blue'), ('future', 'yellow')]:
        for i in range(len(scene_data[f"other/{timezone}/xy"])):
            for other_position, other_yaw, other_valid in zip(
                    scene_data[
                        f"other/{timezone}/xy"][i],
                    scene_data[f"other/{timezone}/yaw"][i],
                    scene_data[f"other/{timezone}/valid"][i]):
                if other_valid.item() == 0:
                    continue
                plot_arrowbox(
                    other_position, other_yaw, scene_data["other/history/length"][i][-1],
                    scene_data["target/history/width"][0][-1], color, alpha=0.5)

    for timezone, color in [('history', 'red'), ('future', 'green')]:
        for target_position, target_yaw, target_valid in zip(
                scene_data[f"target/{timezone}/xy"][0],
                scene_data[f"target/{timezone}/yaw"][0, :, 0],
                scene_data[f"target/{timezone}/valid"][0, :, 0]):
            if target_valid == 0:
                continue

            plot_arrowbox(target_position, target_yaw, scene_data["target/history/length"][0][-1],
                          scene_data["target/history/width"][0][-1], color)

    for trajectory_idx in range(coordinates.shape[0]):
        for pred_position, target_yaw, target_valid in zip(
                coordinates[trajectory_idx],
                scene_data["target/future/yaw"][0, :, 0],
                scene_data["target/future/valid"][0, :, 0]):
            if target_valid == 0:
                continue

            plot_arrowbox(pred_position, target_yaw, scene_data["target/history/length"][0][-1],
                          scene_data["target/history/width"][0][-1], "purple")

    plot_roadlines(scene_data["road_network_segments"])
    plt.savefig('trajectories.png')


def pred_one_scene(config, model, scene_data, coeff):
    with torch.no_grad():
        # Normalise the test data if the configuration requires it
        if config["train"]["normalize"]:
            data = normalize(scene_data, coeff)

        dict_to_cuda(data)  # Move data to GPU
        probas, coordinates, _, _ = model(data, 0)

        # Reverse the normalisation process
        if config["train"]["normalize_output"]:
            # xy_future_gt = normalize_future_xy(xy_future_gt, norm_coeffs)
            coordinates = denormalize_future_xy(coordinates, coeff)
            assert torch.isfinite(coordinates).all()

    return coordinates.cpu()[0]


def main():
    parser = argparse.ArgumentParser(description='Pred visualisation MultiPath++ Single')
    parser.add_argument('--data-folder', type=str, help='Path to the data folder')
    parser.add_argument('--norm-coeffs', type=str, help='Path to the normalization coefficient (.npy) file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the saved checkpoint')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--agent-idx', type=int, help='Agent to visualise in the data folder')
    args = parser.parse_args()

    # Load normalisation coefficients and move them to GPU
    coeff = np.load(args.norm_coeffs, allow_pickle=True)[()]
    to_cuda(coeff)  # to GPU

    # Load the configuration file
    config = get_config(args.config)

    # Set the inference data path in the configuration
    config["val"]["data_config"]["dataset_config"]["data_path"] = args.data_folder
    config["val"]["data_config"]["dataloader_config"]["batch_size"] = 1  # plot can only take 1 scene

    # Create dataloader for loading scene
    dataloader = get_dataloader(config["val"]["data_config"])
    dataloader_iter = iter(dataloader)

    def _get_nth_agent_data(data_iter, n):
        agent_data = None
        for i in range(n):
            agent_data = next(data_iter)
        return agent_data

    scene_data = _get_nth_agent_data(dataloader_iter, args.agent_idx)  # load one scene of one agent

    model = load_inference_model(config, args.checkpoint)

    pred = pred_one_scene(config, model, scene_data, coeff)

    # plotting all trajectories on the same plot
    plot_scene_pred(scene_data, pred)


if __name__ == '__main__':
    print(ROOT_PATH)
    main()
