import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader


def angle_to_range(yaw):
    yaw = (yaw - np.pi) % (2 * np.pi) - np.pi
    return yaw


def normalize(data, coefficients):
    means = coefficients['mean']
    stds  = coefficients['std']

    keys = [
        'target/history/lstm_data', 'target/history/lstm_data_diff',
        'other/history/lstm_data', 'other/history/lstm_data_diff',
        'target/history/mcg_input_data', 'other/history/mcg_input_data',
        'road_network_embeddings'
    ]

    for k in keys:
        data[k] = (data[k].cuda() - means[k].cuda()) / (stds[k].cuda() + 1e-6)  # avoid divide by zero
        data[k].clamp_(-15, 15)
        data[k] = data[k].type(torch.float32)  # ensure the data type is float32
                                               # otherwise sometime normalisation will cause the dtype to be float64
                                               # which further cause error in the training process

    data['target/history/lstm_data_diff'] *= data['target/history/valid_diff'].cuda()
    data['other/history/lstm_data_diff'] *= data['other/history/valid_diff'].cuda()
    data['target/history/lstm_data'] *= data['target/history/valid'].cuda()
    data['other/history/lstm_data'] *= data['other/history/valid'].cuda()
    return data


def normalize_future_xy(data, coefficients):
    mean = coefficients['mean']['target/history/xy'].cuda()
    std = coefficients['std']['target/history/xy'].cuda() + 1e-6
    normalized_data = (data.cuda() - mean) / std
    return normalized_data.type(torch.float32)  # ensure the data type is float32


def denormalize_future_xy(data, coefficients):
    mean = coefficients['mean']['target/history/xy'].cuda()
    std = coefficients['std']['target/history/xy'].cuda() + 1e-6
    denormalized_data = data.cuda() * std + mean
    return denormalized_data.type(torch.float32) # ensure the data type is float32


def dict_to_cuda(d):
    passing_keys = set([
        'target/history/lstm_data', 'target/history/lstm_data_diff',
        'other/history/lstm_data', 'other/history/lstm_data_diff',
        'target/history/mcg_input_data', 'other/history/mcg_input_data',
        'other_agent_history_scatter_idx', 'road_network_scatter_idx',
        'other_agent_history_scatter_numbers', 'road_network_scatter_numbers',
        'batch_size',
        'road_network_embeddings',
        'target/future/xy', 'target/future/valid'])
    for k in d.keys():
        if k not in passing_keys:
            continue
        v = d[k]
        if not isinstance(v, torch.Tensor):
            continue
        d[k] = d[k].cuda()


def to_cuda(dict_obj):
    """Used for transferring one batch of data to GPU"""
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            to_cuda(value)
        else:
            if isinstance(value, (np.ndarray, list)):
                dict_obj[key] = torch.tensor(value).cuda()


class MultiPathPPDataset(Dataset):
    def __init__(self, config):
        self._data_path = config["data_path"]
        self._config = config
        files = os.listdir(self._data_path)
        self._files = [os.path.join(self._data_path, f) for f in files]
        random.shuffle(self._files)
        if "max_length" in config:
            self._files = self._files[:config["max_length"]]
        assert len(self._files) > 0

    def __len__(self):
        return len(self._files)

    def _generate_sin_cos(self, data):
        data["target/history/yaw_sin"] = np.sin(data["target/history/yaw"])
        data["target/history/yaw_cos"] = np.cos(data["target/history/yaw"])
        data["other/history/yaw_sin"] = np.sin(data["other/history/yaw"])
        data["other/history/yaw_cos"] = np.cos(data["other/history/yaw"])
        return data

    def _add_length_width(self, data):
        data["target/history/length"] = \
            data["target/length"].reshape(-1, 1, 1) * np.ones_like(data["target/history/yaw"])
        data["target/history/width"] = \
            data["target/width"].reshape(-1, 1, 1) * np.ones_like(data["target/history/yaw"])

        data["other/history/length"] = \
            data["other/length"].reshape(-1, 1, 1) * np.ones_like(data["other/history/yaw"])
        data["other/history/width"] = \
            data["other/width"].reshape(-1, 1, 1) * np.ones_like(data["other/history/yaw"])
        return data

    def _compute_agent_diff_features(self, data):
        diff_keys = ["target/history/xy", "target/history/yaw", "target/history/speed",
                     "other/history/xy", "other/history/yaw", "other/history/speed"]
        for key in diff_keys:
            if key.endswith("yaw"):
                data[f"{key}_diff"] = angle_to_range(np.diff(data[key], axis=1))
            else:
                data[f"{key}_diff"] = np.diff(data[key], axis=1)
        data["target/history/yaw_sin_diff"] = np.sin(data["target/history/yaw_diff"])
        data["target/history/yaw_cos_diff"] = np.cos(data["target/history/yaw_diff"])
        data["other/history/yaw_sin_diff"] = np.sin(data["other/history/yaw_diff"])
        data["other/history/yaw_cos_diff"] = np.cos(data["other/history/yaw_diff"])
        data["target/history/valid_diff"] = (data["target/history/valid"] * \
                                             np.concatenate([
                                                 data["target/history/valid"][:, 1:, :],
                                                 np.zeros((data["target/history/valid"].shape[0], 1, 1))
                                             ], axis=1))[:, :-1, :]
        data["other/history/valid_diff"] = (data["other/history/valid"] * \
                                            np.concatenate([data["other/history/valid"][:, 1:, :],
                                                            np.zeros((data["other/history/valid"].shape[0], 1, 1))],
                                                           axis=1))[:, :-1, :]
        return data

    def _compute_agent_type_and_is_sdc_ohe(self, data, subject):
        I = np.eye(5)
        agent_type_ohe = I[np.array(data[f"{subject}/agent_type"])]
        is_sdc = np.array(data[f"{subject}/is_sdc"]).reshape(-1, 1)
        ohe_data = np.concatenate([agent_type_ohe, is_sdc], axis=-1)[:, None, :]
        ohe_data = np.repeat(ohe_data, data["target/history/xy"].shape[1], axis=1)
        return ohe_data

    def _mask_history(self, ndarray, fraction):
        assert fraction >= 0 and fraction < 1
        ndarray = ndarray * (np.random.uniform(size=ndarray.shape) > fraction)
        return ndarray

    def _compute_lstm_input_data(self, data):
        keys_to_stack = self._config["lstm_input_data"]
        keys_to_stack_diff = self._config["lstm_input_data_diff"]
        for subject in ["target", "other"]:
            agent_type_ohe = self._compute_agent_type_and_is_sdc_ohe(data, subject)
            data[f"{subject}/history/lstm_data"] = np.concatenate(
                [data[f"{subject}/history/{k}"] for k in keys_to_stack] + [agent_type_ohe], axis=-1)
            data[f"{subject}/history/lstm_data"] *= data[f"{subject}/history/valid"]
            data[f"{subject}/history/lstm_data_diff"] = np.concatenate(
                [data[f"{subject}/history/{k}_diff"] for k in keys_to_stack_diff] + \
                [agent_type_ohe[:, 1:, :]], axis=-1)
            data[f"{subject}/history/lstm_data_diff"] *= data[f"{subject}/history/valid_diff"]
        return data

    def _compute_mcg_input_data(self, data):
        for subject in ["target", "other"]:
            agent_type_ohe = self._compute_agent_type_and_is_sdc_ohe(data, subject)
            lstm_input_data = data[f"{subject}/history/lstm_data"]
            I = np.eye(lstm_input_data.shape[1])[None, ...]
            timestamp_ohe = np.repeat(I, lstm_input_data.shape[0], axis=0)
            data[f"{subject}/history/mcg_input_data"] = np.concatenate(
                [lstm_input_data, timestamp_ohe], axis=-1)
        return data

    def _calculate_features(self, idx, np_data):
        np_data["scenario_id"] = np_data["scenario_id"].item()
        np_data["filename"] = self._files[idx]
        np_data["target/history/yaw"] = angle_to_range(np_data["target/history/yaw"])
        np_data["other/history/yaw"] = angle_to_range(np_data["other/history/yaw"])
        np_data = self._generate_sin_cos(np_data)
        np_data = self._add_length_width(np_data)
        if self._config["mask_history"]:
            for subject in ["target", "other"]:
                np_data[f"{subject}/history/valid"] = self._mask_history(
                    np_data[f"{subject}/history/valid"], self._config["mask_history_fraction"])
        np_data = self._compute_agent_diff_features(np_data)
        np_data = self._compute_lstm_input_data(np_data)
        np_data = self._compute_mcg_input_data(np_data)
        return np_data

    def get_item_with_retries(self, idx, read_attempts=5):
        for i in range(read_attempts):
            try:
                np_data = dict(np.load(self._files[idx], allow_pickle=True))
            except:
                if i + 1 == read_attempts:
                    print(f"Skipping {self._files[idx]} due to reading error after {read_attempts} read attempts")
                    return None
                print(f"Retrying to read {self._files[idx]} after read attempt #{i}")
                continue
            break

        return self._calculate_features(idx, np_data)

    def __getitem__(self, idx):
        try:
            np_data = dict(np.load(self._files[idx], allow_pickle=True))
        except:
            print("Error reading", self._files[idx])
            idx = 0
            np_data = dict(np.load(self._files[0], allow_pickle=True))

        return self._calculate_features(idx, np_data)

    @staticmethod
    def collate_fn(batch):
        batch_keys = batch[0].keys()
        result_dict = {k: [] for k in batch_keys}
        other_agent_history_scatter_idx = []
        road_network_scatter_idx = []
        other_agent_history_scatter_numbers = []
        road_network_scatter_numbers = []
        for sample_num, sample in enumerate(batch):
            for k in batch_keys:
                if not isinstance(sample[k], str) and len(sample[k].shape) == 0:
                    result_dict[k].append(sample[k].item())
                else:
                    result_dict[k].append(sample[k])
                if k == "road_network_embeddings":
                    road_network_scatter_idx.extend([sample_num] * sample[k].shape[0])
                    road_network_scatter_numbers.append(sample[k].shape[0])
                if k == "other/history/xy":
                    other_agent_history_scatter_idx.extend([sample_num] * sample[k].shape[0])
                    other_agent_history_scatter_numbers.append(sample[k].shape[0])
        for k, v in result_dict.items():
            if not isinstance(v[0], np.ndarray):
                continue
            result_dict[k] = torch.Tensor(np.concatenate(v, axis=0))
        result_dict["other_agent_history_scatter_idx"] = torch.Tensor(
            other_agent_history_scatter_idx).type(torch.long)
        result_dict["road_network_scatter_idx"] = torch.Tensor(
            road_network_scatter_idx).type(torch.long)
        result_dict["other_agent_history_scatter_numbers"] = torch.Tensor(
            other_agent_history_scatter_numbers).type(torch.long)
        result_dict["road_network_scatter_numbers"] = torch.Tensor(
            road_network_scatter_numbers).type(torch.long)
        result_dict["batch_size"] = len(batch)
        return result_dict


def get_dataloader(config):
    dataset = MultiPathPPDataset(config["dataset_config"])
    dataloader = DataLoader(
        dataset, collate_fn=MultiPathPPDataset.collate_fn, **config["dataloader_config"])
    return dataloader


if __name__ == "__main__":
    import yaml
    from yaml import Loader

    with open(
            "/Users/xuyixuan/Downloads/Project/waymo-motion-prediction-challenge-2022-multipath-plus-plus/code/configs/final_RoP_Cov_Single.yaml",
            'r') as stream:
        config = yaml.load(stream, Loader)

    dataset = MultiPathPPDataset(config["train"]["data_config"]["dataset_config"])

    print(dataset[0])
