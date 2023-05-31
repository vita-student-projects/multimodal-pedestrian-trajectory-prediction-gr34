import yaml
from yaml import Loader
import numpy as np

yaml.warnings({'YAMLLoadWarning': False})


def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader)
    return config


def data_to_numpy(data):
    for k, v in data.items():
        data[k] = v.numpy()
    data["parsed"] = True


def filter_valid(item, valid_array):
    return item[valid_array.flatten() > 0]


def get_filter_valid_roadnetwork_keys():
    filter_valid_roadnetwork = [
        "roadgraph_samples/xyz", "roadgraph_samples/id", "roadgraph_samples/type",
        "roadgraph_samples/valid"]
    return filter_valid_roadnetwork


def get_filter_valid_anget_history():
    result = []
    key_with_different_timezones = ["x", "y", "speed", "bbox_yaw", "valid"]
    common_keys = [
        "state/id", "state/is_sdc", "state/type", "state/current/width", "state/current/length"]
    for key in key_with_different_timezones:
        for zone in ["past", "current", "future"]:
            result.append(f"state/{zone}/{key}")
    result.extend(common_keys)
    return result


if __name__ == "__main__":
    config = get_config('/Users/xuyixuan/Downloads/Project/waymo-motion-prediction-challenge-2022-multipath-plus-plus/code/configs/prerender.yaml')
    print(config)