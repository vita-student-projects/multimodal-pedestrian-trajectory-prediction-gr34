from tqdm import tqdm
from utils.prerender_utils import get_visualizers, merge_and_save
from utils.nuscenes_conversion import get_full_scene_data
from utils.utils import get_config
from nuscenes import NuScenes
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", type=str, required=True, help="Path to nuscenes data")
    parser.add_argument("--data-path", type=str, required=True, help="Path to nuscenes data")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save data")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    config = get_config(args.config)
    visualizers = get_visualizers(config["renderers"])

    nuscenes = NuScenes(args.data_version, dataroot=args.data_path)
    for scene_id in tqdm(range(len(nuscenes.scene))):
        data = get_full_scene_data(nuscenes, config, scene_id)
        merge_and_save(visualizers=visualizers, data=data, output_path=args.output_path, is_nuscenes=True)


if __name__ == "__main__":
    main()
