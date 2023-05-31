import argparse
import numpy as np

from tqdm import tqdm

from model.data import MultiPathPPDataset
from prerender.utils.utils import get_config


def generate_batches(dataset, start_index, batch_size):
    """
    Generator function to yield one value at a time from the current batch.

    Args:
        dataset (object): The dataset object from which to retrieve data.
        start_index (int): The starting index for the batch in the dataset.
        batch_size (int): The size of the batch.

    Yields:
        dict: The next value from the dataset within the current batch range.
              If the value is None, it is skipped.
    """
    end_index = min(start_index + batch_size, len(dataset))

    for index in range(start_index, end_index):
        value = dataset.get_item_with_retries(index)
        if value is None:
            continue
        yield value


def get_valid_values(value, key, count, validity_key, respect_validity):
    """
    Function to obtain valid values from a given data structure based on a provided key and validity key.

    Parameters:
    value (dict): The data structure (usually a dictionary) from which to extract the valid values.
    key (str): The key to use to extract the required data from the provided data structure.
    count (int): The number of elements to consider from the beginning of the data array.
    validity_key (str): The key to use to extract the validity data from the provided data structure.
    respect_validity (bool): Flag to indicate whether to consider the validity data while extracting the required data.
    """
    if respect_validity:
        valid_filter = value[validity_key].squeeze(axis=2) > 0
        return value[key][:, :, :count][valid_filter]
    else:
        return value[key][:, :, :count]


def calculate_normalization_coefficients(
        dataset,
        history_timesteps,
        future_timesteps,
        agent_feature_count,
        agent_diff_feature_count,
        road_network_feature_count,
        feature_dimension_map,
        respect_validity,
        batch_size=1000
):
    # Define pairs of keys and their corresponding dimensions
    key_dimension_pairs = [
        ('target/history/agent_features', agent_feature_count),
        ('other/history/agent_features', agent_feature_count),
        ('target/history/agent_features_diff', agent_diff_feature_count),
        ('other/history/agent_features_diff', agent_diff_feature_count),
        ('road_network_embeddings', road_network_feature_count),
        ('target/future/xy', 2)
    ]

    means = {key: np.zeros(dimension) for key, dimension in key_dimension_pairs}
    squared_means = {key: np.zeros(dimension) for key, dimension in key_dimension_pairs}
    counts = {key: 0 for key, _ in key_dimension_pairs}

    dataset_length = len(dataset)

    for start in tqdm(range(0, dataset_length, batch_size)):
        for value in generate_batches(dataset, start, batch_size):
            if value is None:
                continue

            data_pairs = {
                'target/history/agent_features': get_valid_values(value, 'target/history/lstm_data', agent_feature_count, 'target/history/valid', respect_validity),
                'other/history/agent_features': get_valid_values(value, 'other/history/lstm_data', agent_feature_count, 'other/history/valid', respect_validity),
                'target/history/agent_features_diff': get_valid_values(value, 'target/history/lstm_data_diff', agent_diff_feature_count, 'target/history/valid_diff', respect_validity),
                'other/history/agent_features_diff': get_valid_values(value, 'other/history/lstm_data_diff', agent_diff_feature_count, 'other/history/valid_diff', respect_validity),
                'road_network_embeddings': value['road_network_embeddings'][:, :, :road_network_feature_count].squeeze(),
                'target/future/xy': get_valid_values(value, 'target/future/xy', 2, 'target/future/valid', respect_validity)
            }

            # Update counts, means, and squared_means for each key
            for k, v in data_pairs.items():
                counts[k] += v.shape[0]
                np.add(means[k], np.sum(v, axis=0), out=means[k])
                np.add(squared_means[k], np.sum(np.square(v), axis=0), out=squared_means[k])

    # Calculate final means and standard deviations for each key
    means = {k: v / counts[k] for k, v in means.items()}
    stds = {k: np.sqrt(squared_means[k] / counts[k] - np.square(means[k])) for k in means.keys()}

    def _feature_key_to_aggregation_key(key):
        if key == 'target/history/lstm_data':
            return 'target/history/agent_features'
        if key == 'target/history/mcg_input_data':
            return 'target/history/agent_features'

        if key == 'other/history/lstm_data':
            return 'other/history/agent_features'
        if key == 'other/history/mcg_input_data':
            return 'other/history/agent_features'

        if key == 'target/history/lstm_data_diff':
            return 'target/history/agent_features_diff'
        if key == 'other/history/lstm_data_diff':
            return 'other/history/agent_features_diff'

        return key

    result = {'mean': {}, 'std': {}}
    for feature, dim in feature_dimension_map.items():
        mean = means[_feature_key_to_aggregation_key(feature)]
        std = stds[_feature_key_to_aggregation_key(feature)]

        result['mean'][feature] = np.concatenate([mean, np.zeros(dim - mean.size)])
        result['std'][feature] = np.concatenate([std, np.ones(dim - std.size)])

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to pre-rendered data")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save normalizations")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    config = get_config(args.config)
    config['dataset_config']['data_path'] = args.data_path

    dataset = MultiPathPPDataset(config["dataset_config"])

    result = calculate_normalization_coefficients(
        dataset,
        history_timesteps=config['history_timesteps'],
        future_timesteps=config['future_timesteps'],
        agent_feature_count=config['agent_feature_count'],
        agent_diff_feature_count=config['agent_diff_feature_count'],
        road_network_feature_count=config['road_network_feature_count'],
        feature_dimension_map=config['feature_dimension_map'],
        respect_validity=config['respect_validity']
    )

    np.save(args.output_path, result)


if __name__ == "__main__":
    main()
