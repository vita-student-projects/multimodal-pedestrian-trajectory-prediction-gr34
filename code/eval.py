import math
import os
import time
import torch
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import motion_metrics_pb2


def _default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
  track_steps_per_second: 10
  prediction_steps_per_second: 2
  track_history_samples: 10
  track_future_samples: 80
  speed_lower_bound: 1.4
  speed_upper_bound: 11.0
  speed_scale_lower: 0.5
  speed_scale_upper: 1.0
  step_configurations {
    measurement_step: 3
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
  }
  step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.8
    longitudinal_miss_threshold: 3.6
  }
  step_configurations {
    measurement_step: 8
    lateral_miss_threshold: 3.0
    longitudinal_miss_threshold: 6.0
  }
  max_predictions: 6
  """
    text_format.Parse(config_text, config)
    return config


def format_prediction(trajectory, score):
    """
    This function prepares the prediction trajectory and score for evaluation.

    Args:
        trajectory (torch.Tensor): The predicted trajectory. Expected shape is [batch size, n_trajectory, timesteps, xy].
        score (torch.Tensor): The scores corresponding to the predicted trajectories. Expected shape is [batch size, n_trajectory].

    Returns:
        tuple: Contains the reshaped trajectory and score tensors.
            trajectory (torch.Tensor): The reshaped trajectory tensor with the following dimensions: [batch size, num_agents=1, n_trajectory, 1 (mutual independent), timesteps, xy].
            score (torch.Tensor): The reshaped score tensor with the following dimensions: [batch size, num_agent=1, n_trajectory].
    """
    # Detach the tensors from the computation graph, move them to CPU and reshape to match the expected format.
    trajectory = trajectory.detach().cpu()
    trajectory.unsqueeze_(1)
    trajectory.unsqueeze_(3)

    score = score.detach().cpu()
    score.unsqueeze_(1)

    return trajectory, score


def format_ground_truth(data):
    """
    This function prepares the ground truth data for evaluation.

    Args:
        data (dict): A dictionary containing the ground truth data. The keys are the names of the data fields and the values are PyTorch tensors.

    Returns:
        tuple: Contains the reshaped ground truth tensor and its validity tensor.
            ground_truth (torch.Tensor): The reshaped ground truth tensor with the following dimensions: [batch size, num_agents=1, timesteps, 7]
                                         where 7 is composed of [x, y, length, width, heading, velocity_x, velocity_y]
            gt_validity (torch.Tensor): A boolean tensor indicating the validity of the ground truth data. Expected shape is [batch size, num_agents=1, timesteps].
    """
    # The keys of the data fields we are interested in.
    keys = ['xy', 'length', 'width', 'yaw', 'velocity_x', 'velocity_y']

    # Concatenate the history and future ground truth data along the time dimension.
    ground_truth = torch.cat([torch.cat((data[f'target/history/{key}'].cpu(),
                                         data[f'target/future/{key}'].cpu()), dim=1) for key in keys], dim=-1)

    # Concatenate the validity data for the history and future ground truth data along the time dimension.
    gt_validity_float = torch.cat((data['target/history/valid'].cpu(), data['target/future/valid'].cpu()), dim=1)

    # Convert the validity data to boolean type.
    gt_validity = gt_validity_float != 0

    return ground_truth.unsqueeze(1), gt_validity.permute(0, 2, 1)


class WaymoMetrics:
    """Wrapper for waymo motion prediction metrics computation."""

    def __init__(self):
        self._metrics_config = _default_metrics_config()
        self._prediction_trajectory = []
        self._prediction_score = []
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_ground_truth_indices = []
        self._prediction_ground_truth_indices_mask = []
        self._object_type = []

    def reset(self):
        self._prediction_trajectory = []
        self._prediction_score = []
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_ground_truth_indices = []
        self._prediction_ground_truth_indices_mask = []
        self._object_type = []

    def update(self, data, coordinates, probas):
        batch_size = data['target/agent_type'].shape[0]
        pred_gt_indices = torch.zeros(batch_size, 1, 1).cpu().long()
        pred_gt_indices_mask = torch.ones(batch_size, 1, 1, dtype=torch.bool).cpu()
        agent_type = data['target/agent_type'].reshape(-1, 1).cpu()

        pred_trajectory, pred_score = format_prediction(coordinates, probas)
        gt_trajectory, gt_is_valid = format_ground_truth(data)

        self._prediction_trajectory.append(pred_trajectory)
        self._prediction_score.append(pred_score)
        self._ground_truth_trajectory.append(gt_trajectory)
        self._ground_truth_is_valid.append(gt_is_valid)
        self._prediction_ground_truth_indices.append(pred_gt_indices)
        self._prediction_ground_truth_indices_mask.append(pred_gt_indices_mask)
        self._object_type.append(agent_type)

    def result(self):
        # Here we convert tensors to numpy arrays just before we need them

        # [batch_size, num_preds, 1, 1, steps, 2]
        # The ones indicate top_k = 1, num_agents_per_joint_prediction = 1
        prediction_trajectory = tf.concat([t.numpy() for t in self._prediction_trajectory], 0)

        # [batch_size, num_preds, 1]
        prediction_score = tf.concat([s.numpy() for s in self._prediction_score], 0)

        # [batch_size, num_agents, steps, 7]
        ground_truth_trajectory = tf.concat([g.numpy() for g in self._ground_truth_trajectory], 0)

        # [batch_size, num_agents, steps]
        ground_truth_is_valid = tf.concat([g.numpy() for g in self._ground_truth_is_valid], 0)

        # [batch_size, num_preds, 1]
        prediction_ground_truth_indices = tf.concat([p.numpy() for p in self._prediction_ground_truth_indices], 0)

        # [batch_size, num_preds, 1]
        prediction_ground_truth_indices_mask = tf.concat(
            [p.numpy() for p in self._prediction_ground_truth_indices_mask], 0)

        # [batch_size, num_agents]
        object_type = tf.cast(tf.concat([o.numpy() for o in self._object_type], 0), tf.int64)

        # We are predicting more steps than needed by the eval code. Subsample.
        interval = (
                self._metrics_config.track_steps_per_second //
                self._metrics_config.prediction_steps_per_second)
        prediction_trajectory = prediction_trajectory[...,
                                (interval - 1)::interval, :]

        return py_metrics_ops.motion_metrics(
            config=self._metrics_config.SerializeToString(),
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type)


if __name__ == '__main__':
    from waymo_open_dataset.metrics.python import config_util_py as config_util
    # sample usage
    probas, coordinates, covariance_matrices, loss_coeff = model(data, num_steps)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_computation._metrics_config)

    metrics_computation.update(data, coordinates, probas)

    train_metric_values = metrics_computation.result()

    # Open the file in append mode so that new metrics are added to the end of the file
    with open('metrics_output.txt', 'a') as f:
        for i, metric in enumerate(['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
            for j, name in enumerate(metric_names):
                if 'PEDESTRIAN' in name:
                    # Print to the file instead of the console
                    print('{}/{}: {}'.format(metric, name, train_metric_values[i][j]), file=f)