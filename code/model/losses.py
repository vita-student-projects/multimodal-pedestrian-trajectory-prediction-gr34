import torch
from torch import nn
import numpy as np
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal


def nll_with_covariances(gt, predictions, confidences, avails, covariance_matrices):
    """
    Calculate the negative log-likelihood (NLL) loss for predicted object trajectories with covariances.

    Args:
    gt (torch.Tensor): Ground truth trajectories with shape (batch_size, num_coords, 2).
    predictions (torch.Tensor): Predicted trajectories with shape (batch_size, num_modes, num_coords, 2).
    confidences (torch.Tensor): Confidence scores for each predicted trajectory with shape (batch_size, num_modes).
    avails (torch.Tensor): Availability of ground truth coordinates with shape (batch_size, num_coords).
    covariance_matrices (torch.Tensor): Covariance matrices for each predicted trajectory with shape (batch_size, num_modes, num_coords, 2, 2).

    Returns:
    torch.Tensor: Mean NLL loss across the batch.
    """

    # Compute the precision matrices by inverting covariance_matrices
    precision_matrices = torch.inverse(covariance_matrices)

    # Add an extra dimension to gt for broadcasting
    gt = torch.unsqueeze(gt, 1)

    # Add an extra dimension to avails for broadcasting
    avails = avails[:, None, :, None]

    # Compute the difference between gt and predictions
    coordinates_delta = (gt - predictions).unsqueeze(-1)

    # Compute the errors
    errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta

    # Compute the log-likelihoods
    errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * torch.logdet(covariance_matrices).unsqueeze(-1))

    # Ensure that all elements in errors tensor are finite
    assert torch.isfinite(errors).all()

    # Calculate log-softmax of confidences and add it to the sum of errors
    with np.errstate(divide="ignore"):
        errors = nn.functional.log_softmax(confidences, dim=1) + \
                 torch.sum(errors, dim=[2, 3])

    # Compute the negative log-likelihood loss
    errors = -torch.logsumexp(errors, dim=-1, keepdim=True)

    # Return the mean NLL loss across the batch
    return torch.mean(errors)


def pytorch_neg_multi_log_likelihood_batch(gt, predictions, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        predictions (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords
    error = torch.sum(
        ((gt - predictions) * avails) ** 2, dim=-1
    )  # reduce coords and use availability
    with np.errstate(
            divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time
    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)
    return torch.mean(error)
