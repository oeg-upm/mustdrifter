import os
import json
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.special import kl_div

import logging
logger = logging.getLogger(__name__)

def js_drift(reference_sample, test_sample, filename):
    reference_sample = np.asarray(reference_sample, dtype=np.float64)
    test_sample = np.asarray(test_sample, dtype=np.float64)

    reference_distribution = reference_sample.mean(axis=0)
    test_distribution = test_sample.mean(axis=0)

    magnitude = float(
        jensenshannon(reference_distribution, test_distribution, base=2.0)
    )

    with open(filename, "w") as f:
        json.dump({"magnitude": magnitude}, f)

    return magnitude


def kl_drift(reference_sample, test_sample, filename, eps=1e-12):
    """
    Computes the Kullback-Leibler Divergence (P || Q).
    
    Args:
        reference_sample (np.ndarray): Baseline distribution.
        test_sample (np.ndarray): Target distribution.
        filename (str): Path to store the drift magnitude.
        eps (float): Small constant to avoid log(0) and division by zero.

    Returns:
        float: Total KL divergence sum.
    """
    p = np.asarray(reference_sample, dtype=np.float64).mean(axis=0)
    q = np.asarray(test_sample, dtype=np.float64).mean(axis=0)

    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()

    magnitude = float(np.sum(kl_div(p, q)))

    with open(filename, "w") as f:
        json.dump({"magnitude": magnitude}, f)

    return magnitude


## ------ Log likelihood drift ------ ##
def _mean_distribution(sample):
    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim == 1:
        return sample
    return sample.mean(axis=0)


def _safe_model_distribution(prob_vector, alpha=1e-12):
    prob_vector = np.asarray(prob_vector, dtype=np.float64)
    prob_vector = prob_vector + alpha
    return prob_vector / prob_vector.sum()


def _average_log_likelihood(sample, model_distribution):
    sample = np.asarray(sample, dtype=np.float64)
    log_model = np.log(model_distribution)

    if sample.ndim == 1:
        return float(np.sum(sample * log_model))

    return float(np.mean(np.sum(sample * log_model, axis=1)))

def _log_likelihood_drift_magnitude(reference_sample, test_sample, alpha=1e-12):
    reference_distribution = _mean_distribution(reference_sample)
    reference_distribution = _safe_model_distribution(reference_distribution, alpha=alpha)

    reference_ll = _average_log_likelihood(reference_sample, reference_distribution)
    test_ll = _average_log_likelihood(test_sample, reference_distribution)

    return float(reference_ll - test_ll)

def run_log_likelihood_permutation(
    permutation,
    aggregated_samples,
    reference_sample_size,
    test_sample_size,
    alpha=1e-12
):
    rng = np.random.default_rng(seed=permutation)
    shuffled = rng.permutation(aggregated_samples)

    permutation_reference_sample = shuffled[:reference_sample_size]
    permutation_test_sample = shuffled[
        reference_sample_size: reference_sample_size + test_sample_size
    ]

    return _log_likelihood_drift_magnitude(
        permutation_reference_sample,
        permutation_test_sample,
        alpha=alpha
    )


def log_likelihood_drift(
    reference_sample,
    test_sample,
    filename,
    K=1000,
    n_jobs=10,
    alpha=1e-12
):
    reference_sample = np.asarray(reference_sample, dtype=np.float64)
    test_sample = np.asarray(test_sample, dtype=np.float64)

    if reference_sample.ndim != 2 or test_sample.ndim != 2:
        raise ValueError("reference_sample and test_sample must be 2D arrays.")

    if reference_sample.shape[1] != test_sample.shape[1]:
        raise ValueError("reference_sample and test_sample must have the same number of columns.")

    reference_sample_size = len(reference_sample)
    test_sample_size = len(test_sample)
    aggregated_samples = np.concatenate([reference_sample, test_sample], axis=0)

    drift_magnitude = _log_likelihood_drift_magnitude(
        reference_sample,
        test_sample,
        alpha=alpha
    )

    permutation_magnitudes = Parallel(n_jobs=n_jobs)(
        delayed(run_log_likelihood_permutation)(
            permutation,
            aggregated_samples,
            reference_sample_size,
            test_sample_size,
            alpha
        )
        for permutation in range(K)
    )

    p_value = (1 + sum(m >= drift_magnitude for m in permutation_magnitudes)) / (K + 1)

    with open(filename, "w") as f:
        json.dump({"magnitude": drift_magnitude, "p_value": p_value}, f)

    return drift_magnitude, p_value
