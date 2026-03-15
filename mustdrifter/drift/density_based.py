import os
import json
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.special import kl_div

import logging
logger = logging.getLogger(__name__)

import os 
# Needed for parallel processing to ensure that all CPU cores are utilized effectively
os.system("taskset -p 0xff %d" % os.getpid())


def js_drift(reference_sample, test_sample, filename):
    logger.info("Running JS drift detection.")
    with open(filename, "w") as f:
        json.dump({}, f)
        logger.debug(f"Initialized empty JSON file for drift results: {filename}")
    reference_sample = np.asarray(reference_sample, dtype=np.float64)
    test_sample = np.asarray(test_sample, dtype=np.float64)

    reference_distribution = reference_sample.mean(axis=0)
    test_distribution = test_sample.mean(axis=0)

    logger.info("Computed mean distributions for reference and test samples.")

    magnitude = float(
        jensenshannon(reference_distribution, test_distribution, base=2.0)
    )

    with open(filename, "w") as f:
        json.dump({"magnitude": magnitude}, f)

    logger.info(f"JS drift magnitude ({magnitude}) saved to {filename}")
    
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
    
    logger.info("Running KL drift detection.")
    with open(filename, "w") as f:
        json.dump({}, f)
        logger.debug(f"Initialized empty JSON file for drift results: {filename}")

    p = np.asarray(reference_sample, dtype=np.float64).mean(axis=0)
    q = np.asarray(test_sample, dtype=np.float64).mean(axis=0)

    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    
    logger.info("Computed mean distributions and applied smoothing for KL divergence.")
    
    magnitude = float(np.sum(kl_div(p, q)))

    with open(filename, "w") as f:
        json.dump({"magnitude": magnitude}, f)

    logger.info(f"KL drift magnitude ({magnitude}) saved to {filename}")
    return magnitude


## ------ Log likelihood drift ------ ##
def _mean_distribution(sample):
    logger.debug("Calculating mean distribution from the sample...")
    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim == 1:
        return sample
    return sample.mean(axis=0)

def _safe_model_distribution(prob_vector, alpha=1e-12):
    logger.debug("Ensuring model distribution is valid and non-zero...")
    prob_vector = np.asarray(prob_vector, dtype=np.float64)
    prob_vector = prob_vector + alpha
    logger.debug(f"Added alpha={alpha} to model distribution to avoid zeros.")
    return prob_vector / prob_vector.sum()

def _average_log_likelihood(sample, model_distribution):
    logger.debug("Calculating average log likelihood of the sample under the model distribution...")
    sample = np.asarray(sample, dtype=np.float64)
    log_model = np.log(model_distribution)

    if sample.ndim == 1:
        return float(np.sum(sample * log_model))
    
    logger.debug("Sample is 2D, calculating average log likelihood across all samples...")
    return float(np.mean(np.sum(sample * log_model, axis=1)))

def _log_likelihood_drift_magnitude(reference_sample, test_sample, alpha=1e-12):
    logger.debug("Calculating log likelihood drift magnitude...")
    reference_distribution = _mean_distribution(reference_sample)
    reference_distribution = _safe_model_distribution(reference_distribution, alpha=alpha)

    reference_ll = _average_log_likelihood(reference_sample, reference_distribution)
    test_ll = _average_log_likelihood(test_sample, reference_distribution)
    
    logger.debug(f"Calculated log likelihoods: reference_ll={reference_ll}, test_ll={test_ll}")
    return float(reference_ll - test_ll)

def run_log_likelihood_permutation(
    permutation,
    aggregated_samples,
    reference_sample_size,
    test_sample_size,
    alpha=1e-12
):
    os.system('taskset -p 0xffffffff %d' % os.getpid())
    logger.debug(f"Permutation {permutation}: Running permutation for log likelihood drift in PID {os.getpid()}")
    rng = np.random.default_rng(seed=permutation)
    shuffled = rng.permutation(aggregated_samples)

    permutation_reference_sample = shuffled[:reference_sample_size]
    permutation_test_sample = shuffled[
        reference_sample_size: reference_sample_size + test_sample_size
    ]
    logger.debug(f"Permutation {permutation}: Created reference and test samples for log likelihood drift.")
    
    results= _log_likelihood_drift_magnitude(
        permutation_reference_sample,
        permutation_test_sample,
        alpha=alpha
    )

    logger.debug(f"Permutation {permutation}: Calculated log likelihood. Drift magnitude: {results}")
    
    return results 

def log_likelihood_drift(
    reference_sample,
    test_sample,
    filename,
    K=1000,
    n_jobs=10,
    alpha=1e-12
):
    logger.info("Running log likelihood drift detection.")
    with open(filename, "w") as f:
        json.dump({}, f)
        logger.debug(f"Initialized empty JSON file for drift results: {filename}")
    reference_sample = np.asarray(reference_sample, dtype=np.float64)
    test_sample = np.asarray(test_sample, dtype=np.float64)

    if reference_sample.ndim != 2 or test_sample.ndim != 2:
        raise ValueError("reference_sample and test_sample must be 2D arrays.")

    if reference_sample.shape[1] != test_sample.shape[1]:
        raise ValueError("reference_sample and test_sample must have the same number of columns.")

    logger.info("Validated shapes of reference and test samples.")

    reference_sample_size = len(reference_sample)
    test_sample_size = len(test_sample)
    aggregated_samples = np.concatenate([reference_sample, test_sample], axis=0)
    
    drift_magnitude = _log_likelihood_drift_magnitude(
        reference_sample,
        test_sample,
        alpha=alpha
    )
    logger.info(f"Calculated log likelihood drift magnitude: {drift_magnitude}")

    permutation_magnitudes = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_log_likelihood_permutation)(
            permutation,
            aggregated_samples,
            reference_sample_size,
            test_sample_size,
            alpha
        )
        for permutation in range(K)
    )
    logger.info(f"Completed {K} permutations for log likelihood drift detection.")

    p_value = (1 + sum(m >= drift_magnitude for m in permutation_magnitudes)) / (K + 1)

    with open(filename, "w") as f:
        json.dump({"magnitude": drift_magnitude, "p_value": p_value}, f)
    logger.info(f"Log likelihood drift results saved to {filename}")
    
    return drift_magnitude, p_value
