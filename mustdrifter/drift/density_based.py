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
# os.system("taskset -p 0xff %d" % os.getpid())


def js_drift(reference_sample, test_sample, filename):
    """
    Compute Jensen-Shannon (JS) divergence between two distributions.

    This function estimates distributional drift between a reference sample
    and a test sample using Jensen-Shannon divergence. Results can
    optionally be exported to a JSON file.

    Parameters
    ----------
    reference_sample : np.ndarray
        Reference sample distribution.
    test_sample : np.ndarray
        Test sample distribution.
    filename : str
        Output path used to save the drift results. If empty, results are
        not exported.

    Returns
    -------
    dict
        Dictionary containing the drift results.

        - ``"magnitude"``: Jensen-Shannon divergence magnitude.

    Notes
    -----
    Jensen-Shannon divergence is a symmetric measure of the difference
    between two probability distributions.

    Input samples are converted to `float64` NumPy arrays before
    computation.

    If `filename` is provided, the results are exported in JSON format.
    """
    if filename=="": filename=None

    logger.debug("Running JS drift detection.")
    if filename is not None:
        with open(filename, "w") as f:
            json.dump({}, f)
            logger.debug(f"Initialized empty JSON file for drift results: {filename}")
    reference_sample = np.asarray(reference_sample, dtype=np.float64)
    test_sample = np.asarray(test_sample, dtype=np.float64)

    logger.debug("Computed mean distributions for reference and test samples.")

    magnitude = float(
        jensenshannon(reference_sample, test_sample, base=2.0)
    )
    result = {"magnitude": magnitude}

    if filename is not None:
        with open(filename, "w") as f:
            json.dump({"magnitude": magnitude}, f)

        logger.debug(f"JS drift magnitude ({magnitude}) saved to {filename}")
        
    return result


def kl_drift(reference_sample, test_sample, filename, eps=1e-12):
    r"""
    Compute Kullback-Leibler (KL) divergence between two distributions.

    This function estimates distributional drift between a reference sample
    and a test sample using Kullback-Leibler divergence. Input distributions
    are normalized and smoothed to avoid numerical instability. Results can
    optionally be exported to a JSON file.

    Parameters
    ----------
    reference_sample : np.ndarray
        Reference probability distribution.
    test_sample : np.ndarray
        Test probability distribution.
    filename : str
        Output path used to save the drift results. If empty, results are
        not exported.
    eps : float, optional
        Small smoothing constant used to avoid division by zero and
        logarithms of zero. Defaults to `1e-12`.

    Returns
    -------
    dict
        Dictionary containing the drift results.

        - ``"magnitude"``: Kullback-Leibler divergence magnitude.

    Raises
    ------
    ValueError
        If the input distributions do not have the same shape.

    Notes
    -----
    Kullback-Leibler divergence is computed as:

    .. math::

        \sum P(x) \log \frac{P(x)}{Q(x)}

    Input distributions are normalized to sum to 1 before computation.

    Smoothing is applied using `eps` to improve numerical stability.
    """
    if filename=="": filename=None

    logger.debug("Running KL drift detection.")
    if filename is not None:
        with open(filename, "w") as f:
            json.dump({}, f)
            logger.debug(f"Initialized empty JSON file for drift results: {filename}")

    p = np.asarray(reference_sample, dtype=np.float64)
    q = np.asarray(test_sample, dtype=np.float64)

    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    
    logger.debug("Computed mean distributions and applied smoothing for KL divergence.")
    
    magnitude = float(np.sum(kl_div(p, q)))
    result = {"magnitude": magnitude}
    if filename is not None:
        with open(filename, "w") as f:
            json.dump(result, f)
        logger.debug(f"KL drift magnitude ({magnitude}) saved to {filename}")
    return result


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
    # os.system('taskset -p 0xffffffff %d' % os.getpid())
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
    """
    Perform log-likelihood drift detection between a reference sample and a
    test sample.

    This function estimates distributional drift using a log-likelihood
    approach and evaluates statistical significance through permutation
    testing. Results can optionally be exported to a JSON file.

    Parameters
    ----------
    reference_sample : np.ndarray
        Reference sample distribution.
    test_sample : np.ndarray
        Test sample distribution.
    filename : str
        Output path used to save the drift results. If empty, results are
        not exported.
    K : int, optional
        Number of permutations used for significance estimation.
        Defaults to 1000.
    n_jobs : int, optional
        Number of parallel jobs used during permutation testing.
        Defaults to 10.
    alpha : float, optional
        Smoothing parameter used in the log-likelihood computation.
        Defaults to `1e-12`.

    Returns
    -------
    dict
        Dictionary containing the drift results.

        - ``"magnitude"``: log-likelihood drift magnitude.
        - ``"p_value"``: permutation-test p-value.

    Notes
    -----
    Permutation testing is used to estimate whether the observed drift
    magnitude significantly differs from the null distribution generated by
    random resampling.
    """
    if filename=="": filename=None

    logger.debug("Running log likelihood drift detection.")
    if filename is not None:
        with open(filename, "w") as f:
            json.dump({}, f)
            logger.debug(f"Initialized empty JSON file for drift results: {filename}")
    reference_sample = np.asarray(reference_sample, dtype=np.float64)
    test_sample = np.asarray(test_sample, dtype=np.float64)

    reference_sample_size = len(reference_sample)
    test_sample_size = len(test_sample)
    aggregated_samples = np.concatenate([reference_sample, test_sample], axis=0)
    
    drift_magnitude = _log_likelihood_drift_magnitude(
        reference_sample,
        test_sample,
        alpha=alpha
    )
    logger.debug(f"Calculated log likelihood drift magnitude: {drift_magnitude}")

    permutation_magnitudes = Parallel(n_jobs=n_jobs, backend="loky", verbose=n_jobs)(
        delayed(run_log_likelihood_permutation)(
            permutation,
            aggregated_samples,
            reference_sample_size,
            test_sample_size,
            alpha
        )
        for permutation in range(K)
    )
    logger.debug(f"Completed {K} permutations for log likelihood drift detection.")

    p_value = (1 + sum(m >= drift_magnitude for m in permutation_magnitudes)) / (K + 1)
    result= {"magnitude": drift_magnitude, "p_value": p_value}
    if filename is not None:
        with open(filename, "w") as f:
            json.dump(result, f)
        logger.debug(f"Log likelihood drift results saved to {filename}")
    
    return result
