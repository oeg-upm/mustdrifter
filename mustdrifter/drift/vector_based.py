from scipy.spatial.distance import pdist
from functools import partial
from frouros.detectors.data_drift import MMD
from frouros.utils.kernels import rbf_kernel
import json
from joblib import Parallel, delayed
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

## ------ MMD drift ------ ##

def run_mmd_permutation(
    permutation,
    aggregated_samples,
    reference_sample_size,
    test_sample_size,
    custom_kernel,
    drift_magnitude
):
    rng = np.random.default_rng(seed=permutation)
    shuffled = rng.permutation(aggregated_samples)

    permutation_reference_sample = shuffled[:reference_sample_size]
    permutation_test_sample = shuffled[
        reference_sample_size: reference_sample_size + test_sample_size
    ]

    permutation_detector = MMD(kernel=custom_kernel)
    permutation_detector.fit(X=permutation_reference_sample)

    result, _ = permutation_detector.compare(X=permutation_test_sample)
    permutation_drift_magnitude = abs(result.distance)

    return int(permutation_drift_magnitude >= drift_magnitude)

def mmd_drift(reference_sample, test_sample, filename, K=100, n_jobs=10):
    reference_sample_size= len(reference_sample)
    test_sample_size= len(test_sample)

    aggregated_samples= np.concatenate([reference_sample, test_sample])

    bak_filename= filename.replace(".json", "_bak.json")
    if os.path.exists(bak_filename):
        with open(bak_filename, "r") as f:
            permutation_bak= json.load(f)
        
        sigma_median= permutation_bak["sigma_median"]
    
        # Use the computed sigma for the RBF kernel
        custom_kernel = partial(rbf_kernel, sigma=sigma_median)
        drift_magnitude= permutation_bak["magnitude"]

        ### For measuring the drift significance
        permutation_test= permutation_bak["permutation_test"]
        permutation_range= range(permutation_bak["permutation"]+1, K)

    else:   
        # Compute pairwise distances and get the median
        pairwise_dists = pdist(aggregated_samples, metric="euclidean")
        sigma_median = np.median(pairwise_dists)

        # Use the computed sigma for the RBF kernel
        custom_kernel = partial(rbf_kernel, sigma=sigma_median)

        ### Measure the drift magnitude
        # Initialize the MMD detector with the automatically selected sigma
        detector = MMD(kernel=custom_kernel)

        # Fit on reference embeddings
        detector.fit(X=reference_sample)

        result, _ = detector.compare(X=test_sample)
        drift_magnitude = abs(result.distance) # Critical value during permutation test

        ### Measure the drift significance
        permutation_test= []
        permutation_range= range(K)


    
    results = Parallel(n_jobs=n_jobs)(
    delayed(run_mmd_permutation)(
            permutation,
            aggregated_samples,
            reference_sample_size,
            test_sample_size,
            custom_kernel,
            drift_magnitude
        )
        for permutation in permutation_range
    )

    permutation_test.extend(results)
    
    p_value = (1 + sum(permutation_test)) / (K + 1)

    with open(filename, "w") as f:
        json.dump({"magnitude": drift_magnitude, "p_value": p_value}, f)

    if os.path.exists(bak_filename):
        os.remove(bak_filename)

    return drift_magnitude,p_value


## ------ Cos drift ------ ##

def median_embedding(embeddings):
    return np.median(np.array(embeddings.tolist()), axis=0)

def closest_to_median(embeddings):
    median= median_embedding(embeddings)
    closest_point = min(embeddings, key=lambda p: np.linalg.norm(p - median))  # Find closest point
    return closest_point

def run_cos_permutation(
    permutation,
    aggregated_samples,
    reference_sample_size,
    test_sample_size,
    drift_magnitude
):
    rng = np.random.default_rng(seed=permutation)
    shuffled = rng.permutation(aggregated_samples)

    permutation_reference_sample = shuffled[:reference_sample_size]
    permutation_test_sample = shuffled[
        reference_sample_size: reference_sample_size + test_sample_size
    ]

    permutation_reference_vector = closest_to_median(permutation_reference_sample)
    permutation_test_vector = closest_to_median(permutation_test_sample)

    permutation_drift_magnitude = abs(
        pdist([permutation_reference_vector, permutation_test_vector], metric="cosine")[0]
    )

    return int(permutation_drift_magnitude >= drift_magnitude)


def cos_drift(reference_sample, test_sample, filename, K=100, n_jobs=10):
    reference_sample = np.asarray(reference_sample, dtype=np.float64)
    test_sample = np.asarray(test_sample, dtype=np.float64)

    reference_sample_size = len(reference_sample)
    test_sample_size = len(test_sample)

    aggregated_samples = np.concatenate([reference_sample, test_sample], axis=0)

    bak_filename = filename.replace(".json", "_bak.json")

    if os.path.exists(bak_filename):
        with open(bak_filename, "r") as f:
            permutation_bak = json.load(f)

        drift_magnitude = permutation_bak["magnitude"]
        permutation_test = permutation_bak["permutation_test"]
        permutation_range = range(permutation_bak["permutation"] + 1, K)

    else:
        reference_sample_vector = closest_to_median(reference_sample)
        test_sample_vector = closest_to_median(test_sample)

        drift_magnitude = abs(
            pdist([reference_sample_vector, test_sample_vector], metric="cosine")[0]
        )

        permutation_test = []
        permutation_range = range(K)

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_cos_permutation)(
            permutation,
            aggregated_samples,
            reference_sample_size,
            test_sample_size,
            drift_magnitude
        )
        for permutation in permutation_range
    )

    permutation_test.extend(results)

    p_value = (1 + sum(permutation_test)) / (K + 1)

    with open(filename, "w") as f:
        json.dump({"magnitude": drift_magnitude, "p_value": p_value}, f)

    if os.path.exists(bak_filename):
        os.remove(bak_filename)

    return drift_magnitude, p_value


