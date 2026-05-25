from scipy.spatial.distance import pdist
from functools import partial
from frouros.detectors.data_drift import MMD
from frouros.utils.kernels import rbf_kernel
import json
from joblib import Parallel, delayed
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import logging
logger = logging.getLogger(__name__)

import gc

# Needed for parallel processing to ensure that all CPU cores are utilized effectively
# os.system("taskset -p 0xff %d" % os.getpid())

## ------ MMD drift ------ ##
def build_mmd_bak_data(permutation_test, drift_magnitude, sigma_median):
    return {
        "magnitude": drift_magnitude,
        "permutation_test": permutation_test,
        "sigma_median": sigma_median
    }

def estimate_sigma_median(sample, n_pairs=100000, seed=42):
    sample = np.asarray(sample, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n = len(sample)

    idx1 = rng.integers(0, n, size=n_pairs)
    idx2 = rng.integers(0, n, size=n_pairs)

    valid = idx1 != idx2
    idx1 = idx1[valid]
    idx2 = idx2[valid]

    dists = np.linalg.norm(sample[idx1] - sample[idx2], axis=1)
    return float(np.median(dists))

def run_mmd_permutation(
    permutation,
    aggregated_samples,
    reference_sample_size,
    test_sample_size,
    custom_kernel,
    drift_magnitude
):
    # os.system('taskset -p 0xffffffff %d' % os.getpid())
    
    logger.debug(f"Permutation {permutation}: Running MMD permutation in PID {os.getpid()}.")
    rng = np.random.default_rng(seed=permutation)
    shuffled = rng.permutation(aggregated_samples)

    logger.debug(f"Permutation {permutation}: Shuffled aggregated samples for MMD drift permutation test.")
    permutation_reference_sample = shuffled[:reference_sample_size]
    permutation_test_sample = shuffled[
        reference_sample_size: reference_sample_size + test_sample_size
    ]
    logger.debug(f"Permutation {permutation}: Created reference and test samples for MMD drift.")

    logger.debug(f"Permutation {permutation}: Initializing MMD detector with custom RBF kernel for permutation test.")
    permutation_detector = MMD(kernel=custom_kernel)
    logger.debug(f"Permutation {permutation}: Fitting MMD detector on reference sample for permutation test.")
    permutation_detector.fit(X=permutation_reference_sample)
    logger.debug(f"Permutation {permutation}: Fitted MMD detector on reference sample.")
    
    logger.debug(f"Permutation {permutation}: Comparing test sample against reference sample using MMD detector for permutation test.")
    result, _ = permutation_detector.compare(X=permutation_test_sample)
    permutation_drift_magnitude = abs(result.distance)
    logger.debug(f"Permutation {permutation}: Calculated MMD drift magnitude: {permutation_drift_magnitude}")
    
    del shuffled, permutation_reference_sample, permutation_test_sample, permutation_detector, result
    gc.collect()
    return int(permutation_drift_magnitude >= drift_magnitude)

def mmd_drift(reference_sample, test_sample, filename, K=100, n_jobs=10):
    """
    Perform Maximum Mean Discrepancy (MMD) drift detection between a reference sample and a test sample.
    This function calculates the drift magnitude and significance (p-value) using the MMD method with an RBF kernel.
    It supports saving intermediate results to a file for backup and resuming computations if interrupted.
    
    Parameters
    ----------
        reference_sample (np.ndarray): The reference sample data.
        test_sample (np.ndarray): The test sample data to compare against the reference.
        filename (str): Path to the JSON file where results will be saved. If empty, no file will be created.
        K (int, optional): Number of permutations for significance testing. Defaults to 100.
        n_jobs (int, optional): Number of parallel jobs for permutation testing. Defaults to 10.
    
    Raises
    ------
    RuntimeError
        If some permutations are not completed.
    
    Returns
    -------
    dict
        A dictionary containing the drift magnitude and p-value:
            - "magnitude" (float): The calculated drift magnitude.
            - "p_value" (float): The p-value indicating the significance of the drift.
    """
    if filename=="": filename=None

    logger.debug("Running MMD drift detection.")
    
    with open(filename, "w") as f:
        json.dump({}, f)
        logger.debug(f"Initialized empty JSON file for drift results: {filename}")

    reference_sample_size= len(reference_sample)
    test_sample_size= len(test_sample)
    
    aggregated_samples= np.concatenate([reference_sample, test_sample])
    logger.debug(f"Aggregated reference and test samples. Total size: {len(aggregated_samples)}")
    
    bak_filename= filename.replace(".json", "_bak.json")
    logger.debug(f"Checking for backup file: {bak_filename}")

    if os.path.exists(bak_filename):
        with open(bak_filename, "r") as f:
            permutation_bak= json.load(f)
        logger.debug(f"Loaded backup file: {bak_filename}")
        
        sigma_median= permutation_bak["sigma_median"]
    
        # Use the computed sigma for the RBF kernel
        custom_kernel = partial(rbf_kernel, sigma=sigma_median)
        drift_magnitude= permutation_bak["magnitude"]

        ### For measuring the drift significance
        permutation_test= permutation_bak["permutation_test"]
        if len(permutation_test) != K:
            logger.warning(f"Backup file has {len(permutation_test)} permutations, expected {K}. Will recompute all permutations.")
            permutation_test = [None] * K
        # permutation_range= range(permutation_bak["permutation"]+1, K)

    else:
        logger.debug("No backup file found.")   
        # Compute pairwise distances and get the median
        logger.debug("Computing pairwise distances to determine median for RBF kernel sigma.")
        if len(aggregated_samples) <= 10000:
            # pdist exacto
            pairwise_dists = pdist(aggregated_samples, metric="euclidean")
            sigma_median = np.median(pairwise_dists)
            logger.debug(f"Computed median pairwise distance for sigma: {sigma_median}")
        else:
            # estimación por muestreo
            sigma_median = estimate_sigma_median(aggregated_samples, n_pairs=100000)
            logger.debug(f"Estimated median pairwise distance for sigma using sampling: {sigma_median}")

        # Use the computed sigma for the RBF kernel
        custom_kernel = partial(rbf_kernel, sigma=sigma_median)

        ### Measure the drift magnitude
        # Initialize the MMD detector with the automatically selected sigma
        detector = MMD(kernel=custom_kernel)
        logger.debug("Initialized MMD detector with RBF kernel using median pairwise distance as sigma.")

        # Fit on reference embeddings
        logger.debug("Fitting MMD detector on reference sample.")
        detector.fit(X=reference_sample)
        logger.debug("Fitted MMD detector on reference sample.")

        logger.debug("Comparing test sample against reference sample using MMD detector.")
        result, _ = detector.compare(X=test_sample)
        drift_magnitude = abs(result.distance) # Critical value during permutation test
        logger.debug(f"Calculated MMD drift magnitude: {drift_magnitude}")
        
        ### Measure the drift significance
        # permutation_test= []
        # permutation_range= range(K)
        permutation_test = [None] * K    

        with open(bak_filename, "w") as f:
            json.dump(build_mmd_bak_data(permutation_test=permutation_test, drift_magnitude=drift_magnitude, sigma_median=sigma_median), f)

    pending_permutations = [i for i, v in enumerate(permutation_test) if v is None]

    logger.debug(f"Running {K} permutations for MMD drift significance testing with {n_jobs} parallel jobs...")

    permutation_test = run_parallel_permutations(
        worker_fn=run_mmd_permutation,
        pending_items=pending_permutations,
        results_buffer=permutation_test,
        bak_filename=bak_filename,
        build_bak_data_fn=build_mmd_bak_data,
        worker_kwargs={
            "aggregated_samples": aggregated_samples,
            "reference_sample_size": reference_sample_size,
            "test_sample_size": test_sample_size,
            "custom_kernel": custom_kernel,
            "drift_magnitude": drift_magnitude
        },
        build_bak_data_kwargs={
            "drift_magnitude": drift_magnitude,
            "sigma_median": sigma_median
        },
        n_jobs=n_jobs,
        save_every=n_jobs
    )
    
    missing = [i for i, v in enumerate(permutation_test) if v is None]
    if missing:
        raise RuntimeError(f"{len(missing)} permutations were not completed.")
    
    p_value = (1 + sum(permutation_test)) / (K + 1)
    logger.debug(f"MMD drift detection completed. Drift magnitude: {drift_magnitude}, p-value: {p_value}")

    result= {"magnitude": drift_magnitude, "p_value": p_value}
    with open(filename, "w") as f:
        json.dump(result, f)
    logger.debug(f"MMD drift results saved to {filename}")

    if os.path.exists(bak_filename):
        os.remove(bak_filename)
    logger.debug(f"Removed backup file: {bak_filename}")

    return result


## ------ Cos drift ------ ##
def build_cos_bak_data(permutation_test, drift_magnitude):
    return {
        "magnitude": drift_magnitude,
        "permutation_test": permutation_test
    }

def median_embedding(embeddings):
    logger.debug("Calculating median embedding...")
    return np.median(np.array(embeddings.tolist()), axis=0)

def closest_to_median(embeddings):
    logger.debug("Finding closest embedding to median...")
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
    
    logger.debug(f"Permutation {permutation}: Running cosine permutation in PID {os.getpid()}.")
    rng = np.random.default_rng(seed=permutation)
    shuffled = rng.permutation(aggregated_samples)

    logger.debug(f"Permutation {permutation}: Shuffled aggregated samples for cosine drift permutation test.")
    permutation_reference_sample = shuffled[:reference_sample_size]
    permutation_test_sample = shuffled[
        reference_sample_size: reference_sample_size + test_sample_size
    ]
    logger.debug(f"Permutation {permutation}: Created reference and test samples for cosine drift.")

    permutation_reference_vector = closest_to_median(permutation_reference_sample)
    permutation_test_vector = closest_to_median(permutation_test_sample)

    permutation_drift_magnitude = abs(
        pdist([permutation_reference_vector, permutation_test_vector], metric="cosine")[0]
    )
    logger.debug(f"Permutation {permutation}: Calculated cosine drift magnitude: {permutation_drift_magnitude}")
    
    return int(permutation_drift_magnitude >= drift_magnitude)


def cos_drift(reference_sample, test_sample, filename, K=100, n_jobs=10):
    """
    Detects drift between a reference sample and a test sample using cosine similarity.
    This function calculates the cosine drift magnitude between the reference and test samples
    and performs a permutation test to assess the statistical significance of the drift. The
    results, including the drift magnitude and p-value, are saved to a JSON file.
    
    Parameters
    ----------
        reference_sample (np.ndarray): The reference sample data.
        test_sample (np.ndarray): The test sample data.
        filename (str): Path to the JSON file where results will be saved. If empty, no file is saved.
        K (int, optional): Number of permutations for significance testing. Defaults to 100.
        n_jobs (int, optional): Number of parallel jobs for permutation testing. Defaults to 10.
    
    Raises
    ------
    ValueError
        If the backup file exists but the number of permutations does not match K.
    RuntimeError
        If some permutations are not completed.
    
    Returns
    -------
    dict
        A dictionary containing the following keys:
            - "magnitude" (float): The calculated cosine drift magnitude.
            - "p_value" (float): The p-value from the permutation test.
    """
    
    if filename=="": filename=None

    logger.debug("Running cosine drift detection.")
    with open(filename, "w") as f:
        json.dump({}, f)
        logger.debug(f"Initialized empty JSON file for drift results: {filename}")
    reference_sample = np.asarray(reference_sample, dtype=np.float64)
    test_sample = np.asarray(test_sample, dtype=np.float64)

    reference_sample_size = len(reference_sample)
    test_sample_size = len(test_sample)

    aggregated_samples = np.concatenate([reference_sample, test_sample], axis=0)

    bak_filename = filename.replace(".json", "_bak.json")
    logger.debug(f"Checking for backup file: {bak_filename}")
    if os.path.exists(bak_filename):
        with open(bak_filename, "r") as f:
            permutation_bak = json.load(f)
        logger.debug(f"Loaded backup file: {bak_filename}")
        
        drift_magnitude = permutation_bak["magnitude"]
        permutation_test = permutation_bak["permutation_test"]

        if len(permutation_test) != K:
            raise ValueError(
                f"Backup size mismatch: expected K={K}, got {len(permutation_test)} results."
            )
        
        logger.debug(f"Using drift magnitude from backup: {drift_magnitude}")

    else:
        logger.debug("No backup file found.")
        reference_sample_vector = closest_to_median(reference_sample)
        test_sample_vector = closest_to_median(test_sample)

        drift_magnitude = abs(
            pdist([reference_sample_vector, test_sample_vector], metric="cosine")[0]
        )
        logger.debug(f"Calculated cosine drift magnitude: {drift_magnitude}")

        permutation_test = [None] * K

        with open(bak_filename, "w") as f:
            json.dump(build_cos_bak_data(permutation_test, drift_magnitude), f)

    pending_permutations = [i for i, v in enumerate(permutation_test) if v is None]
    logger.debug(f"Running {K} permutations for cosine drift significance testing with {n_jobs} parallel jobs...")

    permutation_test = run_parallel_permutations(
        worker_fn=run_cos_permutation,
        pending_items=pending_permutations,
        results_buffer=permutation_test,
        bak_filename=bak_filename,
        build_bak_data_fn=build_cos_bak_data,
        worker_kwargs={
            "aggregated_samples": aggregated_samples,
            "reference_sample_size": reference_sample_size,
            "test_sample_size": test_sample_size,
            "drift_magnitude": drift_magnitude
        },
        build_bak_data_kwargs={
            "drift_magnitude": drift_magnitude
        },
        n_jobs=n_jobs,
        save_every=n_jobs
    )
    
    missing = [i for i, v in enumerate(permutation_test) if v is None]
    if missing:
        raise RuntimeError(f"{len(missing)} permutations were not completed.")
    
    logger.debug(f"Completed permutations for cosine drift significance testing.")
    # permutation_test.extend(results)

    p_value = (1 + sum(permutation_test)) / (K + 1)

    result = {"magnitude": drift_magnitude, "p_value": p_value}
    with open(filename, "w") as f:
        json.dump(result, f)
    logger.debug(f"Cosine drift results saved to {filename}")
    
    if os.path.exists(bak_filename):
        os.remove(bak_filename)
    logger.debug(f"Removed backup file: {bak_filename}")
    
    return result



def run_parallel_permutations(
    worker_fn,
    pending_items,
    results_buffer,
    bak_filename,
    build_bak_data_fn,
    worker_kwargs=None,
    build_bak_data_kwargs=None,
    n_jobs=10,
    save_every=None,
):
    """
    Generic async parallel execution engine with immediate task replenishment.

    Parameters
    ----------------
    worker_fn : callable
        Function with signature worker_fn(item, *worker_args).
    pending_items : list[int]
        Items still pending execution (e.g. permutation ids).
    results_buffer : list
        Preallocated buffer where each result is stored by its item index.
        Example: [None] * K
    bak_filename : str
        Backup JSON filename.
    save_callback : callable
        Function with signature save_callback(results_buffer) -> dict
        Used to build backup payload.
    n_jobs : int
        Maximum number of parallel workers.
    save_every : int | None
        Save every N completed tasks. If None, defaults to n_jobs.
    *worker_args :
        Extra positional args passed to worker_fn.

    Returns
    -------
    list
        Updated results_buffer.
    """
    if worker_kwargs is None:
        worker_kwargs = {}

    if build_bak_data_kwargs is None:
        build_bak_data_kwargs = {}

    if save_every is None:
        save_every = n_jobs

    completed_since_save = 0
    pending_idx = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}

        while pending_idx < len(pending_items) and len(futures) < n_jobs:
            item = pending_items[pending_idx] # e.g. permutation id
            future = executor.submit(worker_fn, item, **worker_kwargs)
            futures[future] = item
            pending_idx += 1

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                item = futures.pop(future)
                result = future.result()

                results_buffer[item] = result
                completed_since_save += 1

                if pending_idx < len(pending_items):
                    new_item = pending_items[pending_idx]
                    new_future = executor.submit(worker_fn, new_item, **worker_kwargs)
                    futures[new_future] = new_item
                    pending_idx += 1

            if completed_since_save >= save_every or not futures:
                bak_data = build_bak_data_fn(permutation_test=results_buffer, **build_bak_data_kwargs)

                with open(bak_filename, "w") as f:
                    json.dump(bak_data, f)
                    
                done_count = sum(v is not None for v in results_buffer)
                total_count = len(results_buffer)
                logger.debug(f"Saved backup after {done_count}/{total_count} completed tasks.")

                completed_since_save = 0
                gc.collect()
    

    return results_buffer