from frouros.detectors.data_drift import MMD, KSTest
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

## ------ KS drift ------ ##

def ks_drift(reference_sample, test_sample, filename):
    if filename=="": filename=None
    logger.debug("Running KS drift detection.")

    with open(filename, "w") as f:
        json.dump({}, f)
        logger.debug(f"Initialized empty JSON file for drift results: {filename}")
    n_dimensions= reference_sample.shape[-1]
    
    
    logger.debug("Initializing KS detectors on each dimension...")
    detectors = [KSTest() for _ in range(n_dimensions)]
    # Fit on reference embeddings
    for dim_idx in range(n_dimensions): 
        detectors[dim_idx].fit(reference_sample[:,dim_idx])
    logger.debug("KS detectors initialized and fitted on reference sample.")    
    
    logger.debug("Comparing test sample against reference sample using KS detectors...")
    statistics= []
    p_values= []
    for dim_idx in range(n_dimensions): 
        result, _ = detectors[dim_idx].compare(X=test_sample[:,dim_idx])
        statistics.append(result.statistic)
        p_values.append(result.p_value)
        
    logger.debug("KS drift detection completed.")

    result= {"magnitude": np.median(statistics), "p_value": np.median(p_values), 
                "p_value_median": np.median(p_values), "p_value_mean": np.mean(p_values),
                "p_value_min": np.min(p_values), "p_value_max": np.max(p_values)}
    with open(filename, "w") as f:
        json.dump(result, f)

    logger.debug(f"KS drift results saved to {filename}")

    return result
