from frouros.detectors.data_drift import MMD, KSTest
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

## ------ KS drift ------ ##

def ks_drift(reference_sample, test_sample, filename):
    n_dimensions= reference_sample.shape[-1]

    detectors = [KSTest() for _ in range(n_dimensions)]
    # Fit on reference embeddings
    for dim_idx in range(n_dimensions): 
        detectors[dim_idx].fit(reference_sample[:,dim_idx])
        
    statistics= []
    p_values= []
    for dim_idx in range(n_dimensions): 
        result, _ = detectors[dim_idx].compare(X=test_sample[:,dim_idx])
        statistics.append(result.statistic)
        p_values.append(result.p_value)

    with open(filename, "w") as f:
        json.dump({"magnitude": np.median(statistics), "p_value": np.min(p_values)}, f)

    return np.median(statistics), np.min(p_values)
