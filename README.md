# MuSTDrifter

<p align="center">
  <img src="docs/assets/logo.svg" width="180">
</p>

<p align="center">
  <b>Mu</b>lti-<b>S</b>ource <b>T</b>emporal <b>Drifter</b>
</p>

<p align="center">
  Unsupervised multidimensional discourse drift detection in Python.
</p>

---

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

MuSTDrifter is a Python framework for quantifying how discourse evolves over time through multidimensional data drift analysis.

The framework models discourse evolution across:

- Semantic dimensions
- Lexical dimensions
- Syntactic content
- Syntactic style
- Thematic distributions

and estimates temporal drift using complementary forms of:

- Covariate shift detection
- Prior probability shift detection

---

## Features

- Multidimensional discourse representations
- Multiple drift metrics
- Permutation-based significance estimation
- Parallelized drift computation
- Automatic heatmap reporting
- Modular architecture
- Reproducible analysis pipelines

---

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/oeg-upm/mustdrifter.git
cd mustdrifter
poetry install
```

---

## Quick Example

```python
import pandas as pd

from mustdrifter import MuSTDrifter

df = pd.DataFrame({
    "doc_id": [1, 2, 3],
    "content": [
        "Political text A",
        "Political text B",
        "Political text C",
    ],
    "period_id": [1, 1, 2]
})

drifter = MuSTDrifter(
    df=df,
    df_name="example",
    results_path="./results",
)

# Generate discourse representations
drifter.generate_drift_dimensions()

# Compute multidimensional drift
drifter.calculate_drift()

# Generate heatmaps
drifter.report_heatmaps(
    export=True,
    aggregate_by="dimension"
)
```

---

## Documentation

Full documentation is available [here](https://oeg-upm.github.io/mustdrifter/).

Including:

- Installation
- Quickstart
- API Reference
- Drift metrics
- Reporting utilities

---

## Drift Dimensions

| Dimension | Description |
|---|---|
| Semantic | Embedding distribution drift |
| Lexical | Content-word lemma drift |
| Syntactic Content | POS-rule structural drift |
| Syntactic Style | Conditional POS transition drift |
| Thematic | Topic distribution drift |

---

## Drift Metrics

### Semantic Drift
- Maximum Mean Discrepancy (MMD)
- Cosine Drift
- Kolmogorov–Smirnov (KS)

### Lexical, Syntactic, and Thematic Drift
- Jensen–Shannon Divergence (JS)
- Kullback–Leibler Divergence (KL)
- Log-Likelihood divergence

---

## Citation

```bibtex
TBD
```

---

## License

Creative Commons Attribution 4.0 International License
[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

---

## Author

Ibai Guillén Pacho
