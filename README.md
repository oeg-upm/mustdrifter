# MuSTDrifter

**MuSTDrifter** (**Mu**lti-**S**ource **T**emporal **Drifter**) is a Python framework for **unsupervised multidimensional discourse drift detection**, designed to quantify how communication changes over time through complementary forms of distribution shift analysis.

By combining **covariate shift** and **prior probability shift** detection, MuSTDrifter provides a modular methodology to represent discourse across multiple linguistic dimensions, estimate temporal drift between periods, and generate analytical reports.

---
## Conceptual Framework

MuSTDrifter models discourse evolution through two complementary forms of distribution shift:

### Covariate Shift

For semantic, lexical, and syntactic dimensions, discourse change is modeled as shifts in

\[
P_t(x)
\]

assuming relative stability in

\[
P_t(y \mid x)
\]

capturing changes in observed linguistic feature distributions over time.


### Prior Probability Shift

For the thematic dimension, discourse change is modeled as shifts in

\[
P_t(y)
\]

assuming relative stability in

\[
P_t(x \mid y)
\]

capturing variation in the prior distribution of latent themes across temporal periods.

---

## Framework Components

MuSTDrifter consists of four main components:

### 1. Discourse Representations and Drift Estimation

The framework models drift across five complementary dimensions:

**Semantic Drift** captures shifts in embedding distributions using:
- Maximum Mean Discrepancy (MMD)  
- Cosine Distance  
- Kolmogorov–Smirnov (KS)

**Lexical Drift** measures divergence in content-word lemma distributions using:
- Jensen–Shannon Divergence (JS)  
- Kullback–Leibler Divergence (KL)  
- Log-Likelihood divergence

**Syntactic Content Drift** captures structural changes through content-word POS rule distributions using:
- Jensen–Shannon Divergence (JS)  
- Kullback–Leibler Divergence (KL)  
- Log-Likelihood divergence

**Syntactic Style Drift** models stylistic shifts through conditional POS transition distributions using:
- Jensen–Shannon Divergence (JS)  
- Kullback–Leibler Divergence (KL)  
- Log-Likelihood divergence

**Thematic Drift** measures changes in topic distributions over time through prior probability shift using:
- Jensen–Shannon Divergence (JS)  
- Kullback–Leibler Divergence (KL)  
- Log-Likelihood divergence

### 2. Drift Computation and Inference

Including support for:

- Permutation-based significance estimation  
- Resumable drift computation  
- Parallelized large-scale processing

### 3. Analysis and Reporting

- Pairwise drift matrices  
- Heatmap reports  
- Aggregated multidimensional drift summaries

---

## Installation with Poetry
Required Python 3.12

```bash
git clone https://github.com/your-user/mustdrifter.git
cd mustdrifter
poetry install
```

---

## Quick Start

### Data conversion
MuSTDrifter expects a dataframe containing at least:

| Column | Description |
|--------|-------------|
| `content` | Document text |
| `period_id` | Temporal period identifier |
| `doc_id` *(recommended)* | Unique document identifier |

```python
#Convert data to desired fotmat
import pandas as pd

df = pd.DataFrame({
    "doc_id": [1,2,3],
    "content": [
        "Political text A",
        "Political text B",
        "Political text C"
    ],
    "period_id": [1,1,2]
})
```

### MuSTDrifter class instantiation with the desired params
| Parameter | Description |
|----------|-------------|
| `df` | Input dataframe |
| `df_name` | Dataset name used to organize outputs |
| `results_path` | Directory where generated files and reports are stored |
| `n_jobs` | Number of parallel workers |
| `K` | Number of permutations for significance estimation |
| `device` | Computation device, e.g. `"cuda"` or `"cpu"` |

```python
#Instantiate class
from mustdrifter import MuSTDrifter

drifter = MuSTDrifter(
    df=df,
    df_name="my_df",
    results_path="./my_path",
    n_jobs=20,
    K=100,
    device="cpu"
)
```

### Generate drift dimension representation for the desired dimensions

| Parameter | Description |
|----------|-------------|
| `dimensions` | List of discourse dimensions to generate. Defaults to all available dimensions (`semantic`, `syntactic_content`, `syntactic_style`, `lexical`, `thematic`). |
| `**kwargs` | Additional keyword arguments forwarded to the corresponding dimension generators. |

```python
# Option 1: selective dimension generation.
# Generate lexical and syntactic representations only
drifter.generate_drift_dimensions(
    dimensions=[
        "lexical",
        "syntactic_content",
        "syntactic_style"
    ]
)
```

```python
# Option 2: complete dimension generation.
# Generate representations for all dimensions
drifter.generate_drift_dimensions()
```
The generated dimension representations will appear in the following directory path
```text
my_path/
└── my_df/
    └── data/
        ├── semantic/
        ├── lexical/
        ├── syntactic/
        └── thematic/
```

### Calculate the drift for the desired dimensions
| Parameter | Description |
|----------|-------------|
| `drift_dimensions` | List of discourse dimensions for which drift will be computed. Defaults to all available dimensions (`semantic`,`syntactic_content`, `syntactic_style`, `lexical`, `thematic`) |
| `metrics` | Optional list of drift metrics to apply. By default, semantic diemsnion uses `mmd`, `ks`, and `cos`, while the rest use `js`, `kl`, and `log` (log-likelihood). |
| `rebase` | If `True`, recomputes and overwrites previously stored drift results. |

```python
# Option 1: selective data-drift.
# I.e. calculate Jensen Shannon divergence for lexical and syntactic dimensions
drifter.calculate_drift(
    drift_dimensions=[
        "lexical",
        "syntactic_content",
        "syntactic_style"
    ],
    metrics= ["js"]
)
```

```python
# Option 2: complete data-drift.
# Calculate all metrics (corresponding ones) for all dimensions
drifter.calculate_drift()
```

The computed drift will appear in the following directory path
```text
my_path/
└── my_df/
    └── drift/
        ├── semantic/
        ├── lexical/
        ├── syntactic/
        └── thematic/
```

### Report the results in heatmaps

| Parameter | Description |
|----------|-------------|
| `periods` | Optional list of period identifiers to include. If `None`, all periods in the dataframe are used. |
| `export` | If `True`, exports the generated heatmaps as SVG files. |
| `aggregate_by` | Aggregation mode. Use `"dimension"` for a single aggregated meta-magnitude heatmap, `"metric"` for one heatmap per dimension, or `None` for one heatmap per dimension and metric. |
| `period_labels` | Optional dictionary mapping period identifiers to readable labels. |
| `title` | Optional custom title for the generated heatmap. |


```python
# Option 1: aggregate all dimensions and metrics into a single heatmap
drifter.report_heatmaps(
    export=True,
    aggregate_by="dimension",
    period_labels={1:"2020", 2:"Today"},
)
```

```python
# Option 2: aggregate metrics within each dimension
# Generates one heatmap per discourse dimension
drifter.report_heatmaps(
    export=True,
    aggregate_by="metric",
    period_labels={1:"2020", 2:"Today"},
)
```

```python
# Option 3: no aggregation
# Generates one heatmap per dimension and metric
drifter.report_heatmaps(
    export=True,
    period_labels={1:"2020", 2:"Today"},
)
```

Generated reports will appear in the following directory path:

```text
my_path/
└── my_df/
    └── report/
```
---

## Use Cases

MuSTDrifter has been used for:

- Campaign vs routine political discourse analysis  

---

## Citation

TBD

---

## License

CC-BY

---

## Author

Ibai Guillén Pacho