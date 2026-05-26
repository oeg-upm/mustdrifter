# Quickstart

## Input Data

MuSTDrifter expects a dataframe containing at least:

| Column | Description |
|---|---|
| `content` | Document text |
| `period_id` | Temporal period identifier |
| `doc_id` | Unique document identifier (recommended) |

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "doc_id": [1, 2, 3],
    "content": [
        "Political text A",
        "Political text B",
        "Political text C",
    ],
    "period_id": [1, 1, 2]
})
```

Each row represents a document associated with a temporal period.

MuSTDrifter uses these periods to estimate how discourse distributions
change over time.

---

## Initialize MuSTDrifter

```python
from mustdrifter import MuSTDrifter

drifter = MuSTDrifter(
    df=df,
    df_name="example_dataset",
    results_path="./results",
    n_jobs=20,
    K=100,
    device="cpu"
)
```

### Parameters

| Parameter | Description |
|---|---|
| `df` | Input dataframe |
| `df_name` | Dataset name used to organize outputs |
| `results_path` | Directory where generated files and reports are stored |
| `n_jobs` | Number of parallel workers |
| `K` | Number of permutations used in significance estimation |
| `device` | Computation device (`"cpu"` or `"cuda"`) |

The framework automatically creates the directory structure required to
store generated representations, drift results, and reports.

---

## Generate discourse representations

Generate all dimensions:

```python
drifter.generate_drift_dimensions()
```

Or generate selected dimensions only:

```python
drifter.generate_drift_dimensions(
    dimensions=[
        "lexical",
        "syntactic_content",
        "syntactic_style"
    ]
)
```

### What this step does

This stage transforms discourse into structured representations that can
later be compared across periods.

MuSTDrifter supports five complementary dimensions:

| Dimension | Description |
|---|---|
| `semantic` | Embedding distributions representing semantic meaning |
| `lexical` | Lemma distributions of content words |
| `syntactic_content` | POS-rule distributions capturing structural content |
| `syntactic_style` | Conditional POS transition distributions capturing writing style |
| `thematic` | Topic distributions generated with BERTopic |

Generated representations are stored in:

```text
results/
└── example_dataset/
    └── data/
```

---

## Compute drift

Calculate drift for all dimensions and metrics:

```python
drifter.calculate_drift()
```

Or selectively:

```python
drifter.calculate_drift(
    drift_dimensions=[
        "lexical",
        "syntactic_content",
        "syntactic_style"
    ],
    metrics=["js"]
)
```

### What this step does

This stage compares discourse representations between temporal periods
and estimates the magnitude of distributional change.

By default:

- Semantic drift uses:
  - MMD
  - Cosine Drift
  - Kolmogorov–Smirnov

- Lexical, syntactic, and thematic drift use:
  - Jensen–Shannon divergence
  - Kullback–Leibler divergence
  - Log-Likelihood divergence

Computed drift results are stored in:

```text
results/
└── example_dataset/
    └── drift/
```

---

## Generate reports

Generate aggregated multidimensional heatmaps:

```python
drifter.report_heatmaps(
    export=True,
    aggregate_by="dimension",
    period_labels={
        1: "2020",
        2: "Today",
    },
)
```

Generate one heatmap per dimension:

```python
drifter.report_heatmaps(
    export=True,
    aggregate_by="metric",
)
```

Generate one heatmap per metric and dimension:

```python
drifter.report_heatmaps(
    export=True,
)
```

### What this step does

This stage converts drift estimations into interpretable visual reports.

Depending on the aggregation mode:

| Mode | Description |
|---|---|
| `"dimension"` | Single aggregated multidimensional heatmap |
| `"metric"` | One heatmap per discourse dimension |
| `None` | One heatmap per dimension and metric |

Reports are stored in:

```text
results/
└── example_dataset/
    └── report/
```
