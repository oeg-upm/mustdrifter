# Installation

## Requirements

- Python 3.12+

## Install with Poetry

```bash
pip install mustdrifter
```

## Verify installation

```python
from mustdrifter import MuSTDrifter
```

## Optional GPU support

MuSTDrifter supports GPU acceleration through PyTorch and compatible embedding models.

Example:

```python
drifter = MuSTDrifter(
    df=df,
    df_name="my_dataset",
    results_path="./results",
    device="cuda"
)
```
