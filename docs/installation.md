# Installation

## Requirements

- Python 3.12+

## Install with Poetry

```bash
git clone https://github.com/your-user/mustdrifter.git
cd mustdrifter
poetry install
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
