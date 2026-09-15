# mammo_prep

Utility package for preprocessing mammography images.

## Installation

From the project root folder (where `pyproject.toml` is located):

```bash
pip install -e .
```

The `-e` flag (editable mode) makes any changes you make to the code
apply automatically without needing to reinstall.

## Usage in notebooks

```python
# Import full modules
from mammo_prep import io, normalize, artifacts, viz

# Or import specific functions
from mammo_prep.io import load_dicom
from mammo_prep.normalize import clahe_normalize
from mammo_prep.artifacts import crop_to_breast
from mammo_prep.viz import plot_comparison

# Basic example
img = load_dicom("path/to/image.dcm")
img_norm = clahe_normalize(img)
img_crop = crop_to_breast(img_norm)
plot_comparison(img, img_crop, titles=["Original", "Processed"])
```

## Modules

| Module | Contents |
|---|---|
| `io.py` | Loading and saving images (DICOM, PNG, JPG) |
| `normalize.py` | Pixel normalization and standardization |
| `artifacts.py` | Background removal, artifact removal, and orientation |
| `viz.py` | Visualization and histograms |
