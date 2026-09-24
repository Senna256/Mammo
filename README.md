# Mammo

Research code for **lesion detection in mammography** on the [VinDr-Mammo](https://physionet.org/content/vindr-mammo/) dataset: DICOM preprocessing, dataset handling and a Swin-Tiny + YOLOv8 detector.

This repository is part of the [IMPACT project](https://github.com/IMPACT-project-UdG) (PID2024-157201OB-C21).

> ⚠️ Research use only. Not a medical device and not intended for clinical decisions.

## Repository structure

```
Mammo/
├── mammo_prep/               # Mammography preprocessing pipeline
├── vindr_dataset/            # VinDr-Mammo dataset utilities (labels, splits, loaders)
├── Lesion Detection/
│   └── Yolo v8/              # Swin-Tiny backbone + YOLOv8 detection head
└── .gitignore
```

### `mammo_prep/`
Preprocessing of full-field digital mammograms:
- DICOM reading with windowing / normalization for 16-bit images
- Breast tissue segmentation and ROI cropping
- Contrast enhancement (CLAHE / linear)

### `vindr_dataset/`
Scripts and utilities to work with VinDr-Mammo: annotations, conversion to YOLO format and data splits.

### `Lesion Detection/Yolo v8/`
Lesion detector combining a **Swin-Tiny** backbone ([timm](https://github.com/huggingface/pytorch-image-models)) with the **YOLOv8** detection head and loss ([Ultralytics](https://github.com/ultralytics/ultralytics)).
- Input: 1024×1024 preprocessed mammograms
- Training: mixed precision (AMP), gradient accumulation

## Getting started

### 1. Requirements
- Python 3.10+
- PyTorch with CUDA
- `timm`, `ultralytics`, `pydicom`, `opencv-python`, `numpy`

```bash
git clone https://github.com/Senna256/Mammo.git
cd Mammo
pip install torch timm ultralytics pydicom opencv-python numpy
```

### 2. Data
Download VinDr-Mammo from PhysioNet (credentialed access required) and set the dataset path in the scripts:

```bash
export VINDR_ROOT=/path/to/vindr-mammo
```

### 3. Training

```bash
# 
cd "Lesion Detection/Yolo v8"
python train.py
```

## Dataset

Nguyen, H. T. et al. *VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography.* Scientific Data, 2023. The dataset is **not** included in this repository and is subject to its own license.

## Status

Work in progress.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

This repository is part of my PhD thesis, carried out at [ViCOROB](https://vicorob.udg.edu/) (University of Girona) under the supervision of Dr. Robert Martí, within the [IMPACT project](https://github.com/IMPACT-project-UdG).

This work was funded by Proyecto PID2024-157201OB-C21 (IMPACT), financed by MICIU/AEI/10.13039/501100011033 and by FEDER, UE.

## Authors

- **Enric** ([@Senna256](https://github.com/Senna256)) — PhD student, [ViCOROB](https://vicorob.udg.edu/), University of Girona · [ORCID](https://orcid.org/0009-0005-6355-2165)
- **Dr. Robert Martí** (supervisor) — [ViCOROB](https://vicorob.udg.edu/), University of Girona · [ORCID](https://orcid.org/0000-0002-8080-2710)