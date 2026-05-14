# Deeply Mapping Cell and Spot in Joint Latent Space

This repository provides the implementation for **deeply mapping single‑cell RNA‑seq and spatial transcriptomics data into a joint latent space**. The model learns a shared representation that aligns cellular and spatial measurements, enabling cross‑modality integration, imputation, and downstream analysis.

## 📌 Overview

  ![Method: Figure1](./Figure1-final-fit-nolegend-2.png)

Single‑cell RNA‑seq (scRNA‑seq) captures gene expression at high resolution but loses spatial context, while spatial transcriptomics (ST) retains tissue architecture but at lower cellular resolution. This project bridges the two modalities by learning a **joint embedding** where cells from scRNA‑seq and spots from ST are mapped to a common latent space. The learned representations can be used for:

- Aligning cell types with spatial locations  
- Predicting unmeasured gene expression in spatial data  
- Integrating multiple datasets across technologies




## 🚀 Getting Started

### Prerequisites
conda env create -f environment.yml

- Python 3.8 or later  
- PyTorch (version ≥1.10)  
- CUDA‑capable GPU (recommended)  

### Installation

Clone the repository and install the required packages:
conda env create -f environment.yml

import os
import scanpy as sc
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import softmax, cosine_similarity
import logging
import numpy as np

### Input Data Preparation：
Download here: https://drive.google.com/drive/folders/1Vf8iVi29hQqXOYWpDYSgmuAbWvS5l6XL?usp=sharing

### Example:
Run this file step by step: python JointEmbedding4.py
