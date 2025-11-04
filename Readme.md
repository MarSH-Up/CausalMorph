# CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **CausalMorph**, a data preconditioning algorithm that projects observational datasets toward the Linear Non-Gaussian Acyclic Model (LiNGAM) compatible regime.

**Paper**: CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models
**Authors**: Mario De Los Santos-Hernández, Samuel Montero-Hernández, Felipe Orihuela-Espina, L. Enrique Sucar
**Journal**: Knowledge-Based Systems (Under Review)
**Manuscript ID**: KNOSYS-D-25-17892

## Overview

The Linear Non-Gaussian Acyclic Model (LiNGAM) family provides a unique advantage in causal discovery: unlike most methods, LiNGAM can identify a single, fully directed causal graph from purely observational data. However, LiNGAM's strict assumptions of **linearity** and **non-Gaussian noise** are often violated in practice, limiting its applicability.

**CausalMorph** addresses this challenge through a three-stage data preconditioning process:

1. **Stage I: Local Linearization** - MDL-guided polynomial approximation with Taylor expansion
2. **Stage II: Non-Gaussian Synthesis** - Whitening-recoloring with explicit non-Gaussian residuals
3. **Stage III: Orthogonalization** - Enforcement of statistical uncorrelatedness between noise and parents

### Key Results

Across 17,280 unique synthetic configurations (34,560 total runs), CausalMorph achieves:

- **37.7% relative reduction** in Structural Hamming Distance (SHD) for DirectLiNGAM (p < 0.001)
- **16.4% relative reduction** in SHD for ICALiNGAM (p < 0.001)
- **Regularization effect**: Improved performance even when LiNGAM conditions are met

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Basic Installation

```bash
git clone https://github.com/MarSH-Up/CausalMorph.git
cd CausalMorph
pip install -r requirements.txt
```

### Development Installation

```bash
git clone https://github.com/MarSH-Up/CausalMorph.git
cd CausalMorph
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from causalmorph import CausalMorph
from lingam import DirectLiNGAM

# Load your observational data
X = np.loadtxt('data/raw/your_data.csv', delimiter=',')

# Initialize CausalMorph with a tentative causal order
# (can be from domain knowledge, heuristics, or iterative refinement)
tentative_order = [0, 1, 2, 3, 4]  # Variable indices in causal order

# Apply CausalMorph preconditioning
morph = CausalMorph()
X_transformed = morph.fit_transform(X, causal_order=tentative_order)

# Run LiNGAM on the transformed data
model = DirectLiNGAM()
model.fit(X_transformed)

# Get the adjacency matrix
adjacency_matrix = model.adjacency_matrix_
```

## Repository Structure

```
CausalMorph/
├── causalmorph/              # Core implementation
│   ├── core/                 # Main algorithm components
│   │   ├── __init__.py
│   │   ├── linearization.py  # Stage I: MDL-guided linearization
│   │   ├── synthesis.py      # Stage II: Non-Gaussian synthesis
│   │   └── orthogonalization.py  # Stage III: Orthogonalization
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py        # SHD, F1, precision, recall
│   │   ├── visualization.py  # Plotting utilities
│   │   └── data_utils.py     # Data loading/preprocessing
│   └── tests/                # Unit tests
│
├── experiments/              # Experimental code
│   ├── synthetic_data/       # Data generation scripts
│   │   ├── generate_dags.py
│   │   ├── generate_sem.py
│   │   └── config.yaml
│   ├── benchmarks/           # Benchmark experiments
│   │   ├── run_baseline.py
│   │   ├── run_causalmorph.py
│   │   └── compare_methods.py
│   └── analysis/             # Result analysis
│       ├── statistical_tests.py
│       └── visualize_results.py
│
├── data/                     # Data directory
│   ├── raw/                  # Original datasets
│   ├── processed/            # Transformed datasets
│   └── results/              # Experimental results
│
├── docs/                     # Documentation
│   ├── paper/                # Paper-related materials
│   │   └── KNOSYS-D-25-17892.pdf
│   ├── tutorials/            # Usage tutorials
│   └── api/                  # API documentation
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_introduction.ipynb
│   ├── 02_synthetic_experiments.ipynb
│   ├── 03_real_world_examples.ipynb
│   └── 04_ablation_studies.ipynb
│
├── examples/                 # Example scripts
│   ├── basic_usage.py
│   ├── iterative_refinement.py
│   └── real_world_application.py
│
├── config/                   # Configuration files
│   ├── experiment_config.yaml
│   └── model_config.yaml
│
├── scripts/                  # Utility scripts
│   ├── setup_environment.sh
│   └── run_full_experiments.sh
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Methodology

### Stage I: MDL-Guided Local Linearization

For each variable Xi with tentative parent set pa(Xi):

1. Fit polynomial p(·) to standardized parent data
2. Select optimal degree d* using MDL criterion:
   ```
   MDL(d) = n log(MSE_d + ε_log) + λk_d
   ```
3. Compute local linear approximation via Taylor expansion:
   ```
   x_i,lin = p̂(x_0) + (X_pa - x_0)∇p̂(x_0)
   ```

### Stage II: Synthetic Non-Gaussian Residuals

1. Extract original residual: ε_orig = x_i - x_i,lin
2. Whiten residual and obtain coloring matrix C
3. Sample from non-Gaussian distributions (Laplace, Uniform, Exponential, Student's t)
4. Recolor: ε_synth = C·z_cand
5. Select distribution with minimum p-value on normality test

### Stage III: Orthogonalization and Variance Matching

1. Compute orthonormal basis Q for parent space
2. Orthogonalize: ε_ortho = ε_synth - QQ^T ε_synth
3. Scale to match original variance:
   ```
   x_i,new = x_i,lin + ε_ortho · (σ(ε_orig) / σ(ε_ortho))
   ```

## Citation

If you use CausalMorph in your research, please cite:

```bibtex
@article{delossantos2025causalmorph,
  title={CausalMorph: Preconditioning Data for Linear Non-Gaussian Acyclic Models},
  author={De Los Santos-Hern{\'a}ndez, Mario and Montero-Hern{\'a}ndez, Samuel and Orihuela-Espina, Felipe and Sucar, L. Enrique},
  journal={Knowledge-Based Systems},
  year={2025},
  note={Manuscript ID: KNOSYS-D-25-17892, Under Review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Mario De Los Santos-Hernández**: madlsh3517@gmail.com
- **Project Repository**: https://github.com/MarSH-Up/CausalMorph

## Acknowledgments

- Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE), Puebla, México
- School of Computer Science, University of Birmingham, United Kingdom 