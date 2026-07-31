# IFACE

[![arXiv](https://img.shields.io/badge/arXiv-2603.09860-b31b1b.svg)](https://arxiv.org/abs/2603.09860)

**IFACE (Intrinsic Field–Aligned Coupled Embedding)** is a framework for computing **feature-aware soft correspondences and distances between protein surfaces**.

It combines **intrinsic geometry, mean-curvature information, and physicochemical surface fields** to quantify protein-surface similarity while also providing an explicit soft correspondence between surface vertices.

IFACE enables:

- protein surface comparison  
- vertex correspondence estimation between surfaces  
- computation of surface feature-fields, geometry, chemical and IFACE distances  

---

## IFACE Pipeline

```
Input protein surfaces with physicochemical fields
        ↓
Optimize a soft coupling matrix between surface vertices
        ↓
Construct bidirectional surface maps
        ↓
Compute structural, mean-curvature, and physicochemical field distances
        ↓
Aggregate the normalized components into geometry, chemical, and overall IFACE distances
```

---

## Abstract

Protein function is executed at the molecular surface, where shape and chemistry act together to govern interaction. Yet most comparison methods treat these aspects separately, privileging either global fold or local descriptors and missing their coupled organization.

Here we introduce IFACE (Intrinsic Field–Aligned Coupled Embedding), a correspondence-based framework that aligns protein surfaces through probabilistic coupling of intrinsic geometry with spatially distributed chemical fields. From this alignment, we derive a joint geometric–chemical distance that integrates structural and physicochemical discrepancies within a single formulation.

Across diverse proteins, this distance separates conformational variability from true structural divergence more effectively than fold-based similarity measures. Applied to the cytochrome P450 family, it reveals coherent family-level organization and identifies conserved buried catalytic pockets despite the complex topology.

By linking interpretable surface correspondences with a unified distance, IFACE establishes a principled basis for comparing protein interfaces and detecting functionally related interaction patches across proteins.

---

---

## Hardware Environment

All experiments were conducted on a workstation with the following configuration:

- **CPU:** Intel Core i9-14900K (24 cores, 32 threads)  
- **Memory:** 64 GB RAM  
- **Operating System:** Ubuntu 24.04.3 LTS  

---

## Table of Contents

- [Method Overview](#method-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Layout](#data-layout)
- [Input Requirements](#input-requirements)
- [Preprocessing](#preprocessing)
- [Running IFACE](#running-iface)
- [Feature Selection](#feature-selection)
- [Outputs](#outputs)
- [Coupling Matrices](#coupling-matrices)
- [Feature Distances](#feature-distances)
- [Analysis Notebook](#analysis-notebook)
- [Run Analysis](#run-analysis)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Method Overview

IFACE represents each protein surface as a **triangular mesh equipped with multiple physicochemical feature fields**.

The comparison pipeline:

1. Represent protein surfaces as meshes.
2. Attach **per-vertex geometry-physicochemical features**:
   - charge (electrostatic)
   - hphob (hydrophobicity)
   - hbond (hydrogen bonding propensity)
   - mean curvature
3. Compute a **coupling matrix (soft correspondence matrix)** between vertices of the two surfaces.
4. Compute geometric, feature-field-specific, and combined **distances** (IFACE, geometry and chemical) between the surfaces.

Outputs include:

- coupling matrix (vertex soft correspondences)  
- distances  

---

## Installation

```
conda create -n iface python=3.10.18 -y
conda activate iface
pip install -r requirements.txt
```

Verify installation:

```
python -c "import numpy, scipy, open3d, trimesh, halo, ot, pyvista, pymeshlab, cython, cycpd; print('IFACE ready')"
```

Expected output:

```
IFACE ready
```

---

## Quick Start

Run the IFACE pipeline:

Choose the mesh resolution by setting `TARGET_VERTEX_COUNT` in `config.py`.

Use `1000` or `2000` vertices for fast screening, and `3000` vertices for detailed analysis.

```bash
python data/preprocessing/preprocess.py
python run_iface.py --surf1 all --surf2 all
```

Then assemble the final dataset:

```bash
jupyter nbconvert --to notebook --execute analysis.ipynb --inplace
```

---

## Data Layout

Input data must be placed in:

```
data/raw/
```

Each protein must have its **own folder**, named by the protein ID.

Example:

```
data/raw/
└── 1JPZ/
    ├── 1JPZ.ply
    ├── 1JPZ_charge.npy
    ├── 1JPZ_hbond.npy
    ├── 1JPZ_hphob.npy
    └── 1JPZ_mean_curvature.npy
```

---

## Input Requirements

- Folder name and file basename must match exactly.
- Feature arrays must be **per-vertex**.
- Feature arrays must align with mesh vertex ordering.
- Each feature array must contain the same number of vertices as the mesh.

---

## Preprocessing

After placing protein folders in `data/raw/`, run:

```bash
python data/preprocessing/preprocess.py
```

This step preprocesses the input data, computes the required geodesic information, and prepares it for IFACE.

Processed data is stored in:

```
data/processed/
```

Precomputed geodesic information is stored in:

```
precomputed_geodesics/
```

---

## Running IFACE

`run_iface.py` compares protein surfaces specified by folder name under `data/processed/`.

### Single Pair

```bash
python run_iface.py --surf1 1JPZ --surf2 1TQN
```

### Batch Modes

#### All vs All

```bash
python run_iface.py --surf1 all --surf2 all
```

#### All vs One

```bash
python run_iface.py --surf1 all --surf2 1JPZ
```

#### One vs All

```bash
python run_iface.py --surf1 1JPZ --surf2 all
```

---

## Feature Selection

Default features used by IFACE:

- charge  
- hphob  
- hbond  
- mean_curvature  

Override the feature list:

```bash
python run_iface.py --surf1 1JPZ --surf2 1TQN \
    --features_list charge hbond hphob mean_curvature
```

---

## Outputs

Results are written under:

```
results/
```

```
results/
├── coupling_matrix/
├── distances/
│   ├── charge/
│   ├── hbond/
│   ├── hphob/
│   ├── mean_curvature/
│   └── structural/
└── csv/
```

---

## Coupling Matrices

```
results/coupling_matrix/
```

These matrices represent **soft correspondences between vertices of the two surfaces**.

---

## Feature Distances

```
results/distances/
```

Each directory contains pairwise **distances** computed for a specific surface feature.

- **structural/** — Intrinsic geometric (surface structure) distance.
- **mean_curvature/** — Mean curvature scalar field distance.
- **charge/** — Electrostatic field distance.
- **hphob/** — Hydrophobicity field distance.
- **hbond/** — Hydrogen-bonding propensity field distance.

---

## Analysis Notebook

The main analysis entry point is:

```
analysis.ipynb
```

This notebook:

- reads pairwise results from `results/distances/`
- assembles a structured dataframe
- exports the final dataset used for analysis

The resulting DataFrame contains the following columns:

- `ID1`, `ID2`: Identifiers of the two protein surfaces being compared. These correspond to folder names under `data/processed/`.
- Columns ending in `_norm`: Range-normalized distances. Columns without the `_norm` suffix contain the corresponding raw distances computed by IFACE.
- `chemical`: Aggregated chemical distance computed from the normalized electrostatic, hydrogen-bonding propensity, and hydrophobicity distances (`charge_norm`, `hbond_norm`, and `hphob_norm`).
- `geometry`: Aggregated geometric distance computed from the normalized intrinsic structural and mean-curvature distances (`structural_norm` and `mean_curvature_norm`).
- `iface`: Overall IFACE distance obtained by combining the `geometry` and `chemical` distances.

Columns:

- ID1  
- ID2  
- iface
- geomtery  
- chemical  
- structural_norm
- mean_curvature_norm  
- charge_norm  
- hbond_norm  
- hphob_norm  
- structural  
- mean_curvature
- charge  
- hbond  
- hphob  

Example output:

```
results/csv/distances.csv
```

---

## Run Analysis

Headless execution:

```bash
jupyter nbconvert --to notebook --execute analysis.ipynb --inplace
```

Interactive execution:

```bash
jupyter notebook analysis.ipynb
```

---

## Visualization

Visualization utilities are located in:

```
visualization/
```

Example notebook:

```
visualization/example.ipynb
```

These tools visualize **surface correspondences** using color mapping based on the coupling (soft correspondence) matrix.

---

## Project Structure

```
iface/
├── source/
│   ├── __init__.py
│   ├── config.py
│   ├── distance.py
│   ├── geometry.py
│   ├── model.py
│   ├── optim.py
│   └── utils.py
│
├── run_iface.py
│
├── analysis.ipynb
│
├── visualization/
│   ├── color_mapping.py
│   └── example.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── preprocessing/
│       └── preprocess.py
│
├── precomputed_geodesics/
│
├── results/
│   ├── coupling_matrix/
│   ├── distances/
│   └── csv/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Citation

If you use **IFACE** in your work, please cite the following paper:

```bibtex
@misc{swami2026jointgeometricchemicaldistanceprotein,
      title={Joint Geometric-Chemical Distance for Protein Surfaces}, 
      author={Himanshu Swami and John M. McBride and Jean-Pierre Eckmann and Tsvi Tlusty},
      year={2026},
      eprint={2603.09860},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2603.09860}, 
}
```

---

## License

This project is licensed under the **Apache License 2.0**.

See the `LICENSE` file for details.


