# FraGNNet: A Deep Probabilistic Model for Mass Spectrum Prediction

## Overview

This repository contains the code and models for the paper **FraGNNet: A Deep Probabilistic Model for Mass Spectrum Prediction**. FraGNNet is a deep probabilistic graph neural network (GNN) model for predicting tandem mass spectra (MS/MS) from molecular structures. FraGNNet predicts MS/MS spectra from molecular structures using a combination of:
- A **recursive bond-breaking algorithm** for fragment generation.
- A **hierarchical graph neural network (GNN)** for probabilistic mass spectrum prediction.

This method provides:
- High mass accuracy.
- Scalability for large compound libraries.
- Interpretability through latent variable distributions (formula and fragment annotations).

---

## Installation

### Environment Setup

Use the provided script to set up the environment:
```bash
bash env/env_setup_gpu_current.sh
```

### Install FraGNNet Python Package
```
bash pip install -I .
```

## Project Structure

```bash
frag_gnn/
├── LICENSE
├── README.md
├── config # config files
│   └── template.yml # config file template
├── containers # apptainer file
├── data # data directory
│   ├── benchmark
│   ├── df
│   ├── frag
│   ├── ms2c
│   ├── proc
│   ├── raw
│   └── split
├── env # environment  
├── figs # figures
├── preproc_scripts # data preprocess scripts
│   ├── msgym_ms2c
│   └── pubchem_ms2c
├── scripts # training and evalution scripts
│   ├── misc
│   ├── ms2c
│   ├── plotting
│   └── testing
├── setup.cfg
├── setup.py
├── src # fragnnet source code
│   ├── frag_gnn
```

## Citation
````bibtex
@article{fragnnet,
  title={FraGNNet: A Deep Probabilistic Model for Mass Spectrum Prediction},
  author={Anonymous},
  year={2025},
}
````
