# FraGNNet: A Deep Probabilistic Model for Mass Spectrum Prediction

## Overview

This repository contains the code and models for the paper **FraGNNet: A Deep Probabilistic Model
for Mass Spectrum Prediction**, and for MASSter, the fragmentation model developed after it. Both
predict tandem mass spectra (MS/MS) from molecular structures through explicit fragments, so every
predicted peak carries formula and fragment annotations.

- **FraGNNet** generates fragments with a **recursive bond-breaking algorithm** that builds a
  fragmentation DAG, then predicts the spectrum from those fragments with a **graph neural network
  (GNN)**. It offers high mass accuracy, scales to large compound libraries, and is interpretable
  through its latent distributions over formulas and fragments.
- **MASSter** enumerates no fragments ahead of time. One GNN pass over the precursor scores the
  chemical bonds; the most probable bond-cut sets are materialized, split into connected
  fragments and expanded into ion channels (charge retention or migration, hydrogen transfers).
  Candidate construction runs on CPU (Cython) or, optionally, CUDA.

Baselines retained for comparison: GRAFF and MassFormer (in-repository), NEIMS, and ICEBERG, which
now runs from the external [ms-pred](https://github.com/coleygroup/ms-pred) repository
([baselines reference](docs/reference/baselines.md)).

---

## Project Structure

```
src/fragnnet/
  dataset/ utils/                    shared datasets, samplers and utilities
  layers/                            shared torch building blocks (GNN cells, pooling, heads)
  loss/                              spectrum losses and distances
  pl_model/                          Lightning base classes (SpectrumPL, BinnedPL)
  inference/                         checkpoint loading and the prediction exporters
  runner.py registry.py hooks.py     training entry point, model registry, per-model hooks
  fragmentation/                     bond-cut chemistry shared by FraGNNet and MASSter
  models/fragnnet/                   FraGNNet: DAG builders (Cython), datasets, samplers, model
  models/masster/                    MASSter: candidate builders (Cython/CUDA), model
  models/neims/ models/precursor/ models/gnn/
  baselines/graff/ baselines/massformer/
```

Shared code never imports a specific model; it reaches each one through `model_type` in the
registry. See the [architecture map](docs/architecture.md) for the data flow and the
dependency rule.

**Project record and agent workflow**: start at [docs/project-state.md](docs/project-state.md);
see [docs/README.md](docs/README.md) for the documentation index.

**Experiment configs**: see [config/README.md](config/README.md). Every model reads the same
`config/template.yml`; `model_type` selects the model.

**Data preprocessing**: see [preproc_scripts/README.md](preproc_scripts/README.md)

**Training and Analysis scripts**: see [scripts/README.md](scripts/README.md)

**Analysis notebooks**: see [notebooks/README.md](notebooks/README.md)

If you are having trouble preprocessing the data or reproducing the results, please send an email to ayoung [AT] cs [DOT] toronto [DOT] edu.

## Installation

**For detailed installation instructions, see [INSTALL.md](INSTALL.md)**

### Quick Start (Recommended)

```bash
# Using conda (GPU, recommended)
bash env/env_setup_gpu_py311_cu121_250602.sh
conda activate FRAGNNET-GPU
```

For development with testing tools:
```bash
pip install -e ".[dev]"
```

### Alternative: Manual Pip Installation

```bash
# Create virtual environment and install PyTorch
python -m venv venv
source venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install fragnnet and dev dependencies
pip install -e ".[dev]"
```

The analysis scripts under `scripts/` need scikit-learn, which `.[dev]` includes; `.[scripts]`
installs it without the development tools. Python 3.10-3.12 is supported.

`env/fragnnet_gpu_py311.lock.txt` pins the exact package versions of the reference environment.
See [INSTALL.md](INSTALL.md) for detailed options, troubleshooting, and HPC cluster setup.

Note that the -e flag is used to install the package in editable mode. The Cython extensions
build at install time; after changing a `.pyx` file, rebuild them with
`python setup.py build_ext --inplace`.

## Training

```bash
python scripts/run_pl_model_fit.py -t config -c <experiment config> [-w online]
```

`-t` is the template directory (`config`), `-c` the experiment config (for example
`config/graff/nist20_merged_inchikey/s0.yml`), and `-w` the W&B mode (default `disabled`).
Inference is documented in the [inference guide](docs/guides/run_inference.md).

### Batching options

- `dynamic_batch_sampler` (FraGNNet): batches by fragment-DAG size instead of a fixed spectrum
  count, to avoid OOM. `dynamic_batch_sampler_mode` is `frag_node` or `frag_edge` and selects
  whether the budget counts `frag_pyg.num_nodes` or `frag_pyg.num_edges`;
  `dynamic_batch_sampler_max` is the per-batch budget. It requires
  `automatic_optimization: False`. The samplers are in
  [fragnnet_data_sampler.py](src/fragnnet/models/fragnnet/fragnnet_data_sampler.py), and the config
  checks in [fragnnet_hooks.py](src/fragnnet/models/fragnnet/fragnnet_hooks.py).
- MASSter bond-size curriculum: MASSter can train first on smaller precursor molecules, ranked by
  heavy-bond count, then linearly expand the eligible training set. Validation always uses the
  full split. Set `bond_size_curriculum: True`, `bond_size_curriculum_start_fraction: 0.25`, and
  `bond_size_curriculum_epochs: 20` to use the recommended 25%-to-full schedule. It requires
  fixed-size, non-grouped training batches and must reach the full dataset before `max_epochs`.

## Contributors

- Adamo Young ([adamoyoung](https://github.com/adamoyoung))
- Fei Wang ([7FeiW](https://github.com/7FeiW))

## Citation

```bibtex
@article{fragnnet,
  title={FraGNNet: A Deep Probabilistic Model for Mass Spectrum Prediction},
  author={Adamo Young, Fei Wang, David Wishart, Bo Wang, Hannes Röst, Russ Greiner},
  year={2024},
}
```
