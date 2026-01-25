# Experiment Configs

## Setting up a Training Run

To train a model, you can use the following command:

```bash
python scripts/run_pl_model_fit.py -c config/${DATASET}_${SPLIT}/${MODEL}/s${SEED}.yml
```

For example, to train a model on the NIST 2020 dataset using the InChIKey split and the NEIMS model with seed 0, you can use the following command:

```bash
python scripts/run_pl_model_fit.py -c config/nist20_inchikey/neims/s0.yml
```

If you installed the package with Weights and Biases (i.e. using the `dev` option), you can enable logging with a flag:

```bash
python scripts/run_pl_model_fit.py -c config/nist20_inchikey/neims/s0.yml -w online
```

## Available Configs

All provided configs use the NIST 2020 dataset.

### Main Experiment Configs

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548

FraGNNet-D3: [inchikey configs](./nist20_inchikey/fragnnet_d3/), [scaffold configs](./nist20_scaffold/fragnnet_d3/)

FraGNNet-D4: [inchikey configs](./nist20_inchikey/fragnnet_d4/), [scaffold configs](./nist20_scaffold/fragnnet_d4/)

NEIMS: [inchikey configs](./nist20_inchikey/neims/), [scaffold configs](./nist20_scaffold/neims/)

MassFormer: [inchikey configs](./nist20_inchikey/massformer/), [scaffold configs](./nist20_scaffold/massformer/)

ICEBERG (Generator model): [inchikey configs](./nist20_inchikey/iceberg_gen/), [scaffold configs](./nist20_scaffold/iceberg_gen/)

ICEBERG (Intensity model): [inchikey configs](./nist20_inchikey/iceberg/), [scaffold configs](./nist20_scaffold/iceberg/)

ICEBERG (Optimal Intensity model): [inchikey configs](./nist20_inchikey/iceberg_opt/), [scaffold configs](./nist20_scaffold/iceberg_opt/)

Note: the ICEBERG Optimal Intensity model is presented in the ablations section in the paper as ICEBERG (+Opt).

### Ablation Experiment Configs

All ablations are performed using the InChIKey split.

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548

FraGNNet-D3 (+Edges) [configs](./ablations/fraggnn_d3_edges/)
FraGNNet-D3 (-CE) [configs](./ablations/fraggnn_d3_noce/)
FraGNNet-D4 (-CE) [configs](./ablations/fraggnn_d4_noce/)

### Entropy Experiment Configs

All entropy experiments are performed using the Scaffold split, using the FraGNNet-D4 model.

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548
- SEED=5: 470932
- SEED=6: 499158
- SEED=7: 842766
- SEED=8: 771113
- SEED=9: 742567
- SEED=10: 315874
- SEED=11: 155679
- SEED=12: 885718
- SEED=13: 263526
- SEED=14: 623677


Baseline Ensemble: [configs](./entropy/d4_baseline_ens/)

Low Entropy Ensemble: [configs](./entropy/d4_low_ens/)

High Entropy Ensemble: [configs](./entropy/d4_high_ens/)

## Configuration Parameters

### Fingerprint Parameters (mol_params)

For models using molecular fingerprints (e.g., NEIMS), the following parameters control fingerprint generation:

- **`morgan_radius`** (int, default: 3): Radius for Morgan fingerprint computation. Must be > 0.
  - Original NEIMS paper likely uses radius=2
  - Current optimized default is 3

- **`morgan_nbits`** (int, default: 2048): Number of bits in Morgan fingerprint. Must be > 0.
  - Controls fingerprint vector size
  - Higher values provide more detailed molecular representations but increase memory/computation

- **`rdkit_nbits`** (int, default: 2048): Number of bits in RDKit fingerprint. Must be > 0.
  - Controls RDKit fingerprint vector size
  - Total fingerprint size = morgan_nbits + rdkit_nbits + 167 (MACCS)

**Example:**
```yaml
mol_params:
  fingerprint: True
  fingerprint_morgan: True
  fingerprint_rdkit: True
  fingerprint_maccs: True
  morgan_radius: 3
  morgan_nbits: 2048
  rdkit_nbits: 2048
```

### NEIMS Architecture Parameters

For NEIMS models, the following parameters control the neural network architecture:

- **`neims_bottleneck_factor`** (float, default: 0.5): Bottleneck compression ratio in NeimsBlock layers.
  - Controls the size of intermediate layer: bottleneck_size = bottleneck_factor × hidden_size
  - Lower values (e.g., 0.3) = more compression, fewer parameters, potentially faster training
  - Higher values (e.g., 0.7) = less compression, more capacity, potentially better performance
  - Original NEIMS paper uses 0.5

- **`ff_output_map_size`** (int, default: -1): Low-rank approximation size for output layers.
  - -1 = use standard Linear layers (original NEIMS)
  - >0 = use LowRankDense layers with specified rank (optimized variant)
  - Reduces parameters from ~150M to ~38.7M per output layer when using low-rank

- **`ff_output_activation`** (str, default: "relu"): Output activation function.
  - Options: "relu", "sigmoid"

**Example:**
```yaml
# Original NEIMS configuration
neims_bottleneck_factor: 0.5
ff_output_map_size: -1  # Standard layers
ff_output_activation: relu

# Optimized configuration with low-rank layers
neims_bottleneck_factor: 0.5
ff_output_map_size: 256  # Low-rank approximation
ff_output_activation: relu
```
