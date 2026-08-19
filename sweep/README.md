## Weights and Biases Sweeps

W&B sweep definitions live in this directory. Their corresponding base training configs
are listed below and passed to `scripts/run_wandb_sweep.py` with `--custom_fp`.

| Sweep definition | Base training config | Purpose |
|---|---|---|
| [nist20_d3_ma_mi_sweep.yml](./nist20_d3_ma_mi_sweep.yml) | `config/sweeps/nist20_d3_ma_mi.yml` | Initial NIST20 D3 MA/MI random sweep |
| [nist20_d3_ma_mi_sweep_2.yml](./nist20_d3_ma_mi_sweep_2.yml) | `config/sweeps/nist20_d3_ma_mi_2.yml` | Follow-up NIST20 D3 MA/MI sweep over hidden sizes, dropout, LR, weight decay, and node entropy |
| [nist20_d3_ma_mi_sweep_sqrt.yml](./nist20_d3_ma_mi_sweep_sqrt.yml) | `config/sweeps/nist20_d3_ma_mi_sqrt.yml` | NIST20 D3 MA/MI sweep using sqrt-bin evaluation setup |
| [nist20_d3_ma_mi_focused_sweep.yml](./nist20_d3_ma_mi_focused_sweep.yml) | `config/sweeps/nist20_d3_ma_mi_focused.yml` | Focused follow-up sweep for molecular encoder width (`256/384/512`) and safer architecture defaults |
| [nist20_d3_ma_mi_frag_competition_sweep.yml](./nist20_d3_ma_mi_frag_competition_sweep.yml) | `config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_frag_competition.yml` | Head-only sweep over fragment top-K, transformer width, and dropout |

Older merged-data benchmark sweeps are under [merged/](./merged/).

## Create A Sweep

Use the W&B sweep definition file:

```bash
wandb sweep --project fragnnet sweep/nist20_d3_ma_mi_focused_sweep.yml
```

W&B prints a sweep id. Use that id to launch agents.

## Run A Sweep Agent

Run one local agent:

```bash
python scripts/run_wandb_sweep.py \
  --custom_fp config/sweeps/nist20_d3_ma_mi_focused.yml \
  --sweep_key <SWEEP_ID>
```

For SLURM, use the same custom config and sweep id with the existing sweep runner:

```bash
python scripts/setup_job_cc.py \
  --python_script_fp scripts/run_wandb_sweep.py \
  --custom_fp config/sweeps/nist20_d3_ma_mi_focused.yml \
  --num_sweep_jobs 50 \
  --num_sweep_agents 5 \
  --sweep_key <SWEEP_ID>
```
