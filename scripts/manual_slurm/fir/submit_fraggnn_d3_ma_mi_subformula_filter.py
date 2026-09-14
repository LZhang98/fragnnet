#!/usr/bin/env python
"""Generate (but do not submit) the Fir SLURM job for the fraggnn_d3 subformula-filter baseline.

Python equivalent of submit_fraggnn_d3_ma_mi_subformula_filter.sh, wrapping
scripts/setup_job_cc.py the same way as the existing
submit_masshatter_merged_architectures.sh convention.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "slurm_scripts"
JOB_DIR = REPO_ROOT / "jobs"

JOB_NAME = "fraggnn_d3_ma_mi_subformula_filter"
CONFIG = "config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_subformula_filter.yml"


def main() -> None:
    config_fp = REPO_ROOT / CONFIG
    if not config_fp.is_file():
        sys.exit(f"Missing config: {config_fp}")

    subprocess.run(
        [
            sys.executable,
            "scripts/setup_job_cc.py",
            "--job_name",
            JOB_NAME,
            "--python_script_fp",
            "scripts/run_pl_model_fit.py",
            "--frag_gnn_dp",
            str(REPO_ROOT),
            "--script_dp",
            str(SCRIPT_DIR),
            "--job_dp",
            str(JOB_DIR),
            "--gpu_profile",
            "fir:h100-3g.40gb",
            "--time",
            "24:00:00",
            "--cc_account",
            "aip-rgreiner",
            "--cc_email",
            "fw4@ualberta.ca",
            "--script_kwargs",
            "w=online",
            f"c={CONFIG}",
            "--no_submit=True",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    job_script = SCRIPT_DIR / f"{JOB_NAME}.sh"
    print(f"Review {job_script}, then run:")
    print(f"  sbatch {job_script}")


if __name__ == "__main__":
    main()
