import argparse
import os

import wandb
from fragnnet.runner import init_run, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--template_fp",
        type=str,
        default="config/template.yml",
        help="path to template config file",
    )
    parser.add_argument("-c", "--custom_fp", type=str, default="config/sweep/fragnnet_d3_sweep.yml")
    parser.add_argument(
        "-w", "--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"]
    )
    parser.add_argument("-j", "--job_id", type=str, required=False)  # Use $SLURM_ARRAY_TASK_ID
    parser.add_argument("-s", "--sweep_key", type=str, required=True)
    args = parser.parse_args()

    def agent_func():
        init_run(args.template_fp, args.custom_fp, args.wandb_mode, args.job_id, None)

    # Only load config if needed for agent (avoid wandb calls in job mode)
    if args.job_id or os.path.exists(os.path.join("job_id", f"{args.job_id}.id")):
        agent_func()
    else:
        config_d = load_config(args.template_fp, args.custom_fp, None)
        wandb.agent(
            args.sweep_key,
            count=1,
            entity=config_d["wandb_entity"],
            project=config_d["wandb_project"],
            function=agent_func,
        )
