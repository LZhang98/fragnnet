import argparse
import logging
import os

import wandb
from fragnnet.runner import init_run, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

    logger.info(f"Starting wandb sweep script with args: {args}")

    def agent_func():
        logger.info("Initializing run in agent_func()")
        override = None
        if args.job_id:
            config_d = load_config(args.template_fp, args.custom_fp, None)
            wandb_name = f"{config_d['wandb_name']}_{args.job_id}"
            override = [f"wandb_name={wandb_name}"]
        init_run(args.template_fp, args.custom_fp, args.wandb_mode, args.job_id, override)

    # Only load config if needed for agent (avoid wandb calls in job mode)
    if args.job_id or os.path.exists(os.path.join("job_id", f"{args.job_id}.id")):
        logger.info("Running agent directly for job_id: %s", args.job_id)
        agent_func()
    else:
        logger.info("Setting up wandb.agent with sweep_key: %s", args.sweep_key)
        config_d = load_config(args.template_fp, args.custom_fp, None)
        wandb.agent(
            args.sweep_key,
            count=1,
            entity=config_d["wandb_entity"],
            project=config_d["wandb_project"],
            function=agent_func,
        )
