import argparse

from fragnnet.runner import init_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--template_fp",
        type=str,
        default="config/template.yml",
        help="path to template config file",
    )
    parser.add_argument("-c", "--custom_fp", type=str, required=False)
    parser.add_argument(
        "-w",
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("-j", "--job_id", type=str, required=False)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. model.lr=1e-4 train.batch_size=64",
    )

    args = parser.parse_args()

    model = init_run(args.template_fp, args.custom_fp, args.wandb_mode, args.job_id, args.override)
