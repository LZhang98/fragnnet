"""Get wandb run IDs filtered by run name or group.

Usage:
    # Filter by run name (substring match)
    python scripts/wandb_scripts/wandb_get_run_ids.py --name my-run

    # Filter by group
    python scripts/wandb_scripts/wandb_get_run_ids.py --group my-sweep-group

    # Both filters (AND logic)
    python scripts/wandb_scripts/wandb_get_run_ids.py --name my-run --group my-group

    # Print only IDs (one per line, useful for piping)
    python scripts/wandb_scripts/wandb_get_run_ids.py --group my-group --ids_only

Outputs a table of matching runs (id, name, group, state) by default,
or just the run IDs when --ids_only is set.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default=None, help="Filter by run name (substring match, case-insensitive)")
    parser.add_argument("--group", default=None, help="Filter by exact group name")
    parser.add_argument("--entity", default="frag-gnn", help="WandB entity (default: frag-gnn)")
    parser.add_argument("--project", default="frag-gnn", help="WandB project (default: frag-gnn)")
    parser.add_argument(
        "--state",
        default=None,
        choices=["running", "finished", "crashed", "failed"],
        help="Filter by run state",
    )
    parser.add_argument(
        "--ids_only",
        action="store_true",
        help="Print only run IDs, one per line (useful for scripting)",
    )
    args = parser.parse_args()

    if args.name is None and args.group is None:
        parser.error("Provide at least one of --name or --group")

    import wandb

    api = wandb.Api()

    filters: dict = {}
    if args.group is not None:
        filters["group"] = args.group
    if args.state is not None:
        filters["state"] = args.state

    runs = api.runs(f"{args.entity}/{args.project}", filters=filters or None)

    matched = []
    for run in runs:
        if args.name is not None and args.name.lower() not in run.name.lower():
            continue
        matched.append(run)

    if not matched:
        print("No runs matched.", file=sys.stderr)
        sys.exit(1)

    if args.ids_only:
        for run in matched:
            print(run.id)
        return

    col_widths = (12, 30, 30, 10)
    header = f"{'ID':<{col_widths[0]}}  {'Name':<{col_widths[1]}}  {'Group':<{col_widths[2]}}  {'State':<{col_widths[3]}}"
    print(header)
    print("-" * len(header))
    for run in matched:
        group = run.group or ""
        print(
            f"{run.id:<{col_widths[0]}}  "
            f"{run.name[:col_widths[1]]:<{col_widths[1]}}  "
            f"{group[:col_widths[2]]:<{col_widths[2]}}  "
            f"{run.state:<{col_widths[3]}}"
        )
    print(f"\n{len(matched)} run(s) matched.")


if __name__ == "__main__":
    main()
