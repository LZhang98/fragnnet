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

    # Output JSON for wandb_download_ckpts.py
    python scripts/wandb_scripts/wandb_get_run_ids.py --group my-group --json out.json

Outputs a table of matching runs (id, name, group, state, link) by default,
or just the run IDs when --ids_only is set,
or a JSON file compatible with wandb_download_ckpts.py when --json is set.
"""

import argparse
import json
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
        nargs="+",
        choices=["running", "finished", "crashed", "failed"],
        help="Filter by run state (one or more values, e.g. --state running finished)",
    )
    parser.add_argument(
        "--ids_only",
        action="store_true",
        help="Print only run IDs, one per line (useful for scripting)",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="OUTPUT_FILE",
        help="Write JSON compatible with wandb_download_ckpts.py to this file",
    )
    parser.add_argument(
        "--best_only",
        action="store_true",
        default=True,
        help="Set best_only=true in JSON output (default: true)",
    )
    parser.add_argument(
        "--all_ckpts",
        action="store_true",
        help="Set best_only=false in JSON output (download all checkpoints)",
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
        if len(args.state) == 1:
            filters["state"] = args.state[0]
        else:
            filters["state"] = {"$in": args.state}

    runs = api.runs(f"{args.entity}/{args.project}", filters=filters or None)

    matched = []
    for run in runs:
        if args.name is not None and args.name.lower() not in run.name.lower():
            continue
        matched.append(run)

    if not matched:
        print("No runs matched.", file=sys.stderr)
        sys.exit(1)

    if args.json is not None:
        best_only = not args.all_ckpts
        records = [
            {"run_id": run.id, "description": run.name, "best_only": best_only}
            for run in matched
        ]
        with open(args.json, "w") as f:
            json.dump(records, f, indent=2)
        print(f"> Wrote {len(records)} run(s) to {args.json}")
        state_counts: dict[str, int] = {}
        for run in matched:
            state_counts[run.state] = state_counts.get(run.state, 0) + 1
        for state, count in sorted(state_counts.items()):
            print(f"  {state}: {count}")
        return

    if args.ids_only:
        for run in matched:
            print(run.id)
        return

    w_id = 12
    w_name = max(len("Name"), max(len(run.name) for run in matched))
    w_group = max(len("Group"), max(len(run.group or "") for run in matched))
    w_state = 10
    header = (
        f"{'ID':<{w_id}}  {'Name':<{w_name}}  "
        f"{'Group':<{w_group}}  {'State':<{w_state}}  Link"
    )
    print(header)
    print("-" * len(header))
    for run in matched:
        group = run.group or ""
        link = f"https://wandb.ai/{args.entity}/{args.project}/runs/{run.id}"
        print(
            f"{run.id:<{w_id}}  "
            f"{run.name:<{w_name}}  "
            f"{group:<{w_group}}  "
            f"{run.state:<{w_state}}  "
            f"{link}"
        )
    print(f"\n{len(matched)} run(s) matched.")


if __name__ == "__main__":
    main()
