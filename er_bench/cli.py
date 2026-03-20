"""
er-bench CLI.

Usage:
    er-bench run --dataset febrl --matcher cascade
    er-bench run --dataset icij --matcher fuzzy --output results.json
    er-bench run --dataset febrl --matcher exact --output results.csv --format csv
    er-bench list-datasets
    er-bench list-matchers
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def cmd_run(args) -> int:
    """Run a benchmark."""
    from er_bench.runner import BenchmarkRunner
    from er_bench.datasets import DATASETS
    from er_bench.matchers import MATCHERS

    dataset_kwargs = {}
    matcher_kwargs = {}

    # Parse extra kwargs from --dataset-opt KEY=VALUE
    for opt in (args.dataset_opt or []):
        k, _, v = opt.partition("=")
        # Try to cast to int/float
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        dataset_kwargs[k] = v

    for opt in (args.matcher_opt or []):
        k, _, v = opt.partition("=")
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        matcher_kwargs[k] = v

    try:
        runner = BenchmarkRunner(
            dataset=args.dataset,
            matcher=args.matcher,
            dataset_kwargs=dataset_kwargs,
            matcher_kwargs=matcher_kwargs,
            notes=args.notes or "",
        )
        report = runner.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output
    output_path: Optional[str] = args.output
    fmt: str = args.format or "json"

    if output_path:
        if fmt == "csv" or output_path.endswith(".csv"):
            report.save_csv(output_path)
        else:
            report.save_json(output_path)
        print(f"Report written to: {output_path}")

    # Always print summary to stdout
    if args.quiet:
        # Just print key metrics as JSON
        print(json.dumps({
            "precision": round(report.precision, 4),
            "recall": round(report.recall, 4),
            "f1": round(report.f1, 4),
            "reduction_ratio": round(report.reduction_ratio, 4),
        }))
    else:
        print(report.summary())

    return 0


def cmd_list_datasets(args) -> int:
    from er_bench.datasets import DATASETS
    print("Available datasets:")
    for name, cls in DATASETS.items():
        desc = getattr(cls, "description", "")
        print(f"  {name:<12} {desc}")
    return 0


def cmd_list_matchers(args) -> int:
    from er_bench.matchers import MATCHERS
    print("Available matchers:")
    for name, cls in MATCHERS.items():
        desc = cls.__doc__.strip().split("\n")[0] if cls.__doc__ else ""
        print(f"  {name:<12} {desc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="er-bench",
        description="Standardized Entity Resolution Benchmark Suite",
    )
    parser.add_argument("--version", action="version", version="er-bench 0.1.0")

    subparsers = parser.add_subparsers(dest="command")

    # run
    run_p = subparsers.add_parser("run", help="Run a benchmark")
    run_p.add_argument(
        "--dataset", "-d",
        required=True,
        help="Dataset name (febrl, icij, dblp_acm)",
    )
    run_p.add_argument(
        "--matcher", "-m",
        required=True,
        help="Matcher name (exact, fuzzy, embedding, cascade)",
    )
    run_p.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (.json or .csv)",
    )
    run_p.add_argument(
        "--format", "-f",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    run_p.add_argument(
        "--notes",
        default="",
        help="Optional notes for the report",
    )
    run_p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Print only key metrics as JSON",
    )
    run_p.add_argument(
        "--dataset-opt",
        action="append",
        metavar="KEY=VALUE",
        help="Dataset constructor kwargs (repeatable)",
    )
    run_p.add_argument(
        "--matcher-opt",
        action="append",
        metavar="KEY=VALUE",
        help="Matcher constructor kwargs (repeatable)",
    )

    # list-datasets
    subparsers.add_parser("list-datasets", help="List available datasets")

    # list-matchers
    subparsers.add_parser("list-matchers", help="List available matchers")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list-datasets":
        return cmd_list_datasets(args)
    elif args.command == "list-matchers":
        return cmd_list_matchers(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
