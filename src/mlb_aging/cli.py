"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlb_aging.dataset import DEFAULT_DATA_DIR
from mlb_aging.metrics import METRICS, get_metric
from mlb_aging.pipeline import run_metric


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="directory holding the CSVs"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlb-aging", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download batting data from FanGraphs")
    _add_data_dir(fetch)

    train = sub.add_parser("train", help="fit aging curves and report peak age / test MAE")
    _add_data_dir(train)
    train.add_argument(
        "--metric", action="append", choices=sorted(METRICS),
        help="metric to fit (repeatable; default: all five)",
    )
    train.add_argument("--ipw", action="store_true", help="use survivorship-corrected weights")
    train.add_argument(
        "--elite", type=float, metavar="THRESHOLD",
        help="restrict training to players with a career mean above THRESHOLD",
    )
    train.add_argument(
        "--elite-test", action="store_true",
        help="also restrict the test set to those players",
    )
    train.add_argument(
        "--cohort-metric", choices=sorted(METRICS),
        help="apply --elite to this metric's career mean instead of the fitted metric's",
    )
    train.add_argument(
        "--curve-reference", metavar="VALUE",
        help="career mean pinned when tracing the curve: a number, or 'train_mean'. "
             "Shifts the curve level only -- the peak age is unaffected.",
    )
    return parser


def _parse_curve_reference(value: str | None) -> float | str:
    """Resolve the flag, keeping the sentinel that means 'use the metric's own'."""
    if value is None:
        return "__spec__"
    if value == "train_mean":
        return value
    try:
        return float(value)
    except ValueError:
        raise SystemExit(f"--curve-reference must be a number or 'train_mean', got {value!r}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "fetch":
        from mlb_aging.fetch import fetch_all

        fetch_all(args.data_dir)
        return 0

    names = args.metric or sorted(METRICS)
    header = (
        f"{'metric':6s} {'peak':>6s} {'peak value':>12s} {'test MAE':>11s} "
        f"{'fit-wt MAE':>11s} {'n_train':>8s} {'n_test':>7s}"
    )
    print(header)
    print("-" * len(header))

    for name in names:
        result = run_metric(
            get_metric(name),
            data_dir=args.data_dir,
            ipw=args.ipw,
            elite_threshold=args.elite,
            elite_test=args.elite_test,
            cohort_metric=get_metric(args.cohort_metric) if args.cohort_metric else None,
            curve_reference=_parse_curve_reference(args.curve_reference),
        )
        spec = result.spec
        # Scored under eval_weight_col; the fit weight is shown alongside when
        # the two differ, so neither weighting is presented as the only answer.
        fit_weighted = (
            f"{result.test_mae_fit_weighted:11.4f}"
            if spec.eval_weight_col != spec.weight_col
            else f"{'—':>11s}"
        )
        print(
            f"{name:6s} {result.peak_age:6.2f} {result.peak_value:12.4f} "
            f"{result.test_mae:11.4f} {fit_weighted} "
            f"{result.n_train:8d} {result.n_test:7d}"
        )

    print(
        f"\ntest MAE is {get_metric(names[0]).eval_weight_col}-weighted; "
        "fit-wt MAE re-scores the same predictions using the fitting weight "
        "(shown only where they differ)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
