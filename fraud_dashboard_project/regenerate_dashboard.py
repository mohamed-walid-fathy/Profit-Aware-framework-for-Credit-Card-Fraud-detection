#!/usr/bin/env python3
"""Rebuild the dashboard files from a saved bundle — no retraining.

Usage:
    python regenerate_dashboard.py
    python regenerate_dashboard.py --bundle dashboard_outputs/deployment_bundle.npz \
                                   --out dashboard_outputs

The bundle is written by the notebook's export cell via fraudcore.save_bundle(...).
This is the way to refresh the dashboard after, say, editing reviewer staffing or a
column label, without sitting through the full pipeline again.
"""
import argparse
import os

from fraudcore.dashboard_export import load_bundle, write_dashboard_outputs


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(here, "dashboard_outputs", "deployment_bundle.npz"))
    ap.add_argument("--out", default=os.path.join(here, "dashboard_outputs"))
    ap.add_argument("--reviewers", type=int, default=None,
                    help="override reviewer count without rebuilding the bundle")
    ap.add_argument("--reviews-per-day", type=int, default=None,
                    help="override reviews/reviewer/day")
    args = ap.parse_args()

    kwargs = load_bundle(args.bundle)
    if args.reviewers is not None:
        kwargs["reviewers"] = args.reviewers
    if args.reviews_per_day is not None:
        kwargs["reviews_per_reviewer_per_day"] = args.reviews_per_day

    res = write_dashboard_outputs(args.out, **kwargs)
    print(f"Wrote dashboard files to {res['out_dir']}")
    print(f"  {len(res['transactions'])} transactions | "
          f"review_rate={100*res['metrics']['review_rate']:.1f}% | "
          f"profit_gain={100*res['metrics']['profit_gain']:.1f}%")


if __name__ == "__main__":
    main()
