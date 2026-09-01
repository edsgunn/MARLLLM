"""Command-line entrypoint.

    .venv/bin/python -m sanctioning <command> --config configs/sanctioning.yaml [filters]

Commands: extract | cluster | rung0 | rung1 | validate | label
Filters (where applicable): --substrate, --run-id (repeatable), --iter-min,
--iter-max, --limit (extract only), --force (rung1 only).
"""

from __future__ import annotations

import argparse

from .config import Config


def _parent(with_limit=False, with_force=False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", required=True, help="path to the pipeline YAML")
    p.add_argument("--substrate", action="append", default=None,
                   help="restrict to this substrate (repeatable)")
    p.add_argument("--run-id", action="append", default=None, dest="run_id",
                   help="restrict to this run_id (repeatable)")
    p.add_argument("--iter-min", type=int, default=None)
    p.add_argument("--iter-max", type=int, default=None)
    if with_limit:
        p.add_argument("--limit", type=int, default=None,
                       help="cap number of posts (smoke tests)")
    if with_force:
        p.add_argument("--force", action="store_true",
                       help="emit Rung-1 despite a failing/missing validation gate")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sanctioning")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extract", parents=[_parent(with_limit=True)],
                   help="run the judge over a corpus slice")
    sub.add_parser("cluster", parents=[_parent()],
                   help="embed + cluster trigger_descriptions")
    sub.add_parser("rung0", parents=[_parent()], help="base-rate table + plots")
    sub.add_parser("rung1", parents=[_parent(with_force=True)],
                   help="consistency/MI tables + plots (gated)")
    sub.add_parser("validate", parents=[_parent()],
                   help="score the judge against hand labels")
    sub.add_parser("label", parents=[_parent()], help="hand-label a stratified sample")

    args = ap.parse_args(argv)
    cfg = Config.load(args.config)

    subs = set(args.substrate) if getattr(args, "substrate", None) else None
    rids = set(args.run_id) if getattr(args, "run_id", None) else None
    common = dict(substrates=subs, run_ids=rids,
                  iter_min=args.iter_min, iter_max=args.iter_max)

    if args.cmd == "extract":
        from .extract import run_extract
        run_extract(cfg, limit=args.limit, **common)
    elif args.cmd == "cluster":
        from .cluster import run_cluster
        run_cluster(cfg, **common)
    elif args.cmd == "rung0":
        from .rung0 import run_rung0
        run_rung0(cfg, **common)
    elif args.cmd == "rung1":
        from .rung1 import run_rung1
        run_rung1(cfg, force=args.force, **common)
    elif args.cmd == "validate":
        from .validate import run_validate
        run_validate(cfg, **common)
    elif args.cmd == "label":
        from .validate import run_label
        run_label(cfg, **common)


if __name__ == "__main__":
    main()
