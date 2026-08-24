from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import run_processes, run_threads



def parse_plan_from_args(args) -> list[dict]:
    plan: list[dict] = []
    if args.loss is not None:
        plan.append({"type": "loss", "pct": float(args.loss), "params": {}})
    if args.retransmit is not None:
        plan.append({"type": "retransmit", "pct": float(args.retransmit), "params": {}})
    if args.seq_offset:
        pct, offset = args.seq_offset.split(":", 1)
        plan.append({"type": "seq_offset", "pct": float(pct), "params": {"offset": int(offset)}})
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if not plan:
        raise SystemExit("No perturbation specified. Use --loss/--retransmit/--seq-offset or --plan.")
    return plan



def main() -> None:
    parser = argparse.ArgumentParser(prog="pcapperturbator", description="PCAP perturbation batch runner")
    parser.add_argument("--in-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--backend", choices=["threads", "processes"], default="threads")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip files whose output PCAP already exists")

    parser.add_argument("--loss", type=float)
    parser.add_argument("--retransmit", type=float)
    parser.add_argument("--seq-offset", dest="seq_offset", help="pct:offset, for example 0.02:500")
    parser.add_argument("--plan", help="JSON plan file. This is required for length_manip and rate_manip stages.")

    args = parser.parse_args()
    plan = parse_plan_from_args(args)

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    runner = run_threads if args.backend == "threads" else run_processes
    results = runner(
        in_root=in_root,
        out_root=out_root,
        plan=plan,
        chunk_size=args.chunk_size,
        selection_seed=args.seed,
        workers=args.workers,
        limit=args.limit,
        verbose=args.verbose,
        resume=args.resume,
        per_file_log=args.verbose,
    )
    print(json.dumps(results[:10], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
