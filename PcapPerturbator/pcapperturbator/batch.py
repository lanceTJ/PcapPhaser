from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .pipeline import apply_perturbations_plan
from .utils import atomic_write_json, ensure_dir, is_encrypted_dir, log, now_iso



def os_walk_skip_encrypted(in_root: Path):
    for root, dirs, files in os.walk(in_root):
        dirs[:] = [directory for directory in dirs if not is_encrypted_dir(Path(directory))]
        yield root, dirs, files



def collect_pcaps(in_root: Path) -> list[Path]:
    """Collect all input files whose names start with 'cap'."""
    pcaps: list[Path] = []
    for root, _, files in os_walk_skip_encrypted(in_root):
        for filename in files:
            if filename.startswith("cap"):
                pcaps.append(Path(root) / filename)
    return pcaps



def _out_paths(in_pcap: Path, in_root: Path, out_root: Path):
    rel = in_pcap.relative_to(in_root)
    relative_dir = rel.parent
    date_dir = str(relative_dir) if str(relative_dir) != "." else "root"
    out_dir = out_root / relative_dir
    base_name = in_pcap.name
    out_name = base_name if in_pcap.suffix.lower() == ".pcap" else f"{base_name}.pcap"
    out_pcap = out_dir / out_name
    meta_json = out_dir / f"{base_name}.metadata.json"
    return date_dir, out_dir, out_pcap, meta_json



def process_single_pcap(
    in_pcap: Path,
    in_root: Path,
    out_root: Path,
    plan: list[dict[str, Any]],
    chunk_size: int = 10000,
    selection_seed: int = 0,
    verbose: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    date_dir, out_dir, out_pcap, meta_path = _out_paths(in_pcap, in_root, out_root)
    ensure_dir(out_dir)

    if resume and out_pcap.exists():
        return {
            "input": str(in_pcap),
            "output": str(out_pcap),
            "date_dir": date_dir,
            "skipped": True,
            "reason": "exists",
        }

    started = time.time()
    result = apply_perturbations_plan(
        in_pcap=str(in_pcap),
        out_pcap=str(out_pcap),
        perturb_plan=plan,
        selection_seed=selection_seed,
        chunk_size=chunk_size,
        show_progress=False,
    )
    elapsed = time.time() - started

    metadata = {
        "pcap_file": in_pcap.name,
        "input": str(in_pcap),
        "output": str(out_pcap),
        "date_dir": date_dir,
        "plan": plan,
        "selection_seed": selection_seed,
        "chunk_size": chunk_size,
        "timestamp": now_iso(),
        "elapsed_sec": round(elapsed, 3),
        "stats": result,
    }
    atomic_write_json(meta_path, metadata)
    return metadata



def _task(args):
    return process_single_pcap(*args)



def _run_common(pcaps, in_root, out_root, plan, chunk_size, selection_seed, workers, verbose, resume, executor_factory):
    tasks = [(pcap, in_root, out_root, plan, chunk_size, selection_seed, verbose, resume) for pcap in pcaps]
    results = []
    with executor_factory(max_workers=workers) as executor:
        future_to_path = {executor.submit(_task, task): task[0] for task in tasks}
        for future in as_completed(future_to_path):
            pcap = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
                if verbose:
                    if result.get("skipped"):
                        log.info("[SKIP] %s -> exists", pcap.name)
                    else:
                        log.info("[DONE] %s in %ss", pcap.name, result.get("elapsed_sec", "?"))
            except Exception as exc:
                results.append({"status": "error", "input": str(pcap), "error": str(exc)})
                log.error("[FAIL] %s: %s", pcap.name, exc)
    return results



def run_threads(
    in_root: Path,
    out_root: Path,
    plan: list[dict[str, Any]],
    chunk_size: int = 10000,
    selection_seed: int = 0,
    workers: int = 4,
    limit: int | None = None,
    verbose: bool = False,
    resume: bool = False,
    per_file_log: bool = False,
):
    pcaps = collect_pcaps(in_root)
    if limit is not None:
        pcaps = pcaps[:limit]

    if workers <= 1:
        class InlineExecutor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, *args, **kwargs):
                from concurrent.futures import Future

                future = Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        executor_factory = lambda max_workers: InlineExecutor()
    else:
        executor_factory = lambda max_workers: ThreadPoolExecutor(max_workers=max_workers)

    return _run_common(
        pcaps,
        in_root,
        out_root,
        plan,
        chunk_size,
        selection_seed,
        workers,
        verbose or per_file_log,
        resume,
        executor_factory,
    )



def run_processes(
    in_root: Path,
    out_root: Path,
    plan,
    chunk_size: int = 10000,
    selection_seed: int = 0,
    workers: int = 2,
    limit: int | None = None,
    verbose: bool = False,
    resume: bool = False,
    per_file_log: bool = False,
):
    pcaps = collect_pcaps(in_root)
    if limit is not None:
        pcaps = pcaps[:limit]

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    executor_factory = lambda max_workers: ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    return _run_common(
        pcaps,
        in_root,
        out_root,
        plan,
        chunk_size,
        selection_seed,
        workers,
        verbose or per_file_log,
        resume,
        executor_factory,
    )
