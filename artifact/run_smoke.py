#!/usr/bin/env python3
"""Run the reviewer-oriented FlowPhaser smoke evaluation without Bash."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from scapy.all import RawPcapReader

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PCAPS = {
    "benign_source.pcap": {
        "packets": 5000,
        "sha256": "ad9f0faa3f29e63f9612f7180a806ec72e86cb9bacf2a0a9471623ba798fa4d5",
    },
    "cap_attack.pcap": {
        "packets": 9060,
        "sha256": "98db430aa80da6eca9473be7bbe9caa2f2a3dbbcc340c5af1eaa2b057279c01c",
    },
}


def run(label: str, command: list[str], cwd: Path) -> None:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_count(path: Path) -> int:
    reader = RawPcapReader(str(path))
    try:
        return sum(1 for _packet, _metadata in reader)
    finally:
        reader.close()


def verify_demo_pcaps() -> dict[str, dict[str, int | str]]:
    demo_dir = REPO_ROOT / "PcapPerturbator" / "demo_inputs" / "demo"
    results = {}
    for filename, expected in EXPECTED_PCAPS.items():
        path = demo_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size > 3 * 1024 * 1024:
            raise RuntimeError(f"Demo capture exceeds 3 MiB: {path} ({size} bytes)")
        digest = file_sha256(path)
        count = packet_count(path)
        if digest != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
        if count != expected["packets"]:
            raise RuntimeError(
                f"Packet-count mismatch for {path}: {count} != {expected['packets']}"
            )
        results[filename] = {"bytes": size, "packets": count, "sha256": digest}
        print(f"[PCAP] PASS {filename}: {count} packets, {size} bytes")
    return results


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Expected artifact outputs are missing: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifact_outputs" / "smoke"),
        help="Directory for generated smoke outputs",
    )
    parser.add_argument(
        "--skip-unit-tests",
        action="store_true",
        help="Run only the executable component demonstrations",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    pcap_results = verify_demo_pcaps()

    if not args.skip_unit_tests:
        for component in ("PcapPerturbator", "PSS", "FlowManifest", "PGCL"):
            run(
                f"{component} tests",
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                REPO_ROOT / component,
            )

    pcap_output = output_dir / "pcapperturbator"
    run(
        "PcapPerturbator loss demo",
        [
            sys.executable,
            "-m",
            "pcapperturbator.cli",
            "--in-root",
            str(REPO_ROOT / "PcapPerturbator" / "demo_inputs"),
            "--out-root",
            str(pcap_output),
            "--backend",
            "threads",
            "--workers",
            "1",
            "--seed",
            "42",
            "--plan",
            str(REPO_ROOT / "PcapPerturbator" / "plans" / "loss.json"),
        ],
        REPO_ROOT,
    )
    perturbed_pcap = pcap_output / "demo" / "cap_attack.pcap"
    loss_stats = pcap_output / "TM1_loss_stats.json"
    run(
        "PcapPerturbator verification",
        [
            sys.executable,
            str(REPO_ROOT / "PcapPerturbator" / "scripts" / "tm1_counts.py"),
            "--before",
            str(REPO_ROOT / "PcapPerturbator" / "demo_inputs" / "demo" / "cap_attack.pcap"),
            "--after",
            str(perturbed_pcap),
            "--mode",
            "loss",
            "--out-json",
            str(loss_stats),
        ],
        REPO_ROOT,
    )
    loss_result = json.loads(loss_stats.read_text(encoding="utf-8"))
    if loss_result["interpretable_metrics"]["lost_packets"] <= 0:
        raise RuntimeError("Loss demonstration did not remove any packets")

    flow_output = output_dir / "flowmanifest"
    template = flow_output / "manifest_template.csv"
    run(
        "FlowManifest example generation",
        [
            sys.executable,
            str(REPO_ROOT / "FlowManifest" / "examples" / "generate_tiny_manifest.py"),
            "--output",
            str(template),
        ],
        REPO_ROOT,
    )
    run(
        "FlowManifest build",
        [
            sys.executable,
            "-m",
            "FlowManifest.cli",
            "build",
            "--dataset",
            "FlowPhaser-demo",
            "--template",
            str(template),
            "--output-dir",
            str(flow_output),
        ],
        REPO_ROOT,
    )
    manifest = flow_output / "manifest.csv"
    run(
        "FlowManifest split",
        [
            sys.executable,
            "-m",
            "FlowManifest.cli",
            "split",
            "--manifest",
            str(manifest),
            "--seed",
            "42",
            "--train-ratio",
            "0.64",
            "--val-ratio",
            "0.16",
            "--test-ratio",
            "0.20",
            "--group-aware",
        ],
        REPO_ROOT,
    )
    run(
        "FlowManifest few-shot",
        [
            sys.executable,
            "-m",
            "FlowManifest.cli",
            "fewshot",
            "--manifest",
            str(manifest),
            "--k",
            "2",
            "--seeds",
            "0",
            "1",
        ],
        REPO_ROOT,
    )
    run(
        "FlowManifest leakage check",
        [
            sys.executable,
            "-m",
            "FlowManifest.cli",
            "check",
            "--manifest",
            str(manifest),
            "--indices-dir",
            str(flow_output / "indices"),
            "--strict",
        ],
        REPO_ROOT,
    )

    pgcl_data = output_dir / "pgcl_data"
    run(
        "PGCL example generation",
        [
            sys.executable,
            str(REPO_ROOT / "PGCL" / "examples" / "generate_tiny_phase_splits.py"),
            "--output-dir",
            str(pgcl_data),
        ],
        REPO_ROOT,
    )
    pgcl_output = output_dir / "pgcl"
    run(
        "PGCL CPU training",
        [
            sys.executable,
            "main.py",
            "--train-config",
            "configs/smoke.yaml",
            "--train-csv",
            str(pgcl_data / "train.csv"),
            "--val-csv",
            str(pgcl_data / "val.csv"),
            "--test-csv",
            str(pgcl_data / "test.csv"),
            "--output-dir",
            str(pgcl_output),
            "--run-name",
            "smoke",
            "--device",
            "cpu",
        ],
        REPO_ROOT / "PGCL",
    )

    required_outputs = [
        perturbed_pcap,
        loss_stats,
        manifest,
        flow_output / "splits" / "parent_split_seed42.csv",
        flow_output / "indices" / "support_k2_seed0.csv",
        pgcl_output / "smoke_3_train_pgcl_phase.csv",
        pgcl_output / "smoke_3_best_pgcl_phase.safetensors",
        pgcl_output / "smoke_3_best_pgcl_phase_finetuned.safetensors",
        pgcl_output / "smoke_3_best_pgcl_phase_metrics.json",
    ]
    require_files(required_outputs)
    metrics = json.loads(required_outputs[-1].read_text(encoding="utf-8"))
    for section in ("validation", "test"):
        if metrics[section] is None:
            raise RuntimeError(f"PGCL {section} metrics were not produced")
        for value in metrics[section].values():
            if not 0.0 <= float(value) <= 1.0:
                raise RuntimeError(f"PGCL metric outside [0, 1]: {value}")

    summary = {
        "status": "passed",
        "elapsed_sec": round(time.time() - started, 3),
        "demo_pcaps": pcap_results,
        "loss_demo": loss_result["interpretable_metrics"],
        "pgcl": metrics,
        "outputs": [str(path.relative_to(output_dir)) for path in required_outputs],
    }
    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFlowPhaser artifact smoke evaluation: PASS")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
