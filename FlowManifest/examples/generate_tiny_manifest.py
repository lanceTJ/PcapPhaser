#!/usr/bin/env python3
"""Generate a deterministic manifest template for artifact evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from FlowManifest.manifest_builder import ManifestBuilder, Variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    builder = ManifestBuilder(dataset_name="FlowPhaser-demo")
    for group_index in range(12):
        group_id = f"capture-{group_index:02d}"
        for label, label_slug, source in (
            ("Benign", "benign", "PcapPerturbator/demo_inputs/demo/benign_source.pcap"),
            ("Malicious", "attack", "PcapPerturbator/demo_inputs/demo/cap_attack.pcap"),
        ):
            parent_id = f"{group_id}-{label_slug}"
            builder.add_flow(
                parent_flow_id=parent_id,
                label=label,
                group_id=group_id,
                source_pcap=source,
                processed_path=f"csv/origin/{parent_id}.csv",
                num_packets=5000 if label == "Benign" else 9060,
                start_time=float(group_index * 1000),
                end_time=float(group_index * 1000 + 120),
            )
            builder.add_variant(
                parent_flow_id=parent_id,
                variant=Variant.CASE1_LOSS.value,
                processed_path=f"csv/case1_loss/{parent_id}.csv",
                perturbation_config='{"loss_pct": 0.1, "seed": 42}',
            )
            builder.add_variant(
                parent_flow_id=parent_id,
                variant=Variant.CASE2_RETRANSMIT.value,
                processed_path=f"csv/case2_retransmission/{parent_id}.csv",
                perturbation_config='{"retransmit_pct": 0.1, "seed": 42}',
            )
            builder.add_variant(
                parent_flow_id=parent_id,
                variant=Variant.CASE3_REORDER.value,
                processed_path=f"csv/case3_reordering/{parent_id}.csv",
                perturbation_config='{"window": 5, "seed": 42}',
            )
            builder.add_variant(
                parent_flow_id=parent_id,
                variant=Variant.CASE4_LENGTH.value,
                processed_path=f"csv/case4_length_padding/{parent_id}.csv",
                perturbation_config='{"profile": "demo", "seed": 42}',
            )
            builder.add_variant(
                parent_flow_id=parent_id,
                variant=Variant.CASE5_RATE.value,
                processed_path=f"csv/case5_timing_delay/{parent_id}.csv",
                perturbation_config='{"profile": "demo", "seed": 42}',
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    builder.save(str(output))
    print(f"Wrote {len(builder.entries)} manifest rows to {output}")


if __name__ == "__main__":
    main()
