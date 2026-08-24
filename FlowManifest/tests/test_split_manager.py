from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from FlowManifest.split_manager import ParentFlowInfo, SplitManager


def make_flows() -> list[ParentFlowInfo]:
    flows = []
    for group_index in range(8):
        group_id = f"group-{group_index}"
        for flow_index in range(2):
            label_id = flow_index
            flows.append(
                ParentFlowInfo(
                    parent_flow_id=f"{group_id}-flow-{flow_index}",
                    group_id=group_id,
                    label="Benign" if label_id == 0 else "Malicious",
                    label_id=label_id,
                )
            )
    return flows


class SplitManagerTests(unittest.TestCase):
    def test_group_aware_split_is_deterministic_and_keeps_groups_together(self) -> None:
        flows = make_flows()
        kwargs = {
            "train_ratio": 0.50,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "group_aware": True,
        }

        first = SplitManager(seed=42).split(flows, **kwargs)
        second = SplitManager(seed=42).split(flows, **kwargs)

        self.assertEqual(first.train, second.train)
        self.assertEqual(first.val, second.val)
        self.assertEqual(first.test, second.test)

        assigned = first.train | first.val | first.test
        self.assertEqual(assigned, {flow.parent_flow_id for flow in flows})
        self.assertFalse(first.train & first.val)
        self.assertFalse(first.train & first.test)
        self.assertFalse(first.val & first.test)

        for group_id in {flow.group_id for flow in flows}:
            parent_ids = {
                flow.parent_flow_id for flow in flows if flow.group_id == group_id
            }
            placements = [
                bool(parent_ids & first.train),
                bool(parent_ids & first.val),
                bool(parent_ids & first.test),
            ]
            self.assertEqual(sum(placements), 1)

    def test_saved_split_round_trips_with_metadata(self) -> None:
        manager = SplitManager(seed=7)
        split = manager.split(
            make_flows(),
            train_ratio=0.50,
            val_ratio=0.25,
            test_ratio=0.25,
            group_aware=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "parent_split.csv"
            manager.save_split(split, str(path))
            loaded = manager.load_split(str(path))

        self.assertEqual(split.train, loaded.train)
        self.assertEqual(split.val, loaded.val)
        self.assertEqual(split.test, loaded.test)
        self.assertEqual(split.seed, loaded.seed)
        self.assertEqual(split.group_aware, loaded.group_aware)

    def test_rejects_ratios_that_do_not_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "Ratios must sum to 1"):
            SplitManager(seed=42).split(
                make_flows(),
                train_ratio=0.60,
                val_ratio=0.30,
                test_ratio=0.30,
            )


if __name__ == "__main__":
    unittest.main()
