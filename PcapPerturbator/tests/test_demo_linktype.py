from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _pcap_utils import flow_iats  # noqa: E402
from pcapperturbator.manip_stages import load_packet_records  # noqa: E402


class DemoLinktypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attack_pcap = ROOT / "demo_inputs" / "demo" / "cap_attack.pcap"

    def test_linux_cooked_capture_flow_keys(self) -> None:
        records, linktype = load_packet_records(str(self.attack_pcap))
        self.assertEqual(linktype, 113)
        self.assertEqual(len(records), 9060)
        self.assertTrue(all(record.flow_id.startswith("ip|") for record in records))
        self.assertEqual(len({record.flow_id for record in records}), 199)

    def test_per_flow_iats_use_capture_linktype(self) -> None:
        values = flow_iats(str(self.attack_pcap))
        self.assertEqual(len(values), 8662)
        self.assertTrue(all(value > 0 for value in values))


if __name__ == "__main__":
    unittest.main()
