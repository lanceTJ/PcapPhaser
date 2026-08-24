import sys
import tempfile
import unittest
from pathlib import Path

from scapy.all import IP, TCP, wrpcap

PSS_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PSS_SRC))

from modules.FeatureExtractor import FeatureExtractor
from modules.utils import canonical_flow_key


class FlowKeyTests(unittest.TestCase):
    def test_key_is_direction_independent_and_preserves_endpoint_pairs(self) -> None:
        forward = canonical_flow_key("10.0.0.1", "10.0.0.2", 1000, 2000, 6)
        reverse = canonical_flow_key("10.0.0.2", "10.0.0.1", 2000, 1000, 6)
        crossed = canonical_flow_key("10.0.0.1", "10.0.0.2", 2000, 1000, 6)

        self.assertEqual(forward, reverse)
        self.assertNotEqual(forward, crossed)

    def test_feature_extractor_keeps_crossed_port_flows_separate(self) -> None:
        packets = []
        for _ in range(3):
            packets.append(
                IP(src="10.0.0.1", dst="10.0.0.2")
                / TCP(sport=1000, dport=2000, flags="A")
            )
            packets.append(
                IP(src="10.0.0.1", dst="10.0.0.2")
                / TCP(sport=2000, dport=1000, flags="A")
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            pcap_path = Path(temp_dir) / "crossed-ports.pcap"
            wrpcap(str(pcap_path), packets)
            features = FeatureExtractor().extract_features(
                str(pcap_path), "packet_length", store=False
            )

        self.assertEqual(len(features), 2)
        self.assertEqual(sorted(len(values) for values in features.values()), [3, 3])


if __name__ == "__main__":
    unittest.main()
