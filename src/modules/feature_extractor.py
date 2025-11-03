# src/modules/feature_extractor.py

import os
import pandas as pd
from scapy.all import rdpcap

class FeatureExtractor:
    def extract_features(self, pcap_path: str) -> pd.DataFrame:
        """
        Extract packet-level features from a PCAP file.
        Features include: timestamp, length, direction, interval.
        """
        packets = rdpcap(pcap_path)
        data = []
        prev_ts = None
        for pkt in packets:
            if not pkt.haslayer('IP'):
                continue
            ts = pkt.time
            length = len(pkt)
            src = pkt['IP'].src
            dst = pkt['IP'].dst
            direction = 1 if src < dst else -1  # Simple direction heuristic
            interval = ts - prev_ts if prev_ts is not None else 0.0
            data.append([ts, length, direction, interval])
            prev_ts = ts
        df = pd.DataFrame(data, columns=['timestamp', 'length', 'direction', 'interval'])
        return df

    def run(self, pcap_path: str, output_path: str) -> (int, dict):
        """
        Run extraction with caching: if output exists, skip and return cached metadata.
        """
        if os.path.exists(output_path):
            return 0, {"status": "cached", "output": output_path}
        df = self.extract_features(pcap_path)
        df.to_pickle(output_path)
        metadata = {
            "input": pcap_path,
            "output": output_path,
            "duration_ms": 0,  # Placeholder, implement timing if needed
            "rows": len(df)
        }
        return 0, metadata