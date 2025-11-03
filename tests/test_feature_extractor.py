# tests/test_feature_extractor.py

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.modules.feature_extractor import FeatureExtractor

@pytest.fixture
def mock_pcap_path():
    return "test.pcap"

def test_extract_features(mock_pcap_path):
    with patch('scapy.all.rdpcap') as mock_rdpcap:
        mock_pkt1 = MagicMock()
        mock_pkt1.haslayer.return_value = True
        mock_pkt1.time = 1.0
        mock_pkt1.__len__.return_value = 100
        mock_pkt1.__getitem__.return_value.src = '192.168.1.1'
        mock_pkt1.__getitem__.return_value.dst = '192.168.1.2'

        mock_pkt2 = MagicMock()
        mock_pkt2.haslayer.return_value = True
        mock_pkt2.time = 2.0
        mock_pkt2.__len__.return_value = 200
        mock_pkt2.__getitem__.return_value.src = '192.168.1.2'
        mock_pkt2.__getitem__.return_value.dst = '192.168.1.1'

        mock_rdpcap.return_value = [mock_pkt1, mock_pkt2]

        extractor = FeatureExtractor()
        df = extractor.extract_features(mock_pcap_path)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['timestamp', 'length', 'direction', 'interval']
        assert df.shape == (2, 4)
        assert df['length'].tolist() == [100, 200]
        assert df['interval'].tolist() == [0.0, 1.0]
        assert df['direction'].tolist() == [1, -1]  # Based on src < dst

def test_run_cache_hit(mock_pcap_path):
    output_path = "test_output.pkl"
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        extractor = FeatureExtractor()
        status, metadata = extractor.run(mock_pcap_path, output_path)
        assert status == 0
        assert metadata["status"] == "cached"

def test_run_extraction_and_save(mock_pcap_path):
    output_path = "test_output.pkl"
    with patch('os.path.exists') as mock_exists, \
         patch('pandas.DataFrame.to_pickle') as mock_to_pickle, \
         patch.object(FeatureExtractor, 'extract_features') as mock_extract:
        mock_exists.return_value = False
        mock_df = pd.DataFrame({'test': [1]})
        mock_extract.return_value = mock_df

        extractor = FeatureExtractor()
        status, metadata = extractor.run(mock_pcap_path, output_path)
        assert status == 0
        mock_to_pickle.assert_called_once_with(output_path)
        assert metadata["rows"] == 1