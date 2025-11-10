# tests/test_feature_extractor.py

import pytest
import pandas as pd
import os
import tempfile
from modules.FeatureExtractor import FeatureExtractor

@pytest.fixture
def real_pcap_path():
    # Return the real PCAP path for integration testing
    return "/mnt/raid/luohaoran/cicids2018/SaP/phased_dataset_gen/tests/capEC2AMAZ-O4EL3NG-172.31.69.29"

def test_extract_features(real_pcap_path):
    extractor = FeatureExtractor()
    df = extractor.extract_features(real_pcap_path)
    
    # Assert basic DataFrame properties from real data
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['timestamp', 'length', 'direction', 'interval']
    assert df.shape[0] > 0  # Ensure at least one packet extracted
    assert df.shape[1] == 4  # Exact column count
    assert df['length'].min() >= 0  # Length should be non-negative
    assert df['interval'].iloc[0] == 0.0  # First interval is always 0

def test_run_cache_hit():
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_output:
        output_path = temp_output.name
    try:
        extractor = FeatureExtractor()
        status, metadata = extractor.run("dummy_path", output_path)  # Dummy input, but cache hit due to existing file
        assert status == 0
        assert metadata["status"] == "cached"
    finally:
        os.remove(output_path)

# tests/test_feature_extractor.py (modified test_run_extraction_and_save)

def test_run_extraction_and_save(real_pcap_path):
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_output:
        output_path = temp_output.name
    try:
        if os.path.exists(output_path):
            os.remove(output_path)  # Ensure no cache hit
        extractor = FeatureExtractor()
        status, metadata = extractor.run(real_pcap_path, output_path)
        assert status == 0
        assert os.path.exists(output_path)  # File saved
        assert metadata["rows"] == 309471  # Expected rows from real PCAP (adjust if file changes)
        assert "duration_ms" in metadata and metadata["duration_ms"] >= 0  # Allow >=0 for precision issues
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)