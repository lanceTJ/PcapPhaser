
"""
Data Store - Layered data storage manager.

Open Science design principles:
- Configurable storage strategies (memory / hybrid / full)
- Traceable data lineage
- Lazy loading support
- Reproducible data selection
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable

from .manifest_builder import Variant


class StorageStrategy(str, Enum):
    """Storage strategy enumeration."""
    ON_DEMAND = "on_demand"        # Extract on demand, no persistence
    CACHE_ORIGIN = "cache_origin"  # Cache origin only
    CACHE_ALL = "cache_all"        # Cache all variants


class DataLayout(str, Enum):
    """Directory layout enumeration."""
    FLAT = "flat"               # Flat layout
    BY_SPLIT = "by_split"       # Organized by split
    BY_VARIANT = "by_variant"   # Organized by variant
    BY_CAPTURE = "by_capture"   # Organized by capture


class DataStore:
    """
    FlowManifest data storage manager.

    Example usage:
        # Basic usage
        store = DataStore(processed_dir)
        path = store.get_pcap_path(parent_flow_id, variant="origin", split="test")

        # Check cache
        if store.is_cached(parent_flow_id, variant="origin"):
            ...

        # List all origin PCAPs in a split
        pcap_paths = store.list_pcaps(split="test", variant="origin")
    """

    DEFAULT_CONFIG = {
        "storage_strategy": StorageStrategy.CACHE_ORIGIN,
        "layout": DataLayout.BY_VARIANT,
        "pcap_subdir": "pcap",
        "csv_subdir": "csv",
        "split_subdir": "splits",
        "index_subdir": "indices",
        "raw_flows_subdir": "raw_flows",
    }

    def __init__(self, processed_dir: Path, config: Optional[Dict] = None):
        self.processed_dir = Path(processed_dir)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        self.pcap_dir = self.processed_dir / self.config["pcap_subdir"]
        self.csv_dir = self.processed_dir / self.config["csv_subdir"]
        self.raw_flows_dir = self.processed_dir / self.config["raw_flows_subdir"]

        # Ensure directories exist
        for d in [self.pcap_dir, self.csv_dir, self.raw_flows_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_raw_flow_path(
        self,
        parent_flow_id: str,
        capture_id: Optional[str] = None,
    ) -> Path:
        """Get path for an origin (raw) single-flow PCAP."""
        if capture_id:
            return self.raw_flows_dir / capture_id / f"{parent_flow_id}.pcap"
        return self.raw_flows_dir / f"{parent_flow_id}.pcap"

    def get_pcap_path(
        self,
        parent_flow_id: str,
        variant: str = "origin",
        split: Optional[str] = None,
        capture_id: Optional[str] = None,
    ) -> Path:
        """
        Get PCAP path.

        Origin maps to raw_flows/; variants map to pcap/.
        Supports multiple layouts:
        - by_variant: pcap/{variant}/{capture}/{flow_id}.pcap
        - by_split: pcap/{split}/{variant}/{flow_id}.pcap
        - flat: pcap/{variant}_{flow_id}.pcap
        """
        if variant == Variant.ORIGIN.value:
            return self.get_raw_flow_path(parent_flow_id, capture_id)

        layout = DataLayout(self.config["layout"])
        rel_path = self._build_pcap_relative_path(
            parent_flow_id, variant, split, capture_id, layout
        )
        return self.pcap_dir / rel_path

    def _build_pcap_relative_path(
        self,
        parent_flow_id: str,
        variant: str,
        split: Optional[str],
        capture_id: Optional[str],
        layout: DataLayout,
    ) -> str:
        """Build relative path."""
        if layout == DataLayout.BY_VARIANT:
            if capture_id:
                return f"{variant}/{capture_id}/{parent_flow_id}.pcap"
            return f"{variant}/{parent_flow_id}.pcap"

        elif layout == DataLayout.BY_SPLIT:
            if not split:
                raise ValueError("split required for BY_SPLIT layout")
            if capture_id:
                return f"{split}/{variant}/{capture_id}/{parent_flow_id}.pcap"
            return f"{split}/{variant}/{parent_flow_id}.pcap"

        elif layout == DataLayout.BY_CAPTURE:
            if not capture_id:
                raise ValueError("capture_id required for BY_CAPTURE layout")
            return f"{capture_id}/{variant}/{parent_flow_id}.pcap"

        else:  # FLAT
            safe_variant = variant.replace("_", "-")
            return f"{safe_variant}_{parent_flow_id}.pcap"

    def get_csv_path(
        self,
        parent_flow_id: str,
        variant: str = "origin",
        split: Optional[str] = None,
        capture_id: Optional[str] = None,
    ) -> Path:
        """Get CSV feature file path."""
        layout = DataLayout(self.config["layout"])
        rel_path = self._build_pcap_relative_path(
            parent_flow_id, variant, split, capture_id, layout
        )
        return (self.csv_dir / rel_path).with_suffix(".csv")

    def is_cached(self, parent_flow_id: str, variant: str = "origin") -> bool:
        """Check if already cached."""
        path = self.get_pcap_path(parent_flow_id, variant)
        return path.exists()

    def get_or_create_variant(
        self,
        parent_flow_id: str,
        variant: str,
        origin_path: Path,
        generator: Callable[[Path, Path], None],
        split: Optional[str] = None,
        capture_id: Optional[str] = None,
    ) -> Path:
        """
        Return cached variant PCAP path, or generate on demand and cache it.

        Args:
            parent_flow_id: Stable parent flow identifier.
            variant: Variant name (e.g. case1_loss).
            origin_path: Path to the origin PCAP.
            generator: Callable(origin_path, variant_path) that produces the variant.
            split: Optional split filter.
            capture_id: Optional capture identifier.

        Returns:
            Path to the variant PCAP (cached or newly created).
        """
        if variant == Variant.ORIGIN.value:
            return self.get_raw_flow_path(parent_flow_id, capture_id)

        path = self.get_pcap_path(parent_flow_id, variant, split, capture_id)
        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        generator(origin_path, path)
        return path

    def list_pcaps(
        self,
        split: Optional[str] = None,
        variant: Optional[str] = None,
        capture_id: Optional[str] = None,
    ) -> List[Path]:
        """List matching PCAP files."""
        paths: List[Path] = []

        # Search raw_flows for origin
        if variant is None or variant == Variant.ORIGIN.value:
            paths.extend(self.raw_flows_dir.glob("**/*.pcap"))

        # Search pcap_dir for non-origin variants
        if variant is None or variant != Variant.ORIGIN.value:
            search_dir = self.pcap_dir
            if split:
                search_dir = search_dir / split
            paths.extend(search_dir.glob("**/*.pcap"))

        # Filter
        if variant and variant != Variant.ORIGIN.value:
            paths = [p for p in paths if variant in str(p)]
        if capture_id:
            paths = [p for p in paths if capture_id in str(p)]

        return paths

    def list_csvs(
        self,
        split: Optional[str] = None,
        variant: Optional[str] = None,
        capture_id: Optional[str] = None,
    ) -> List[Path]:
        """List matching CSV files."""
        if split:
            search_dir = self.csv_dir / split
        else:
            search_dir = self.csv_dir

        pattern = "**/*.csv"
        paths = list(search_dir.glob(pattern))

        if variant:
            paths = [p for p in paths if variant in str(p)]
        if capture_id:
            paths = [p for p in paths if capture_id in str(p)]

        return paths

    def get_data_summary(self) -> Dict:
        """Get data storage summary."""
        raw_flows = list(self.raw_flows_dir.glob("**/*.pcap"))
        raw_flow_count = len(raw_flows)
        raw_flow_size = sum(p.stat().st_size for p in raw_flows) / (1024 ** 2)

        variant_pcaps = list(self.pcap_dir.glob("**/*.pcap"))
        variant_pcap_count = len(variant_pcaps)
        variant_pcap_size = sum(p.stat().st_size for p in variant_pcaps) / (1024 ** 2)

        csv_count = len(list(self.csv_dir.glob("**/*.csv")))

        return {
            "raw_flow_count": raw_flow_count,
            "raw_flow_storage_mb": round(raw_flow_size, 2),
            "variant_pcap_count": variant_pcap_count,
            "variant_pcap_storage_mb": round(variant_pcap_size, 2),
            "csv_count": csv_count,
            "strategy": self.config["storage_strategy"],
            "layout": self.config["layout"],
        }


class DataLineage:
    """
    Data lineage tracker (Open Science reproducibility support).

    Records where data came from, auditable and reproducible.
    """

    def __init__(self, store: DataStore):
        self.store = store
        self.lineage_dir = store.processed_dir / "lineage"
        self.lineage_dir.mkdir(exist_ok=True)

    def record_extraction(
        self,
        parent_flow_id: str,
        source_pcap: str,
        flow_index: int,
        metadata: Dict,
    ):
        """Record flow extraction."""
        record_file = self.lineage_dir / f"{parent_flow_id}.json"
        record = {
            "type": "extraction",
            "parent_flow_id": parent_flow_id,
            "source_pcap": source_pcap,
            "flow_index": flow_index,
            "timestamp": self._get_timestamp(),
            "metadata": metadata,
        }
        self._save_record(record_file, record)

    def record_perturbation(
        self,
        parent_flow_id: str,
        variant: str,
        origin_path: str,
        method: str,
        params: Dict,
    ):
        """Record perturbation generation."""
        record_file = self.lineage_dir / f"{parent_flow_id}_{variant}.json"
        record = {
            "type": "perturbation",
            "parent_flow_id": parent_flow_id,
            "variant": variant,
            "origin_path": origin_path,
            "method": method,
            "params": params,
            "timestamp": self._get_timestamp(),
        }
        self._save_record(record_file, record)

    def get_lineage(self, parent_flow_id: str) -> List[Dict]:
        """Get data lineage."""
        records = []
        for f in self.lineage_dir.glob(f"{parent_flow_id}*.json"):
            with open(f) as fp:
                import json
                records.append(json.load(fp))
        return sorted(records, key=lambda r: r["timestamp"])

    def _get_timestamp(self) -> str:
        import time
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _save_record(self, record_file: Path, record: Dict):
        import json
        with open(record_file, "w") as fp:
            json.dump(record, fp, indent=2)
