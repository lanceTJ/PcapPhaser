"""
Flow Manifest: Parent-flow manifest-based split pipeline.

This module provides a reviewer-friendly, reproducible, leakage-free
data splitting pipeline at the parent-flow level.

Modules:
- parent_flow_id: Stable parent flow ID generation
- manifest_builder: Unified manifest CSV management
- split_manager: Parent-flow level train/val/test splitting
- support_set: Few-shot support set generation
- samplers: PA-FT balanced variant sampling
- sanitization: Feature sanitization to prevent shortcut leakage
- leakage_checker: Automatic leakage checking
- dataset: Manifest-based dataset loading
"""

__version__ = "1.0.0"
__author__ = "FlowPhaser Team"

from .parent_flow_id import (
    generate_parent_flow_id,
    canonical_5tuple,
    FlowMetadata,
    generate_variant_id,
)

from .manifest_builder import (
    ManifestBuilder,
    ManifestEntry,
    Variant,
    Split,
)

from .split_manager import (
    SplitManager,
    ParentSplit,
    ParentFlowInfo,
)

from .support_set import (
    SupportSetGenerator,
    SupportSet,
)

from .sanitization import (
    FeatureSanitizer,
    SanitizationResult,
    BANNED_FIELDS,
    BANNED_REGEXES,
)

from .leakage_checker import (
    LeakageChecker,
    LeakageReport,
    LeakageIssue,
)

from .data_store import (
    DataStore,
    StorageStrategy,
    DataLayout,
)

__all__ = [
    # Parent flow ID
    "generate_parent_flow_id",
    "canonical_5tuple",
    "FlowMetadata",
    "generate_variant_id",
    # Manifest
    "ManifestBuilder",
    "ManifestEntry",
    "Variant",
    "Split",
    # Split manager
    "SplitManager",
    "ParentSplit",
    "ParentFlowInfo",
    # Support set
    "SupportSetGenerator",
    "SupportSet",
    # Sanitization
    "FeatureSanitizer",
    "SanitizationResult",
    "BANNED_FIELDS",
    "BANNED_REGEXES",
    # Leakage checker
    "LeakageChecker",
    "LeakageReport",
    "LeakageIssue",
    # Data store
    "DataStore",
    "StorageStrategy",
    "DataLayout",
]

try:
    from .samplers import (
        PAFTSampler,
        BalancedVariantBatchSampler,
        create_paft_batch_sampler,
    )
    from .dataset import (
        ManifestDataset,
        DatasetFromManifest,
        DatasetMode,
        PretrainMode,
        CleanFTMode,
        PAFTMode,
        ValMode,
        TestMode,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__ += [
        "PAFTSampler",
        "BalancedVariantBatchSampler",
        "create_paft_batch_sampler",
        "ManifestDataset",
        "DatasetFromManifest",
        "DatasetMode",
        "PretrainMode",
        "CleanFTMode",
        "PAFTMode",
        "ValMode",
        "TestMode",
    ]

try:
    from .pcap_integration import (
        FlowKey,
        FlowStats,
        PerturbationPlan,
        PcapManifestPipeline,
        extract_flows_from_pcap,
        write_single_flow_pcap,
        create_manifest_from_existing_pcaps,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scapy":
        raise
else:
    __all__ += [
        "FlowKey",
        "FlowStats",
        "PerturbationPlan",
        "PcapManifestPipeline",
        "extract_flows_from_pcap",
        "write_single_flow_pcap",
        "create_manifest_from_existing_pcaps",
    ]
