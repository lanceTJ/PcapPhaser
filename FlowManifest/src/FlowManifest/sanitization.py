"""
Feature Sanitization: Remove fields that could cause shortcut leakage.

Banned fields include:
- IP addresses (src/dst)
- Port numbers (src/dst)
- MAC addresses
- TLS SNI
- IP identifiers
- IP checksums
- TCP sequence/ack numbers
- TCP timestamps
- Absolute timestamps
- Capture metadata (filename, id)
- Split/variant labels
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Pattern, Set, Tuple

import pandas as pd


# Banned field patterns
BANNED_PATTERNS: List[str] = [
    # IP addresses
    r"^.*src.*ip.*$",
    r"^.*dst.*ip.*$",
    r"^.*source.*ip.*$",
    r"^.*dest.*ip.*$",
    r"^ip.*$",
    # Ports
    r"^.*src.*port.*$",
    r"^.*dst.*port.*$",
    r"^.*source.*port.*$",
    r"^.*dest.*port.*$",
    r"^port.*$",
    # MAC addresses
    r"^.*mac.*$",
    # TLS SNI
    r"^.*sni.*$",
    # IP identifiers/checksums
    r"^.*ip.*id.*$",
    r"^.*ip.*checksum.*$",
    r"^.*checksum.*$",
    # TCP sequence/ack
    r"^.*tcp.*seq.*$",
    r"^.*tcp.*ack.*$",
    r"^.*sequence.*$",
    r"^.*acknowledgment.*$",
    # TCP timestamps
    r"^.*tcp.*timestamp.*$",
    # Absolute timestamps
    r"^.*abs.*timestamp.*$",
    r"^.*absolute.*time.*$",
    # Capture metadata
    r"^.*capture.*$",
    r"^.*pcap.*$",
    r"^.*filename.*$",
    # Split/variant labels
    r"^split$",
    r"^variant$",
    r"^.*variant.*$",
    # Flow/group IDs
    r"^parent.*flow.*id$",
    r"^group.*id$",
    r"^flow.*id$",
]

# Compile regex patterns
BANNED_REGEXES: List[Pattern] = [re.compile(p, re.IGNORECASE) for p in BANNED_PATTERNS]

# Explicit banned field names
BANNED_FIELDS: Set[str] = {
    "src_ip",
    "dst_ip",
    "source_ip",
    "dest_ip",
    "src_port",
    "dst_port",
    "source_port",
    "dest_port",
    "src_mac",
    "dst_mac",
    "sni",
    "ip_id",
    "ip_checksum",
    "tcp_seq",
    "tcp_ack",
    "tcp_timestamp",
    "absolute_timestamp",
    "capture_id",
    "pcap_filename",
    "parent_flow_id",
    "group_id",
    "split",
    "variant",
    "flow_id",
}


@dataclass
class SanitizationResult:
    """Result of feature sanitization."""
    original_columns: List[str]
    removed_columns: List[str]
    kept_columns: List[str]
    sanitization_version: str


class FeatureSanitizer:
    """
    Sanitize features to prevent shortcut leakage.

    Usage:
        sanitizer = FeatureSanitizer()
        df_clean, result = sanitizer.sanitize(df)
    """

    def __init__(
        self,
        sanitization_version: str = "1.0",
        additional_banned: Optional[Set[str]] = None,
        additional_patterns: Optional[List[str]] = None,
    ):
        self.sanitization_version = sanitization_version
        self.banned_fields = BANNED_FIELDS.copy()
        self.banned_regexes = BANNED_REGEXES.copy()

        if additional_banned:
            self.banned_fields.update(additional_banned)
        if additional_patterns:
            self.banned_regexes.extend([re.compile(p, re.IGNORECASE) for p in additional_patterns])

    def is_banned(self, column: str) -> bool:
        """Check if a column should be banned."""
        # Exact match
        if column.lower() in {b.lower() for b in self.banned_fields}:
            return True

        # Pattern match
        for regex in self.banned_regexes:
            if regex.match(column):
                return True

        return False

    def sanitize(self, df: pd.DataFrame, label_column: Optional[str] = None) -> Tuple[pd.DataFrame, SanitizationResult]:
        """
        Sanitize DataFrame by removing banned fields.

        Args:
            df: Input DataFrame
            label_column: Name of label column to preserve

        Returns:
            (sanitized_df, result)
        """
        original_columns = list(df.columns)

        # Identify columns to remove
        to_remove: List[str] = []
        for col in original_columns:
            if col == label_column:
                continue  # Preserve label
            if self.is_banned(col):
                to_remove.append(col)

        kept = [col for col in original_columns if col not in to_remove]
        df_clean = df[kept].copy()

        result = SanitizationResult(
            original_columns=original_columns,
            removed_columns=to_remove,
            kept_columns=kept,
            sanitization_version=self.sanitization_version,
        )

        return df_clean, result

    def get_sanitized_column_list(self, columns: List[str], label_column: Optional[str] = None) -> List[str]:
        """Get list of columns that would be kept after sanitization."""
        kept: List[str] = []
        for col in columns:
            if col == label_column:
                kept.append(col)
            elif not self.is_banned(col):
                kept.append(col)
        return kept
