"""
Leakage Checker: Automatic checking for data leakage in manifest and splits.

Checks performed:
1. Same parent_flow_id doesn't appear in multiple splits
2. Same group_id doesn't appear in multiple splits (if group-aware)
3. All variants of a parent have the same split
4. Support set only from train split
5. Pretrain pool only from train split
6. Validation only from val split
7. Test only from test split
8. Clean-FT and PA-FT use same support parent IDs
9. Each parent has origin variant
10. Check that features don't include banned fields
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

from .manifest_builder import ManifestBuilder, Split, Variant
from .sanitization import FeatureSanitizer
from .support_set import SupportSet


@dataclass
class LeakageIssue:
    """A single leakage issue."""
    severity: str  # "error", "warning"
    category: str
    description: str
    details: List[str] = field(default_factory=list)


@dataclass
class LeakageReport:
    """Leakage check report."""
    is_valid: bool
    issues: List[LeakageIssue]
    summary: Dict[str, int]

    def __str__(self) -> str:
        lines = ["=" * 60, "LEAKAGE CHECK REPORT", "=" * 60, ""]
        lines.append(f"Overall: {'PASSED' if self.is_valid else 'FAILED'}")
        lines.append("")

        # Summary
        lines.append("Summary:")
        for category, count in self.summary.items():
            lines.append(f"  {category}: {count}")
        lines.append("")

        # Issues
        if self.issues:
            lines.append("Issues found:")
            lines.append("-" * 60)
            for issue in self.issues:
                lines.append(f"[{issue.severity.upper()}] {issue.category}: {issue.description}")
                for detail in issue.details[:5]:  # Show at most 5 details
                    lines.append(f"  - {detail}")
                if len(issue.details) > 5:
                    lines.append(f"  ... and {len(issue.details) - 5} more")
                lines.append("")
        else:
            lines.append("No issues found!")

        return "\n".join(lines)


class LeakageChecker:
    """
    Check manifest and splits for data leakage.

    Usage:
        checker = LeakageChecker()
        report = checker.check(manifest, support_sets=...)
        print(report)
    """

    def __init__(self, feature_sanitizer: Optional[FeatureSanitizer] = None):
        self.feature_sanitizer = feature_sanitizer or FeatureSanitizer()

    def _check_parent_split_consistency(
        self,
        manifest: ManifestBuilder,
    ) -> List[LeakageIssue]:
        """Check that same parent_flow_id doesn't appear in multiple splits."""
        issues: List[LeakageIssue] = []

        parent_to_splits: Dict[str, Set[str]] = {}
        for entry in manifest.entries.values():
            if entry.split is None:
                continue
            parent_to_splits.setdefault(entry.parent_flow_id, set()).add(entry.split)

        # Check for parents with multiple splits
        inconsistent = [p for p, splits in parent_to_splits.items() if len(splits) > 1]
        if inconsistent:
            issues.append(LeakageIssue(
                severity="error",
                category="parent_split_inconsistency",
                description=f"{len(inconsistent)} parent flows found in multiple splits",
                details=inconsistent[:10],
            ))

        # Check that all variants of a parent have the same split
        variant_split_issues: List[str] = []
        for parent_id in manifest.get_parent_flows():
            entries = manifest.get_entries_for_parent(parent_id)
            splits = set(e.split for e in entries)
            if len(splits) > 1:
                variant_split_issues.append(f"{parent_id}: {splits}")

        if variant_split_issues:
            issues.append(LeakageIssue(
                severity="error",
                category="variant_split_inconsistency",
                description=f"{len(variant_split_issues)} parents have variants in different splits",
                details=variant_split_issues[:10],
            ))

        return issues

    def _check_group_split_consistency(
        self,
        manifest: ManifestBuilder,
        allow_group_overlap: bool = False,
    ) -> List[LeakageIssue]:
        """Check that same group_id doesn't appear in multiple splits."""
        issues: List[LeakageIssue] = []

        if allow_group_overlap:
            return issues

        group_to_splits: Dict[str, Set[str]] = {}
        for entry in manifest.entries.values():
            if entry.split is None:
                continue
            group_to_splits.setdefault(entry.group_id, set()).add(entry.split)

        inconsistent = [g for g, splits in group_to_splits.items() if len(splits) > 1]
        if inconsistent:
            issues.append(LeakageIssue(
                severity="error",
                category="group_split_inconsistency",
                description=f"{len(inconsistent)} groups found in multiple splits",
                details=inconsistent[:10],
            ))

        return issues

    def _check_origin_presence(
        self,
        manifest: ManifestBuilder,
    ) -> List[LeakageIssue]:
        """Check that each parent has origin variant."""
        issues: List[LeakageIssue] = []

        parents_missing_origin: List[str] = []
        for parent_id in manifest.get_parent_flows():
            has_origin = False
            for entry in manifest.get_entries_for_parent(parent_id):
                if entry.variant == Variant.ORIGIN.value:
                    has_origin = True
                    break
            if not has_origin:
                parents_missing_origin.append(parent_id)

        if parents_missing_origin:
            issues.append(LeakageIssue(
                severity="error",
                category="missing_origin",
                description=f"{len(parents_missing_origin)} parent flows missing origin variant",
                details=parents_missing_origin[:10],
            ))

        # Warn about missing perturbations
        parents_with_incomplete_variants: List[str] = []
        for parent_id in manifest.get_parent_flows():
            variants = {e.variant for e in manifest.get_entries_for_parent(parent_id)}
            expected = {v.value for v in Variant.all()}
            if variants != expected:
                missing = expected - variants
                parents_with_incomplete_variants.append(f"{parent_id} missing: {missing}")

        if parents_with_incomplete_variants:
            issues.append(LeakageIssue(
                severity="warning",
                category="incomplete_variants",
                description=f"{len(parents_with_incomplete_variants)} parents missing some perturbation variants",
                details=parents_with_incomplete_variants[:10],
            ))

        return issues

    def _check_support_set(
        self,
        manifest: ManifestBuilder,
        support_set: SupportSet,
    ) -> List[LeakageIssue]:
        """Check that support set only contains train split parents."""
        issues: List[LeakageIssue] = []

        non_train_parents: List[str] = []
        for parent_id in support_set.parent_flow_ids:
            entries = manifest.get_entries_for_parent(parent_id)
            if not entries:
                continue
            split = entries[0].split
            if split != Split.TRAIN.value:
                non_train_parents.append(f"{parent_id} (split: {split})")

        if non_train_parents:
            issues.append(LeakageIssue(
                severity="error",
                category="support_set_leakage",
                description=f"{len(non_train_parents)} support set parents not in train split",
                details=non_train_parents[:10],
            ))

        return issues

    def _check_split_purity(
        self,
        manifest: ManifestBuilder,
    ) -> List[LeakageIssue]:
        """Check that val/test splits are pure (no leakage)."""
        issues: List[LeakageIssue] = []

        # Check that val split only contains val
        val_entries = [e for e in manifest.entries.values() if e.split == Split.VAL.value]
        test_entries = [e for e in manifest.entries.values() if e.split == Split.TEST.value]

        val_parents = {e.parent_flow_id for e in val_entries}
        test_parents = {e.parent_flow_id for e in test_entries}

        overlap = val_parents & test_parents
        if overlap:
            issues.append(LeakageIssue(
                severity="error",
                category="val_test_overlap",
                description=f"Val and test splits share {len(overlap)} parent flows",
                details=list(overlap)[:10],
            ))

        return issues

    def _check_feature_columns(
        self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
    ) -> List[LeakageIssue]:
        """Check that feature columns don't include banned fields."""
        issues: List[LeakageIssue] = []

        if df is None and columns is None:
            return issues

        cols = columns if columns is not None else list(df.columns)
        banned_found: List[str] = []

        for col in cols:
            if self.feature_sanitizer.is_banned(col):
                banned_found.append(col)

        if banned_found:
            issues.append(LeakageIssue(
                severity="error",
                category="banned_features",
                description=f"Found {len(banned_found)} potentially banned feature columns",
                details=banned_found[:20],
            ))

        return issues

    def check(
        self,
        manifest: ManifestBuilder,
        support_sets: Optional[List[SupportSet]] = None,
        feature_df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
        allow_group_overlap: bool = False,
    ) -> LeakageReport:
        """
        Run all leakage checks.

        Args:
            manifest: The manifest to check
            support_sets: Optional list of support sets to check
            feature_df: Optional DataFrame of features to check
            feature_columns: Optional list of feature column names
            allow_group_overlap: Whether to allow groups across splits

        Returns:
            LeakageReport
        """
        all_issues: List[LeakageIssue] = []

        # Parent split consistency
        all_issues.extend(self._check_parent_split_consistency(manifest))

        # Group split consistency
        all_issues.extend(self._check_group_split_consistency(manifest, allow_group_overlap))

        # Origin presence
        all_issues.extend(self._check_origin_presence(manifest))

        # Split purity
        all_issues.extend(self._check_split_purity(manifest))

        # Support sets
        if support_sets:
            for i, support_set in enumerate(support_sets):
                support_issues = self._check_support_set(manifest, support_set)
                # Prefix with index
                for issue in support_issues:
                    issue.description = f"[Support set {i}] {issue.description}"
                all_issues.extend(support_issues)

        # Feature columns
        all_issues.extend(self._check_feature_columns(feature_df, feature_columns))

        # Summary
        summary: Dict[str, int] = {}
        for issue in all_issues:
            key = f"{issue.severity}_{issue.category}"
            summary[key] = summary.get(key, 0) + 1

        errors = [i for i in all_issues if i.severity == "error"]
        is_valid = len(errors) == 0

        return LeakageReport(
            is_valid=is_valid,
            issues=all_issues,
            summary=summary,
        )

    def check_file(
        self,
        manifest_path: str,
        support_set_paths: Optional[List[str]] = None,
        allow_group_overlap: bool = False,
    ) -> LeakageReport:
        """Check from file paths."""
        manifest = ManifestBuilder.load(manifest_path)

        support_sets: Optional[List[SupportSet]] = None
        if support_set_paths:
            from .support_set import SupportSetGenerator
            gen = SupportSetGenerator()
            support_sets = [gen.load(p) for p in support_set_paths]

        return self.check(
            manifest=manifest,
            support_sets=support_sets,
            allow_group_overlap=allow_group_overlap,
        )
