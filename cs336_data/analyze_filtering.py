"""
Analysis and reporting tools for the WET file filtering pipeline.

This script provides utilities to:
1. Analyze the output statistics from filtering runs
2. Generate reports on filter effectiveness
3. Compare filter configurations
4. Extract insights about the filtered dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FilterAnalyzer:
    """Analyze filtering results and generate reports."""

    def __init__(self, stats_json_path: str | pathlib.Path):
        """
        Load statistics from a JSON file.

        Args:
            stats_json_path: Path to statistics JSON file from filter_data.py
        """
        self.stats_file = pathlib.Path(stats_json_path)
        if not self.stats_file.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_json_path}")

        with open(self.stats_file) as f:
            self.stats = json.load(f)

    def generate_report(self) -> str:
        """Generate a comprehensive text report of the filtering results."""
        report = []

        report.append("=" * 80)
        report.append("DATA FILTERING ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Overview statistics
        report.extend(self._section_overview())
        report.append("")

        # Filter step analysis
        report.extend(self._section_filter_effectiveness())
        report.append("")

        # Rejection analysis
        report.extend(self._section_rejection_analysis())
        report.append("")

        # Data quality insights
        report.extend(self._section_data_insights())
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def _section_overview(self) -> list[str]:
        """Generate overview section."""
        lines = []
        stats = self.stats

        total = stats["total_records"]
        kept = stats["records_output"]
        retention_rate = 100 * kept / total if total > 0 else 0

        lines.append("OVERVIEW")
        lines.append("-" * 80)
        lines.append(f"Total records processed:        {total:,}")
        lines.append(f"Records kept in final output:   {kept:,}")
        lines.append(f"Overall retention rate:        {retention_rate:.2f}%")
        lines.append(f"Total data reduction:          {100*(1-kept/total):.2f}%")
        lines.append(f"Successfully processed files:  {stats['successful_files']}")
        lines.append(f"Failed files:                  {stats['failed_files']}")

        return lines

    def _section_filter_effectiveness(self) -> list[str]:
        """Analyze effectiveness of each filter stage."""
        lines = []
        lines.append("FILTER EFFECTIVENESS ANALYSIS")
        lines.append("-" * 80)

        stats = self.stats
        total = stats["total_records"]

        if total == 0:
            return lines

        filters = [
            ("Text extraction", stats["records_extracted"]),
            ("English language", stats["records_english"]),
            ("Gopher quality filters", stats["records_gopher_pass"]),
            ("Quality classifier", stats["records_quality_pass"]),
            ("NSFW filter", stats["records_not_nsfw"]),
            ("Toxic speech filter", stats["records_not_toxic"]),
            ("Final output", stats["records_output"]),
        ]

        # Build a nice table
        lines.append(
            f"{'Filter Stage':<30} {'Count':>12} {'% of Total':>12} {'Cumulative':>12}"
        )
        lines.append("-" * 80)

        for stage, count in filters:
            if count > 0:  # Only show stages that had records
                pct = 100 * count / total
                lines.append(
                    f"{stage:<30} {count:>12,} {pct:>11.2f}% {f'{100*count/total:.1f}%':>12}"
                )

        # Calculate cumulative rejection at each stage
        lines.append("-" * 80)
        lines.append("\nCUMULATIVE REJECTION AT EACH STAGE:")
        lines.append(f"{'Filter Stage':<30} {'Records Rejected':>15} {'% of input':>12}")
        lines.append("-" * 80)

        previous = total
        for stage, count in filters:
            rejected = previous - count
            rejection_pct = 100 * rejected / total if total > 0 else 0
            if rejected > 0:
                lines.append(
                    f"{stage:<30} {rejected:>15,} {rejection_pct:>11.2f}%"
                )
            previous = count

        return lines

    def _section_rejection_analysis(self) -> list[str]:
        """Analyze reasons for rejection."""
        lines = []
        lines.append("REJECTION ANALYSIS")
        lines.append("-" * 80)

        rejection_counts = self.stats.get("rejected_records", {})
        total = self.stats["total_records"]

        if not rejection_counts:
            lines.append("No detailed rejection information available")
            return lines

        # Sort by frequency
        sorted_rejections = sorted(
            rejection_counts.items(), key=lambda x: x[1], reverse=True
        )

        lines.append(f"{'Rejection Reason':<35} {'Count':>12} {'% of Total':>12}")
        lines.append("-" * 80)

        for reason, count in sorted_rejections:
            pct = 100 * count / total
            lines.append(f"{reason:<35} {count:>12,} {pct:>11.2f}%")

        # Categorize rejections
        lines.append("\nREJECTION CATEGORIES:")
        lines.append("-" * 80)

        categories = {
            "Extraction issues": ["extraction_failed", "extraction_error"],
            "Language issues": ["non_english_fr", "non_english_es", "non_english_de"],
            "Quality issues": [
                k for k in rejection_counts.keys()
                if "gopher" in k or "quality" in k
            ],
            "Content filtering": [
                k for k in rejection_counts.keys() if "nsfw" in k or "toxic" in k
            ],
            "Other": [],
        }

        # Assign remaining to "Other"
        categorized = set()
        for cat_rejections in categories.values():
            categorized.update(cat_rejections)
        categories["Other"] = [k for k in rejection_counts.keys() if k not in categorized]

        for category, rejection_types in categories.items():
            cat_count = sum(rejection_counts.get(r, 0) for r in rejection_types)
            if cat_count > 0:
                cat_pct = 100 * cat_count / total
                lines.append(f"{category:<35} {cat_count:>12,} {cat_pct:>11.2f}%")

        return lines

    def _section_data_insights(self) -> list[str]:
        """Generate insights about the filtered dataset quality."""
        lines = []
        lines.append("DATA QUALITY INSIGHTS")
        lines.append("-" * 80)

        stats = self.stats
        total = stats["total_records"]
        output = stats["records_output"]

        if total == 0 or output == 0:
            lines.append("Insufficient data for quality analysis")
            return lines

        retention = 100 * output / total

        lines.append(f"\nRetention Rate: {retention:.2f}%")

        if retention > 10:
            lines.append("✓ Good data retention - filters are appropriately tuned")
        elif retention > 1:
            lines.append("⚠ Moderate data retention - strong filtering applied")
        else:
            lines.append("⚠ Low data retention - very aggressive filtering")

        # Analyze filter coverage
        extraction_loss = 100 * (
            1 - stats["records_extracted"] / total
        )
        lang_loss = (
            100
            * (stats["records_extracted"] - stats["records_english"])
            / total
        )
        quality_loss = (
            100
            * (stats["records_english"] - stats["records_gopher_pass"])
            / total
        )

        lines.append(f"\nPrimary Loss Sources:")
        lines.append(
            f"  - Extraction failures:    {extraction_loss:6.2f}% of total"
        )
        lines.append(f"  - Non-English content:   {lang_loss:6.2f}% of total")
        lines.append(f"  - Quality filters:       {quality_loss:6.2f}% of total")

        lines.append(
            f"\nEstimated final dataset size (in records): ~{output:,}"
        )

        # Recommendations
        lines.append(f"\nRECOMMENDATIONS:")
        lines.append("-" * 80)

        if retention < 0.1:
            lines.append(
                "• Current filters are extremely aggressive. Consider relaxing"
            )
            lines.append("  quality thresholds or disabling some filters.")
        elif retention < 1:
            lines.append(
                "• Filters appear well-tuned for high-quality output, but may"
            )
            lines.append("  need volume. Consider adjusting min_quality_score.")
        else:
            lines.append(
                "• Good balance between quality and quantity. Current filter"
            )
            lines.append("  configuration appears appropriate.")

        if extraction_loss > 5:
            lines.append("• High extraction failure rate. Check HTML content quality.")

        if lang_loss > 5:
            lines.append("• Significant non-English content detected.")
            lines.append("  Consider: may indicate lower language detection confidence.")

        return lines


def main():
    """Main entry point for analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze filtering statistics and generate reports"
    )
    parser.add_argument(
        "stats_file",
        help="Path to statistics JSON file from filter_data.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write report to file (default: stdout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON format",
    )

    args = parser.parse_args()

    try:
        analyzer = FilterAnalyzer(args.stats_file)

        if args.json:
            output = json.dumps(analyzer.stats, indent=2, default=str)
        else:
            output = analyzer.generate_report()

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Report written to {args.output}")
        else:
            print(output)

    except Exception as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
