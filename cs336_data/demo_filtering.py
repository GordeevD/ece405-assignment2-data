#!/usr/bin/env python
"""
Demo script showing how to use the filtering pipeline programmatically.

This script demonstrates:
1. How to use the DataFilterPipeline class directly
2. How to process a single WET file
3. How to integrate filtering into a custom workflow
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from cs336_data.filter_data import (
    DataFilterPipeline,
    FilterStatistics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demo_basic_filtering():
    """Demonstrate basic filtering on a sample WET file."""
    print("=" * 80)
    print("DEMO: Basic Filtering Pipeline")
    print("=" * 80)
    print()

    # Create pipeline with default configuration
    pipeline = DataFilterPipeline(
        output_directory="./demo_output",
        mask_pii=False,
        filter_nsfw=True,
        filter_toxic=True,
        min_quality_score=0.0,
        preferred_quality="wiki",
    )

    print(f"Pipeline Configuration:")
    print(f"  Output directory: {pipeline.output_directory}")
    print(f"  Mask PII: {pipeline.mask_pii}")
    print(f"  Filter NSFW: {pipeline.filter_nsfw}")
    print(f"  Filter toxic: {pipeline.filter_toxic}")
    print(f"  Min quality score: {pipeline.min_quality_score}")
    print(f"  Preferred quality: {pipeline.preferred_quality}")
    print()

    # Process a WET file
    wet_file = "cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"
    if not pathlib.Path(wet_file).exists():
        print(f"Error: WET file not found: {wet_file}")
        print("Please ensure the CC WET file is at the expected location.")
        return

    try:
        output_path, stats = pipeline.process_wet_file(wet_file)
        print(f"Processing complete!")
        print(f"Output written to: {output_path}")
        print()
        print(f"Statistics:")
        print(f"  Total records: {stats.total_records}")
        print(f"  Extracted: {stats.records_extracted}")
        print(f"  English: {stats.records_english}")
        print(f"  Gopher pass: {stats.records_gopher_pass}")
        print(f"  Quality pass: {stats.records_quality_pass}")
        print(f"  Not NSFW: {stats.records_not_nsfw}")
        print(f"  Not toxic: {stats.records_not_toxic}")
        print(f"  Output: {stats.records_output}")
        print()

        # Show sample of output
        output_file = pathlib.Path(output_path)
        if output_file.exists():
            print(f"Sample of output (first 3 lines):")
            print("-" * 80)
            with open(output_file) as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    # Print truncated line for readability
                    import json

                    data = json.loads(line)
                    text_preview = data["text"][:100].replace("\n", " ")
                    print(f"URL: {data['url']}")
                    print(f"Text: {text_preview}...")
                    print()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return


def demo_custom_filtering():
    """Demonstrate custom filtering with strict quality requirements."""
    print("=" * 80)
    print("DEMO: High-Quality Filtering")
    print("=" * 80)
    print()

    # Create pipeline with high quality requirements
    pipeline = DataFilterPipeline(
        output_directory="./demo_output_hq",
        mask_pii=True,  # Mask PII for privacy
        filter_nsfw=True,
        filter_toxic=True,
        min_quality_score=0.5,  # Only keep documents with quality score >= 0.5
        preferred_quality="wiki",
    )

    print(f"High-Quality Configuration:")
    print(f"  Min quality score: {pipeline.min_quality_score}")
    print(f"  Mask PII: {pipeline.mask_pii}")
    print(f"  Expected retention: ~5-10% (vs 10-20% for balanced config)")
    print()

    # Process a WET file
    wet_file = "cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"
    if not pathlib.Path(wet_file).exists():
        print(f"Note: WET file not found for this demo")
        print("In production, you would process real WET files here.")
        return

    print("(Would process WET file here in production)")


def demo_programmatic_integration():
    """Demonstrate how to integrate filtering into a custom workflow."""
    print("=" * 80)
    print("DEMO: Programmatic Integration")
    print("=" * 80)
    print()

    print("""
This example shows how to use the filtering pipeline in your own code:

```python
from cs336_data.filter_data import DataFilterPipeline
import json

# Create a pipeline
pipeline = DataFilterPipeline(
    output_directory="./filtered",
    min_quality_score=0.3,
)

# Process a file
output_path, stats = pipeline.process_wet_file(
    "input.warc.wet.gz"
)

# Access results
print(f"Kept {stats.records_output} out of {stats.total_records} records")

# Read filtered output
with open(output_path) as f:
    for line in f:
        data = json.loads(line)
        text = data["text"]
        url = data["url"]
        # Use the filtered text for training
        process_text(text)
```

Key API Points:
- DataFilterPipeline: Main class for filtering
  - __init__(): Configure filters
  - process_wet_file(): Process a single WET file
  - extract_and_filter_record(): Filter a single record
  
- FilterStatistics (dataclass):
  - total_records: Total records seen
  - records_output: Records that passed all filters
  - rejected_records: Dict of rejection reasons and counts

Output Format:
  - JSONL file with {"text": "...", "url": "..."}
  - Newline-delimited JSON for easy streaming
  - Compatible with PyTorch datasets
""")


def main():
    """Run demonstrations."""
    parser = argparse.ArgumentParser(
        description="Demonstrate the filtering pipeline"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "basic", "hq", "integration"],
        default="all",
        help="Which demo to run",
    )

    args = parser.parse_args()

    if args.demo in ("all", "basic"):
        demo_basic_filtering()
        print()

    if args.demo in ("all", "hq"):
        demo_custom_filtering()
        print()

    if args.demo in ("all", "integration"):
        demo_programmatic_integration()
        print()


if __name__ == "__main__":
    main()
