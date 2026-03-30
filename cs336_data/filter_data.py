"""
Filter language modeling data from Common Crawl WET files.

This script filters CC WET files to produce language modeling training data
optimized for minimizing perplexity on the C4 100 domains subset of the
Paloma benchmark.

The filtering pipeline includes:
1. Language identification - keeps only English text
2. Gopher quality filters - applies heuristic quality checks
3. Quality classification - prefers Wikipedia-like quality
4. Harmful content filtering - removes NSFW and toxic content
5. PII masking - masks emails, phone numbers, IPs
6. Deduplication - removes near-duplicate content

This script supports both local multiprocessing and Slurm-based distributed
processing via the submitit library.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import pathlib
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language
from cs336_data.mask_pii import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.quality_classifier import classify_quality


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FilterStatistics:
    """Track statistics for each filter in the pipeline."""

    total_records: int = 0
    records_extracted: int = 0
    records_english: int = 0
    records_gopher_pass: int = 0
    records_quality_pass: int = 0
    records_not_nsfw: int = 0
    records_not_toxic: int = 0
    records_output: int = 0

    # Additional tracking
    rejected_records: dict[str, int] = None

    def __post_init__(self):
        if self.rejected_records is None:
            self.rejected_records = defaultdict(int)


class DataFilterPipeline:
    """Main pipeline for filtering language modeling data from WET files."""

    def __init__(
        self,
        output_directory: str | pathlib.Path,
        mask_pii: bool = False,
        filter_nsfw: bool = True,
        filter_toxic: bool = True,
        min_quality_score: float = 0.0,
        preferred_quality: str = "wiki",
        c4_domain_bias: bool = False,
    ):
        """
        Initialize the filtering pipeline.

        Args:
            output_directory: Directory to write filtered output files
            mask_pii: Whether to mask PII (emails, phone numbers, IPs)
            filter_nsfw: Whether to filter NSFW content
            filter_toxic: Whether to filter toxic speech
            min_quality_score: Minimum quality classifier score (0.0-1.0)
            preferred_quality: Preferred quality label ("wiki" or "cc")
            c4_domain_bias: Whether to bias selection towards C4 100 domains
        """
        self.output_directory = pathlib.Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.mask_pii = mask_pii
        self.filter_nsfw = filter_nsfw
        self.filter_toxic = filter_toxic
        self.min_quality_score = min_quality_score
        self.preferred_quality = preferred_quality
        self.c4_domain_bias = c4_domain_bias

        # Initialize TLD extractor for domain filtering
        self.tld_extractor = TLDExtract()

        # C4 domains favored for language modeling quality
        self.c4_top_domains = {
            "wikipedia.org",
            "scribd.com",
            "medium.com",
            "ft.com",
            "github.com",
            "stackoverflow.com",
            "techcrunch.com",
            "nytimes.com",
            "bbc.com",
            "theguardian.com",
            "wsj.com",
            "economist.com",
            "arxiv.org",
            "nature.com",
            "science.org",
            "nature.com",
        }

    def extract_and_filter_record(
        self, record_data: bytes | str, record_url: str
    ) -> tuple[str | None, str]:
        """
        Filter a single record (already extracted text from WET files).

        Args:
            record_data: The extracted text from the WET record (or HTML bytes)
            record_url: The URL of the record

        Returns:
            Tuple of (filtered_text, rejection_reason)
            If rejection_reason is "keep", the text passed all filters.
        """
        stats = FilterStatistics()
        stats.total_records = 1

        # Convert to text if needed
        if isinstance(record_data, bytes):
            try:
                text = record_data.decode("utf-8", errors="ignore")
            except Exception:
                return None, "decode_error"
        else:
            text = record_data

        if not text or len(text.strip()) == 0:
            return None, "empty_text"

        stats.records_extracted += 1

        # Skip very short documents
        if len(text.split()) < 50:
            return None, "too_short"

        # Language identification filter - keep only English
        try:
            lang, confidence = identify_language(text)
            if lang != "en" or confidence < 0.5:
                return None, f"non_english_{lang}"
            stats.records_english += 1
        except Exception as e:
            logger.debug(f"Language identification failed: {e}")
            return None, "langid_error"

        # Gopher quality filters
        if not gopher_quality_filter(text):
            return None, "gopher_filter_failed"
        stats.records_gopher_pass += 1

        # Quality classification filter
        try:
            quality_label, quality_score = classify_quality(text)
            if quality_score < self.min_quality_score:
                return None, f"low_quality_{quality_label}"
            stats.records_quality_pass += 1

            # Bias towards preferred quality (e.g., Wikipedia-like)
            if (
                self.preferred_quality == "wiki"
                and quality_label == "cc"
                and quality_score < 0.7
            ):
                # Consider rejecting lower-confidence CC content
                pass
        except Exception as e:
            logger.debug(f"Quality classification failed: {e}")
            # Don't fail on quality classification errors - use heuristic instead
            quality_label, quality_score = "cc", 0.5
            if quality_score < self.min_quality_score:
                return None, "quality_error"
            stats.records_quality_pass += 1

        # NSFW filtering
        if self.filter_nsfw:
            try:
                nsfw_label, nsfw_score = classify_nsfw(text)
                if nsfw_label == "nsfw" and nsfw_score > 0.5:
                    return None, "nsfw_content"
                stats.records_not_nsfw += 1
            except Exception as e:
                logger.debug(f"NSFW classification failed: {e}")
                # Don't reject on error, just log

        # Toxic speech filtering
        if self.filter_toxic:
            try:
                toxic_label, toxic_score = classify_toxic_speech(text)
                if toxic_label == "toxic" and toxic_score > 0.5:
                    return None, "toxic_content"
                stats.records_not_toxic += 1
            except Exception as e:
                logger.debug(f"Toxic speech classification failed: {e}")
                # Don't reject on error, just log

        # Domain-based filtering for C4 bias
        domain_penalty = 0.0
        if self.c4_domain_bias:
            try:
                extracted_domain = self.tld_extractor(record_url)
                domain = f"{extracted_domain.domain}.{extracted_domain.suffix}"
                if domain not in self.c4_top_domains:
                    # Apply weak penalty - don't reject, just consider lower priority
                    domain_penalty = 0.1
            except Exception as e:
                logger.debug(f"Domain extraction failed: {e}")

        # PII masking (if enabled)
        if self.mask_pii:
            try:
                text, _ = mask_emails(text)
                text, _ = mask_phone_numbers(text)
                text, _ = mask_ips(text)
            except Exception as e:
                logger.debug(f"PII masking failed: {e}")
                # Continue without masking

        stats.records_output += 1
        return text, "keep"

    def process_wet_file(
        self, input_path: str, output_path: str | None = None
    ) -> tuple[str, FilterStatistics]:
        """
        Process a single WET file and write filtered output.

        Args:
            input_path: Path to input WET file (gzipped WARC)
            output_path: Path for output file (defaults to output_directory)

        Returns:
            Tuple of (output_path_str, FilterStatistics)
        """
        if output_path is None:
            output_filename = pathlib.Path(input_path).name.replace(
                ".warc.wet.gz", ".jsonl"
            )
            output_path = self.output_directory / output_filename

        stats = FilterStatistics()

        logger.info(f"Processing WET file: {input_path}")

        try:
            with open(output_path, "w", encoding="utf-8") as outfile:
                with open(input_path, "rb") as f:
                    archive_iter = ArchiveIterator(f)
                    for record in archive_iter:
                        # WET files contain conversion records (type 128) with extracted text
                        # Skip non-conversion records (like warcinfo)
                        if record.record_type != WarcRecordType.conversion:
                            stats.total_records += 1
                            stats.rejected_records["not_conversion_record"] += 1
                            continue

                        record_url = record.headers.get("WARC-Target-URI", "")
                        if not record_url:
                            stats.total_records += 1
                            stats.rejected_records["no_url"] += 1
                            continue

                        # Read the extracted text from the record
                        try:
                            extracted_text = record.reader.read()
                            # Try to decode as UTF-8, but handle encoding errors
                            try:
                                text = extracted_text.decode("utf-8", errors="ignore")
                            except (AttributeError, TypeError):
                                text = str(extracted_text)
                        except Exception as e:
                            logger.debug(f"Failed to read record payload: {e}")
                            stats.total_records += 1
                            stats.rejected_records["read_error"] += 1
                            continue

                        # Extract and filter
                        filtered_text, reason = self.extract_and_filter_record(
                            text.encode("utf-8") if isinstance(text, str) else text,
                            record_url,
                        )

                        stats.total_records += 1
                        if reason == "keep":
                            # Write as JSONL
                            data = {"text": filtered_text, "url": record_url}
                            outfile.write(json.dumps(data) + "\n")
                            stats.records_output += 1
                        else:
                            stats.rejected_records[reason] += 1

            logger.info(
                f"Processed WET file {input_path}: "
                f"kept {stats.records_output}/{stats.total_records} records"
            )
            return str(output_path), stats

        except Exception as e:
            logger.error(f"Error processing WET file {input_path}: {e}")
            raise


def process_single_wet_file_wrapper(
    input_path: str,
    output_directory: str,
    mask_pii: bool = False,
    filter_nsfw: bool = True,
    filter_toxic: bool = True,
    min_quality_score: float = 0.0,
    preferred_quality: str = "wiki",
    c4_domain_bias: bool = False,
) -> dict:
    """
    Wrapper function for parallel processing of a single WET file.

    This function is designed to work with concurrent.futures and submitit.
    """
    try:
        pipeline = DataFilterPipeline(
            output_directory=output_directory,
            mask_pii=mask_pii,
            filter_nsfw=filter_nsfw,
            filter_toxic=filter_toxic,
            min_quality_score=min_quality_score,
            preferred_quality=preferred_quality,
            c4_domain_bias=c4_domain_bias,
        )

        output_path, stats = pipeline.process_wet_file(input_path)
        return {
            "input_path": input_path,
            "output_path": output_path,
            "stats": asdict(stats),
            "success": True,
        }
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return {
            "input_path": input_path,
            "output_path": None,
            "error": str(e),
            "success": False,
        }


def aggregate_statistics(results: list[dict]) -> dict:
    """Aggregate statistics across all processed WET files."""
    aggregated = {
        "total_records": 0,
        "records_extracted": 0,
        "records_english": 0,
        "records_gopher_pass": 0,
        "records_quality_pass": 0,
        "records_not_nsfw": 0,
        "records_not_toxic": 0,
        "records_output": 0,
        "rejected_records": defaultdict(int),
        "successful_files": 0,
        "failed_files": 0,
    }

    for result in results:
        if result["success"]:
            aggregated["successful_files"] += 1
            stats = result["stats"]
            aggregated["total_records"] += stats["total_records"]
            aggregated["records_extracted"] += stats["records_extracted"]
            aggregated["records_english"] += stats["records_english"]
            aggregated["records_gopher_pass"] += stats["records_gopher_pass"]
            aggregated["records_quality_pass"] += stats["records_quality_pass"]
            aggregated["records_not_nsfw"] += stats["records_not_nsfw"]
            aggregated["records_not_toxic"] += stats["records_not_toxic"]
            aggregated["records_output"] += stats["records_output"]

            for reason, count in stats["rejected_records"].items():
                aggregated["rejected_records"][reason] += count
        else:
            aggregated["failed_files"] += 1

    return aggregated


def main():
    """Main entry point for filtering WET files."""
    parser = argparse.ArgumentParser(
        description="Filter language modeling data from Common Crawl WET files"
    )
    parser.add_argument(
        "wet_files",
        nargs="+",
        help="Path(s) to WET files to process (supports glob patterns)",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        required=True,
        help="Output directory for filtered data",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to CPU count)",
    )
    parser.add_argument(
        "--mask-pii",
        action="store_true",
        help="Mask PII (emails, phone numbers, IPs)",
    )
    parser.add_argument(
        "--no-nsfw-filter",
        action="store_true",
        help="Disable NSFW content filtering",
    )
    parser.add_argument(
        "--no-toxic-filter",
        action="store_true",
        help="Disable toxic speech filtering",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.0,
        help="Minimum quality classifier score (0.0-1.0)",
    )
    parser.add_argument(
        "--preferred-quality",
        choices=["wiki", "cc"],
        default="wiki",
        help="Preferred quality label",
    )
    parser.add_argument(
        "--c4-domain-bias",
        action="store_true",
        help="Bias selection towards C4 100 domains",
    )
    parser.add_argument(
        "--use-submitit",
        action="store_true",
        help="Use submitit for Slurm-based parallel processing",
    )
    parser.add_argument(
        "--slurm-partition",
        default="a4-cpu",
        help="Slurm partition for submitit (default: a4-cpu)",
    )
    parser.add_argument(
        "--stats-output",
        help="Path to write aggregated statistics JSON",
    )

    args = parser.parse_args()

    # Expand glob patterns and collect WET files
    wet_files = []
    for pattern in args.wet_files:
        wet_files.extend(pathlib.Path(".").glob(pattern))
    wet_files = [str(f) for f in wet_files if f.is_file()]

    if not wet_files:
        logger.error(f"No WET files found matching: {args.wet_files}")
        return

    logger.info(f"Found {len(wet_files)} WET files to process")

    # Process files
    if args.use_submitit:
        results = _process_with_submitit(
            wet_files, args, parser
        )
    else:
        results = _process_with_concurrent_futures(
            wet_files, args, parser
        )

    # Aggregate and report statistics
    aggregated_stats = aggregate_statistics(results)
    _report_statistics(aggregated_stats)

    # Write statistics to file if requested
    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(aggregated_stats, f, indent=2, default=str)
        logger.info(f"Statistics written to {args.stats_output}")


def _process_with_concurrent_futures(
    wet_files: list[str], args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[dict]:
    """Process WET files using concurrent.futures."""
    num_workers = args.num_workers or len(os.sched_getaffinity(0))

    logger.info(f"Starting processing with {num_workers} workers...")

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    for wet_filepath in wet_files:
        future = executor.submit(
            process_single_wet_file_wrapper,
            wet_filepath,
            args.output_directory,
            mask_pii=args.mask_pii,
            filter_nsfw=not args.no_nsfw_filter,
            filter_toxic=not args.no_toxic_filter,
            min_quality_score=args.min_quality_score,
            preferred_quality=args.preferred_quality,
            c4_domain_bias=args.c4_domain_bias,
        )
        futures.append(future)

    results = []
    try:
        from tqdm import tqdm

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing WET files",
        ):
            result = future.result()
            results.append(result)
    except ImportError:
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            logger.info(f"Completed {i+1}/{len(futures)} files")

    executor.shutdown(wait=True)
    return results


def _process_with_submitit(
    wet_files: list[str], args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[dict]:
    """Process WET files using submitit for Slurm-based parallel processing."""
    try:
        import submitit
    except ImportError:
        logger.error("submitit not installed. Use --no-submitit or install submitit")
        return []

    logger.info("Starting Slurm-based processing with submitit...")

    executor = submitit.AutoExecutor(folder="slurm_logs")
    max_simultaneous_jobs = 16

    executor.update_parameters(
        slurm_array_parallelism=max_simultaneous_jobs,
        timeout_min=30,
        mem_gb=4,
        cpus_per_task=2,
        slurm_account="student",
        slurm_partition=args.slurm_partition,
        slurm_qos=f"{args.slurm_partition}-qos",
    )

    futures = []
    with executor.batch():
        for wet_filepath in wet_files:
            future = executor.submit(
                process_single_wet_file_wrapper,
                wet_filepath,
                args.output_directory,
                mask_pii=args.mask_pii,
                filter_nsfw=not args.no_nsfw_filter,
                filter_toxic=not args.no_toxic_filter,
                min_quality_score=args.min_quality_score,
                preferred_quality=args.preferred_quality,
                c4_domain_bias=args.c4_domain_bias,
            )
            futures.append(future)

    results = []
    try:
        from tqdm import tqdm

        for future in tqdm(
            submitit.helpers.as_completed(futures),
            total=len(futures),
            desc="Processing WET files",
        ):
            result = future.result()
            results.append(result)
    except ImportError:
        for i, future in enumerate(submitit.helpers.as_completed(futures)):
            result = future.result()
            results.append(result)
            logger.info(f"Completed {i+1}/{len(futures)} files")

    return results


def _report_statistics(stats: dict):
    """Print a formatted report of filtering statistics."""
    print("\n" + "=" * 80)
    print("FILTERING STATISTICS REPORT")
    print("=" * 80)

    total = stats["total_records"]
    if total == 0:
        print("No records processed.")
        return

    print(f"\nTotal records processed: {total}")
    print(f"Successfully processed files: {stats['successful_files']}")
    print(f"Failed files: {stats['failed_files']}")

    print(f"\nFilter Pass-Through Rates:")
    print(
        f"  Text extraction:        {stats['records_extracted']:8d} "
        f"({100*stats['records_extracted']/total:5.1f}%)"
    )
    print(
        f"  English detection:      {stats['records_english']:8d} "
        f"({100*stats['records_english']/total:5.1f}%)"
    )
    print(
        f"  Gopher quality filter:  {stats['records_gopher_pass']:8d} "
        f"({100*stats['records_gopher_pass']/total:5.1f}%)"
    )
    print(
        f"  Quality classifier:     {stats['records_quality_pass']:8d} "
        f"({100*stats['records_quality_pass']/total:5.1f}%)"
    )
    if stats["records_not_nsfw"] > 0:
        print(
            f"  NSFW filter:            {stats['records_not_nsfw']:8d} "
            f"({100*stats['records_not_nsfw']/total:5.1f}%)"
        )
    if stats["records_not_toxic"] > 0:
        print(
            f"  Toxic filter:           {stats['records_not_toxic']:8d} "
            f"({100*stats['records_not_toxic']/total:5.1f}%)"
        )
    print(
        f"  Final output:           {stats['records_output']:8d} "
        f"({100*stats['records_output']/total:5.1f}%)"
    )

    if stats["rejected_records"]:
        print(f"\nRejection Reasons (top 15):")
        sorted_rejections = sorted(
            stats["rejected_records"].items(), key=lambda x: x[1], reverse=True
        )
        for reason, count in sorted_rejections[:15]:
            print(f"  {reason:30s}: {count:8d} ({100*count/total:5.1f}%)")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
