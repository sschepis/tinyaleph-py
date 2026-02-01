#!/usr/bin/env python3
"""
CSV to Parquet Converter

Reads a CSV file, processes it, and outputs a Parquet file.

Usage:
    python csv_to_parquet.py <input.csv> [output.parquet]

If output path is not specified, it will use the input filename with .parquet extension.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def csv_to_parquet(
    input_path: str,
    output_path: str | None = None,
    compression: str = "snappy",
) -> Path:
    """
    Read a CSV file and write it as a Parquet file.

    Args:
        input_path: Path to the input CSV file
        output_path: Path to the output Parquet file (optional)
        compression: Compression algorithm for Parquet ('snappy', 'gzip', 'brotli', 'zstd', None)

    Returns:
        Path to the created Parquet file
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_file.suffix.lower() == ".csv":
        print(f"Warning: Input file does not have .csv extension: {input_path}")

    # Determine output path
    if output_path is None:
        output_file = input_file.with_suffix(".parquet")
    else:
        output_file = Path(output_path)

    print(f"Reading CSV: {input_file}")

    # Read CSV with automatic type inference
    df = pd.read_csv(
        input_file,
        low_memory=False,  # Better type inference for large files
        on_bad_lines="warn",  # Skip malformed lines with warning instead of failing
    )

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Display column info
    print("\nColumn types:")
    for col, dtype in df.dtypes.items():
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        print(f"  {col}: {dtype} ({null_count:,} nulls, {null_pct:.1f}%)")

    # Write to Parquet
    print(f"\nWriting Parquet: {output_file}")
    df.to_parquet(
        output_file,
        engine="pyarrow",
        compression=compression,
        index=False,
    )

    # Report output file size
    output_size = output_file.stat().st_size
    input_size = input_file.stat().st_size
    compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0

    print(f"  Output size: {output_size / 1024**2:.2f} MB")
    print(f"  Compression: {compression_ratio:.1f}% reduction")

    return output_file


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python csv_to_parquet.py data.csv
    python csv_to_parquet.py data.csv output.parquet
    python csv_to_parquet.py data.csv --compression gzip
        """,
    )
    parser.add_argument(
        "input",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to output Parquet file (optional, defaults to input name with .parquet extension)",
    )
    parser.add_argument(
        "--compression",
        "-c",
        choices=["snappy", "gzip", "brotli", "zstd", "none"],
        default="snappy",
        help="Compression algorithm (default: snappy)",
    )

    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression

    try:
        output_path = csv_to_parquet(
            args.input,
            args.output,
            compression=compression,
        )
        print(f"\nSuccess! Created: {output_path}")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty", file=sys.stderr)
        return 1
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
