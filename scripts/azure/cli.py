"""
Shared CLI utilities for Azure ML scripts.
"""

import argparse


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments used by both upload and download scripts."""
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset key (e.g., beir_quora, lotte_pooled)",
    )
    parser.add_argument(
        "--kind", "-k",
        type=str,
        choices=["data", "index"],
        help="Asset type: 'data' (raw dataset) or 'index' (WARP index)",
    )
    parser.add_argument(
        "--version", "-v",
        type=str,
        default=None,
        help="Asset version (default: from config, usually '1')",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite/re-download even if asset exists",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_datasets",
        help="List all available datasets and exit",
    )


def validate_common_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate that required arguments are provided."""
    if not args.dataset:
        parser.error("--dataset is required (use --list to see available datasets)")
    if not args.kind:
        parser.error("--kind is required (must be 'data' or 'index')")
