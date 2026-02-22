"""Tests for the CLI entry point."""

import sys
from unittest.mock import patch

from asl_qc.cli import _build_parser


def test_parser_required_args():
    parser = _build_parser()
    # Should fail without --input
    try:
        parser.parse_args([])
        assert False, "Should fail without --input"
    except SystemExit:
        pass


def test_parser_with_args():
    parser = _build_parser()
    args = parser.parse_args([
        "--input", "/path/to/bids",
        "--output-dir", "/tmp/qc",
        "--verbose",
        "--skip-ml",
    ])
    assert args.input == "/path/to/bids"
    assert args.output_dir == "/tmp/qc"
    assert args.verbose
    assert args.skip_ml


def test_parser_defaults():
    parser = _build_parser()
    args = parser.parse_args(["--input", "/data"])
    assert args.output_dir == "./qc_output"
    assert args.config is None
    assert not args.verbose
    assert not args.skip_ml
