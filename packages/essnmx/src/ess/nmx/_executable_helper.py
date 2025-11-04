# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
import logging
import sys

from .types import Compression


def build_reduction_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command line arguments for the NMX reduction. "
        "It assumes 14 Hz pulse speed."
    )
    input_arg_group = parser.add_argument_group("Input Options")
    input_arg_group.add_argument(
        "--input_file", type=str, help="Path to the input file", required=True
    )
    input_arg_group.add_argument(
        "--nbins",
        type=int,
        default=50,
        help="Number of TOF bins",
    )
    input_arg_group.add_argument(
        "--detector_ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Detector indices to process",
    )

    output_arg_group = parser.add_argument_group("Output Options")
    output_arg_group.add_argument(
        "--output_file",
        type=str,
        default="scipp_output.h5",
        help="Path to the output file",
    )
    output_arg_group.add_argument(
        "--compression",
        type=str,
        default=Compression.BITSHUFFLE_LZ4.name,
        choices=[compression_key.name for compression_key in Compression],
        help="Compress option of reduced output file. Default: BITSHUFFLE_LZ4",
    )
    output_arg_group.add_argument(
        "--verbose", "-v", action="store_true", help="Increase output verbosity"
    )

    return parser


def build_logger(args: argparse.Namespace) -> logging.Logger:
    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
