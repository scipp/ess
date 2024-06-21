# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Functions and classes for the POWGEN instrument.

Note that this module is temporary and will be removed in favor of
the ``dream`` module when that is available.
"""

import sciline

from ess.powder import providers as powder_providers
from ess.powder.types import NeXusDetectorName
from . import beamline, data
from .instrument_view import instrument_view

providers = (
    *beamline.providers,
    *data.providers,
)
"""Sciline Providers for POWGEN-specific functionality."""


def default_parameters() -> dict:
    return {NeXusDetectorName: "powgen_detector"}


def PowgenWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Powgen SNS instrument.
    """
    return sciline.Pipeline(
        providers=powder_providers + providers, params=default_parameters()
    )


__all__ = [
    'PowgenWorkflow',
    'beamline',
    'data',
    'default_parameters',
    'instrument_view',
]
