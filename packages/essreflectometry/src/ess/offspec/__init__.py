# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline

from ..reflectometry import providers as reflectometry_providers
from . import conversions, data, load, maskings, normalization, types, workflow

providers = (
    *reflectometry_providers,
    *load.providers,
    *conversions.providers,
    *maskings.providers,
    *workflow.providers,
    *normalization.providers,
)


def OffspecWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Estia instrument.
    """
    return sciline.Pipeline(providers=providers)


__all__ = (
    "conversions",
    "data",
    "load",
    "maskings",
    "types",
    "workflow",
)
