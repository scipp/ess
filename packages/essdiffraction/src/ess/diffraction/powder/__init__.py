# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Components for powder diffraction experiments.
"""
from .conversion import providers as conversion_providers
from .conversion import to_dspacing_with_calibration
from .correction import merge_calibration
from .correction import providers as correction_providers

providers = (*conversion_providers, *correction_providers)
"""Sciline providers for powder diffraction."""
del conversion_providers
del correction_providers

__all__ = ['merge_calibration', 'to_dspacing_with_calibration']
