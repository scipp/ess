# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline
from ess.sans import providers as sans_providers

from .data import load_tutorial_direct_beam, load_tutorial_run
from .general import default_parameters
from .io import read_xml_detector_masking
from .mantidio import providers as mantid_providers


def set_mantid_log_level(level: int = 3):
    try:
        from mantid import ConfigService

        cfg = ConfigService.Instance()
        cfg.setLogLevel(level)  # Silence verbose load via Mantid
    except ImportError:
        pass


def ZoomWorkflow() -> sciline.Pipeline:
    """Create Zoom workflow with default parameters."""
    from . import providers as isis_providers

    set_mantid_log_level()

    params = default_parameters()
    zoom_providers = sans_providers + isis_providers + mantid_providers
    workflow = sciline.Pipeline(providers=zoom_providers, params=params)
    workflow.insert(read_xml_detector_masking)
    return workflow


def ZoomTutorialWorkflow() -> sciline.Pipeline:
    """
    Create Zoom tutorial workflow.

    Equivalent to :func:`ZoomWorkflow`, but with loaders for tutorial data instead
    of Mantid-based loaders.
    """
    workflow = ZoomWorkflow()
    workflow.insert(load_tutorial_run)
    workflow.insert(load_tutorial_direct_beam)
    return workflow
