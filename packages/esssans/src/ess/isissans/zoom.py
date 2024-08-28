# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline
from ess.reduce.nexus.generic_workflow import GenericNeXusWorkflow
from ess.reduce.workflow import register_workflow

from ess.sans import providers as sans_providers
from ess.sans.io import read_xml_detector_masking
from ess.sans.parameters import typical_outputs

from .general import default_parameters
from .io import load_tutorial_direct_beam, load_tutorial_run
from .mantidio import providers as mantid_providers


def set_mantid_log_level(level: int = 3):
    try:
        from mantid import ConfigService

        cfg = ConfigService.Instance()
        cfg.setLogLevel(level)  # Silence verbose load via Mantid
    except ImportError:
        pass


@register_workflow
def ZoomWorkflow() -> sciline.Pipeline:
    """Create Zoom workflow with default parameters."""
    from . import providers as isis_providers

    set_mantid_log_level()

    # Note that the actual NeXus loading in this workflow will not be used for the
    # ISIS files, the providers inserted below will replace those steps.
    workflow = GenericNeXusWorkflow()
    for provider in sans_providers + isis_providers + mantid_providers:
        workflow.insert(provider)
    for key, param in default_parameters().items():
        workflow[key] = param
    workflow.insert(read_xml_detector_masking)
    workflow.typical_outputs = typical_outputs
    return workflow


@register_workflow
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
