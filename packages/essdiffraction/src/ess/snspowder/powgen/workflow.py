# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import sciline

from ess.powder import providers as powder_providers
from ess.powder.types import NeXusDetectorName

from . import beamline


def default_parameters() -> dict:
    return {NeXusDetectorName: "powgen_detector"}


def PowgenWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Powgen SNS instrument.
    """
    # The package does not depend on pooch which is needed for the tutorial
    # data. Delay import until workflow is actually used.
    from . import data

    return sciline.Pipeline(
        providers=powder_providers + beamline.providers + data.providers,
        params=default_parameters(),
    )


__all__ = ['PowgenWorkflow', 'default_parameters']
