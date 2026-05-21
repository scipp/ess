# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import sciline as sl

from ess.reduce.nexus.types import DetectorBankSizes

from .beamline import DETECTOR_BANK_SIZES

default_parameters = {
    DetectorBankSizes: DETECTOR_BANK_SIZES,
}


def MagicWorkflow() -> sl.Pipeline:
    """Workflow for the MAGIC single-crystal magnetism diffractometer.

    This is currently a placeholder skeleton: it registers default beamline
    parameters but provides no reduction providers yet. Single-crystal and
    polarisation-analysis providers will be added as the MAGIC reduction
    pipeline is developed.
    """
    return sl.Pipeline(providers=(), params=default_parameters)
