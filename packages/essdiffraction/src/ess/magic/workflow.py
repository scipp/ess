# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import sciline as sl

from ess.reduce.nexus.types import DetectorBankSizes

from . import mcstas
from .beamline import DETECTOR_BANK_SIZES
from .types import DetectorBank, RunType, SampleRun

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


def MagicMcStasWorkflow() -> sl.Pipeline:
    """Workflow for MAGIC McStas events and geometry, with detector bank A by default.

    Provides ``RawMcStasEvents[RunType]`` and ``EmptyDetector[RunType]`` independently.
    Event assignment and reduction providers are not yet included.
    """
    return sl.Pipeline(
        providers=(mcstas.load_mcstas_events, mcstas.mcstas_detector_geometry),
        constraints={RunType: [SampleRun]},
        params={DetectorBank: DetectorBank.detector_a},
    )
