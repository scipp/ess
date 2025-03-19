# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for DREAM.

Notes on the detector dimensions (2024-05-22):

See https://confluence.ess.eu/pages/viewpage.action?pageId=462000005
(link requires ESS login) and the ICD DREAM interface specification for details.

- The high-resolution and SANS detectors have a very odd numbering scheme.
  The scheme attempts to follows some sort of physical ordering in space (x,y,z),
  but it is not possible to reshape the data into all the logical dimensions.
"""

import sciline

from ess.powder.types import BunkerMonitor, CaveMonitor
from ess.reduce.nexus.types import DetectorBankSizes, NeXusName
from ess.reduce.nexus.workflow import GenericNeXusWorkflow

DETECTOR_BANK_SIZES = {
    "endcap_backward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 11,
        "segment": 28,
        "counter": 2,
    },
    "endcap_forward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 5,
        "segment": 28,
        "counter": 2,
    },
    "mantle_detector": {
        "wire": 32,
        "module": 5,
        "segment": 6,
        "strip": 256,
        "counter": 2,
    },
    "high_resolution_detector": {"strip": 32, "other": -1},
    "sans_detector": {"strip": 32, "other": -1},
}


def LoadNeXusWorkflow() -> sciline.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    wf = GenericNeXusWorkflow()
    wf[DetectorBankSizes] = DETECTOR_BANK_SIZES
    wf[NeXusName[BunkerMonitor]] = "monitor_bunker"
    wf[NeXusName[CaveMonitor]] = "monitor_cave"
    return wf
