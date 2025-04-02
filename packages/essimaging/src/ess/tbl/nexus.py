# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for DREAM.

Notes on the detector dimensions (2024-05-22):

See https://confluence.esss.lu.se/pages/viewpage.action?pageId=462000005
and the ICD DREAM interface specification for details.

- The high-resolution and SANS detectors have a very odd numbering scheme.
  The scheme attempts to follows some sort of physical ordering in space (x,y,z),
  but it is not possible to reshape the data into all the logical dimensions.
"""

import sciline

from ess.reduce.nexus.types import Monitor1, SampleRun
from ess.reduce.nexus.workflow import GenericNeXusWorkflow


def LoadNeXusWorkflow() -> sciline.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[Monitor1])
    return wf
