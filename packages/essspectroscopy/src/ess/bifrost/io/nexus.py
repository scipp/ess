# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus input/output for BIFROST."""

import sciline

from ess.reduce.nexus.generic_workflow import GenericNeXusWorkflow


def LoadNeXusWorkflow() -> sciline.Pipeline:
    """Workflow for loading BIFROST NeXus files."""
    workflow = GenericNeXusWorkflow()
    return workflow
