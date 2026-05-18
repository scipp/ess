# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import sciline as sl
from ess.magic import MagicWorkflow, default_parameters

from ess.reduce.nexus.types import DetectorBankSizes


def test_magic_workflow_can_be_constructed():
    wf = MagicWorkflow()
    assert isinstance(wf, sl.Pipeline)


def test_default_parameters_include_detector_bank_sizes():
    sizes = default_parameters[DetectorBankSizes]
    assert 'detector_a' in sizes
    assert 'detector_b' in sizes
