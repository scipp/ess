# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from ess import freia
from ess.reflectometry.types import CorrectionsToApply, NeXusDetectorName


def test_freia_workflow_uses_freia_defaults():
    params = freia.workflow.default_parameters()

    assert params[NeXusDetectorName] == "multiblade_detector"
    assert params[CorrectionsToApply] == freia.corrections.default_corrections
