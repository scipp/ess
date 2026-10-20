# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import scippneutron as scn
from ess import loki
from ess.loki import diagnostics
from ess.loki.diagnostics import LokiBankViewer


def test_diagnostics_available_at_top_level():
    assert loki.LokiBankViewer is LokiBankViewer
    assert loki.instrument_view is scn.instrument_view
    assert loki.InstrumentView is diagnostics.InstrumentView
    assert {'LokiBankViewer', 'instrument_view', 'InstrumentView'} <= set(loki.__all__)
