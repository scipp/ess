# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
from ess import loki
from ess.sans.types import (
    BeamCenter,
    Filename,
    NeXusDetectorName,
    RawDetector,
    SampleRun,
)


@pytest.fixture(scope='session')
def loki_data():
    wf = loki.LokiWorkflow()
    wf[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    wf[Filename[SampleRun]] = loki.data.loki_coda_file()

    data = sc.DataGroup()
    for i in range(9):
        key = f"loki_detector_{i}"
        wf[NeXusDetectorName] = key
        data[key] = wf.compute(RawDetector[SampleRun])

    return data


@pytest.fixture(scope='session')
def histogrammed_loki_data(loki_data):
    return loki_data.hist()
