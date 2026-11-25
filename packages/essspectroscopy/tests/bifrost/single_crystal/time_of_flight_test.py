# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
import scipp as sc
import scipp.testing
import scippnexus as snx
from ess.bifrost.data import coda_test_run_2026_11
from ess.bifrost.single_crystal import BifrostBraggPeakMonitorWorkflow
from ess.spectroscopy.types import (
    ElasticMonitor,
    Filename,
    NeXusData,
    SampleRun,
    TransformationOverrides,
)

from ess.reduce.unwrap.types import DetectorLtotal, LookupTable

_TANK_ANGLE = '/entry/instrument/detector_tank_angle/value'
_MONITOR_Z = '/entry/instrument/elastic_monitor/transformations/elastic_monitor_t0_z'


@pytest.fixture
def workflow() -> sciline.Pipeline:
    workflow = BifrostBraggPeakMonitorWorkflow()
    workflow[Filename[SampleRun]] = coda_test_run_2026_11()
    return workflow


def _time_series(values: list[float], unit: str) -> sc.DataArray:
    times = sc.datetimes(
        dims=['time'],
        values=[f'2026-11-18T10:0{i}:00' for i in range(len(values))],
        unit='ns',
    )
    return sc.DataArray(
        sc.array(dims=['time'], values=values, unit=unit), coords={'time': times}
    )


def _no_monitor_events() -> NeXusData[ElasticMonitor, SampleRun]:
    raise AssertionError('the lookup table must not depend on the monitor events')


def test_lookup_table_does_not_depend_on_monitor_events(
    workflow: sciline.Pipeline,
) -> None:
    # When streaming, the events are the dynamic input. The table must not depend on
    # them, or it is rebuilt for every chunk.
    workflow.insert(_no_monitor_events)
    table = workflow.compute(LookupTable[SampleRun, snx.NXdetector])
    assert table.array.sizes['distance'] > 0


def test_ltotal_is_scalar_for_time_dependent_tank_angle(
    workflow: sciline.Pipeline,
) -> None:
    static = workflow.compute(DetectorLtotal[SampleRun])
    # The tank rotates about the vertical axis through the sample.
    workflow[TransformationOverrides[ElasticMonitor, SampleRun]] = {
        _TANK_ANGLE: _time_series([-90.0, 0.0, 90.0], unit='deg')
    }
    ltotal = workflow.compute(DetectorLtotal[SampleRun])
    sc.testing.assert_allclose(ltotal, static)


def test_ltotal_raises_if_flight_path_changes_over_time(
    workflow: sciline.Pipeline,
) -> None:
    workflow[TransformationOverrides[ElasticMonitor, SampleRun]] = {
        _MONITOR_Z: _time_series([1.0, 1.1], unit='m')
    }
    with pytest.raises(ValueError, match='changes with the tank angle'):
        workflow.compute(DetectorLtotal[SampleRun])
