# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from ess.reduce.nexus.types import (
    DiskChoppers,
    EmptyMonitor,
    NeXusName,
    Position,
    RawMonitor,
)
from ess.reduce.unwrap import fakes
from ess.reduce.unwrap.types import LookupTableRelativeErrorThreshold, WavelengthMonitor
from scippnexus import NXsource

from ess import estia, freia
from ess.estia.types import BeamMonitor
from ess.freia.types import (
    NormalizationMonitor,
    PSCMonitor,
    ShutterMonitor,
    WBC1Monitor,
    WBC2Monitor,
    WBC3Monitor,
)
from ess.reflectometry.types import SampleRun


@pytest.mark.parametrize(
    ('workflow_factory', 'monitor_type'),
    [
        (estia.EstiaWorkflow, BeamMonitor),
        (freia.FreiaWorkflow, WBC1Monitor),
        (freia.FreiaWorkflow, PSCMonitor),
        (freia.FreiaWorkflow, WBC2Monitor),
        (freia.FreiaWorkflow, WBC3Monitor),
        (freia.FreiaWorkflow, ShutterMonitor),
        (freia.FreiaWorkflow, NormalizationMonitor),
    ],
)
def test_instrument_workflow_computes_monitor_wavelength(
    workflow_factory, monitor_type
):
    wf = workflow_factory(wavelength_from='analytical')
    wf[NeXusName[monitor_type]] = 'monitor'
    wf[LookupTableRelativeErrorThreshold] = {'monitor': float('inf')}
    wf[RawMonitor[SampleRun, monitor_type]] = sc.DataArray(
        sc.ones(sizes={'frame_time': 30}, unit='counts'),
        coords={
            'frame_time': sc.linspace('frame_time', 0.0, 71.0, 31, unit='ms'),
            'position': sc.vector([0.0, 0.0, 75.0], unit='m'),
        },
    )
    wf[EmptyMonitor[SampleRun, monitor_type]] = sc.DataArray(
        sc.scalar(0.0),
        coords={'position': sc.vector([0.0, 0.0, 75.0], unit='m')},
    )
    wf[Position[NXsource, SampleRun]] = fakes.source_position()
    wf[DiskChoppers[SampleRun]] = fakes.psc_choppers()

    result = wf.compute(WavelengthMonitor[SampleRun, monitor_type])

    wavelength = result.coords['wavelength']
    assert wavelength.unit == sc.Unit('angstrom')
    assert sc.isfinite(wavelength).any().value
