# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess import sans
from ess.sans import types as st
from scipp.testing import assert_allclose, assert_identical

from ess.reduce import unwrap
from ess.reduce.nexus.types import DiskChoppers, NeXusName, Position
from ess.reduce.unwrap.types import KeepEventTimeOffset


@pytest.fixture(params=['events', 'histogram'])
def workflow(request):
    wf = sans.SansWorkflow(wavelength_from='analytical')
    wf[unwrap.FrameUnwrapBackend] = unwrap.FrameUnwrapBackend.scipy
    wf[KeepEventTimeOffset] = False
    wf[unwrap.LookupTableRelativeErrorThreshold] = {'monitor': np.inf}
    wf[st.NonBackgroundWavelengthRange] = None
    wf[st.UncertaintyBroadcastMode] = st.UncertaintyBroadcastMode.drop
    wf[st.DirectBeam] = None
    if request.param == 'events':
        events = sc.DataArray(
            sc.ones(dims=['event'], shape=[4], unit='counts', with_variances=True),
            coords={
                'event_time_offset': sc.array(
                    dims=['event'], values=[15.0, 15.0, 30.0, 45.0], unit='ms'
                )
            },
        )
        raw = sc.DataArray(
            sc.bins(data=events, dim='event', begin=sc.index(0), end=sc.index(4))
        )
    else:
        raw = sc.DataArray(
            sc.array(
                dims=['frame_time'],
                values=[2.0, 20.0, 30.0, 4.0],
                variances=[2.0, 20.0, 30.0, 4.0],
                unit='counts',
            ),
            coords={'frame_time': sc.linspace('frame_time', 10.0, 50.0, 5, unit='ms')},
        )
    for run in (st.SampleRun, st.EmptyBeamRun):
        wf[DiskChoppers[run]] = {}
        wf[Position[snx.NXsource, run]] = sc.vector([0.0, 0.0, 0.0], unit='m')
        for monitor in (st.Incident, st.Transmission):
            wf[NeXusName[monitor]] = 'monitor'
            wf[unwrap.MonitorLtotal[run, monitor]] = sc.scalar(75.0, unit='m')
            wf[st.RawMonitor[run, monitor]] = raw
    return wf


@pytest.mark.parametrize(
    'bins_type', [st.WavelengthBins, st.NormalizationWavelengthBins]
)
def test_sans_preserves_native_monitor_binning_by_default(workflow, bins_type):
    edges = sc.linspace('wavelength', 0.0, 4.0, 3, unit='angstrom')
    workflow[bins_type] = edges

    for run in (st.SampleRun, st.EmptyBeamRun):
        for monitor in (st.Incident, st.Transmission):
            raw = workflow.compute(st.RawMonitor[run, monitor])
            converted = workflow.compute(st.WavelengthMonitor[run, monitor])
            assert converted.is_binned == raw.is_binned
            if converted.is_binned:
                assert_identical(converted.bins.sum().data, raw.bins.sum().data)
                expected = converted.hist(wavelength=edges)
            else:
                assert_allclose(converted.sum().data, raw.sum().data)
                expected = converted.rebin(wavelength=edges)
            result = workflow.compute(st.CorrectedMonitor[run, monitor])
            assert_identical(result.coords['wavelength'], edges)
            assert_allclose(result.data, expected.data)

    direct_beam = workflow.compute(st.CleanDirectBeam)
    assert_identical(direct_beam.coords['wavelength'], edges)
    bands = workflow.compute(st.ProcessedWavelengthBands)
    assert_identical(bands.flatten(to='wavelength'), edges['wavelength', [0, 2]])


def test_sans_respects_explicit_component_binning(workflow):
    target = st.WavelengthMonitor[st.SampleRun, st.Incident]
    original = workflow.compute(target)
    transmission = workflow.compute(st.WavelengthMonitor[st.SampleRun, st.Transmission])
    for nbins in (2, 4):
        edges = sc.linspace('wavelength', 0.0, 4.0, nbins + 1, unit='angstrom')
        workflow[unwrap.WavelengthBins[st.Incident]] = edges
        expected = (
            original.hist(wavelength=edges)
            if original.is_binned
            else original.rebin(wavelength=edges)
        )
        assert_identical(workflow.compute(target), expected)
        assert_identical(
            workflow.compute(st.WavelengthMonitor[st.SampleRun, st.Transmission]),
            transmission,
        )


def test_sans_estimates_background_before_rebinning():
    workflow = sans.SansWorkflow()
    monitor = sc.DataArray(
        sc.array(
            dims=['wavelength'],
            values=[2.0, 20.0, 30.0, 4.0],
            variances=[2.0, 20.0, 30.0, 4.0],
            unit='counts',
        ),
        coords={'wavelength': sc.linspace('wavelength', 1.0, 5.0, 5, unit='angstrom')},
    )
    workflow[st.WavelengthMonitor[st.SampleRun, st.Incident]] = monitor
    workflow[st.UncertaintyBroadcastMode] = st.UncertaintyBroadcastMode.drop
    native_edges = monitor.coords['wavelength']
    workflow[st.NonBackgroundWavelengthRange] = sc.concat(
        [
            (native_edges[1] + native_edges[2]) / 2,
            (native_edges[2] + native_edges[3]) / 2,
        ],
        'wavelength',
    )
    edges = native_edges['wavelength', [0, 2, 4]]
    workflow[st.NormalizationWavelengthBins] = edges

    result = workflow.compute(st.CorrectedMonitor[st.SampleRun, st.Incident])

    # Native background bins contain 2 and 4 counts, giving a mean of 3.
    assert_allclose(
        result.data,
        sc.array(
            dims=['wavelength'],
            values=[19.0, 31.0],
            variances=[22.0, 34.0],
            unit='counts',
        ),
    )
