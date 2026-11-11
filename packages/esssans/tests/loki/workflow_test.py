# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import dataclasses

import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess import loki
from ess.loki import LokiAtLarmorWorkflow
from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    DetectorMasks,
    EmptyDetector,
    Filename,
    IntensityQ,
    LookupTableFilename,
    NeXusDetectorName,
    PixelMaskFilenames,
    QBins,
    ReturnEvents,
    SampleRun,
    SolidAngle,
    UncertaintyBroadcastMode,
    WavelengthDetector,
)
from sciline import UnsatisfiedRequirement

from ess.reduce import workflow
from ess.reduce.masking import write_pixel_masks
from ess.reduce.nexus.types import DiskChoppers, Position
from ess.reduce.unwrap import LookupTable, LtotalRange, SourceBounds


def test_sans_workflow_registers_subclasses():
    # Because it was imported
    assert LokiAtLarmorWorkflow in workflow.workflow_registry
    count = len(workflow.workflow_registry)

    @workflow.register_workflow
    class MyWorkflow: ...

    assert MyWorkflow in workflow.workflow_registry
    assert len(workflow.workflow_registry) == count + 1


def test_loki_larmor_workflow_parameters_returns_filtered_params():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, (IntensityQ[SampleRun],))
    assert Filename[SampleRun] in parameters
    assert Filename[BackgroundRun] not in parameters


def test_loki_larmor_workflow_parameters_returns_no_params_for_no_outputs():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, ())
    assert not parameters


def test_loki_larmor_workflow_parameters_with_param_returns_param():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, (ReturnEvents,))
    assert parameters.keys() == {ReturnEvents}


def test_beam_center_does_not_move_detector_positions(larmor_workflow):
    wf = larmor_workflow()
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    reference = wf.compute(EmptyDetector[SampleRun]).coords['position']
    wf[BeamCenter] = sc.vector([0.03, -0.02, 0.0], unit='m')
    position = wf.compute(EmptyDetector[SampleRun]).coords['position']
    assert sc.identical(position, reference)


def test_solid_angle_is_independent_of_beam_center(larmor_workflow):
    wf = larmor_workflow()
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    reference = wf.compute(SolidAngle[SampleRun])
    wf[BeamCenter] = sc.vector([0.03, -0.02, 0.0], unit='m')
    assert sc.identical(wf.compute(SolidAngle[SampleRun]), reference)


def test_loki_larmor_workflow_compute_with_single_pixel_mask(larmor_workflow):
    wf = larmor_workflow(no_masks=False)
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    wf[PixelMaskFilenames] = loki.data.loki_tutorial_mask_filenames()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')

    result = wf.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
    assert sc.identical(result.coords['Q'], wf.compute(QBins))
    assert result.sizes['Q'] == 100


@pytest.mark.parametrize("bank", list(range(9)))
def test_loki_workflow_needs_lookup_table(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    with pytest.raises(UnsatisfiedRequirement, match='LookupTableFilename'):
        wf.compute(WavelengthDetector[SampleRun])


@pytest.mark.parametrize("bank", list(range(9)))
def test_loki_workflow_can_compute_wavelength(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    wf[LookupTableFilename] = loki.data.loki_lookup_table_no_choppers()
    result = wf.compute(WavelengthDetector[SampleRun])
    assert 'wavelength' in result.bins.coords


@pytest.mark.parametrize("bank", list(range(9)))
def test_loki_workflow_can_compute_iofq(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    wf[LookupTableFilename] = loki.data.loki_lookup_table_no_choppers()

    result = wf.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
    assert sc.identical(result.coords['Q'], wf.compute(QBins))


def test_loki_analytical_lookup_table_matches_simulated_source_pulse():
    # Reference: table built from a tof simulation of the McStas ESS source, no
    # choppers, 1-10 Angstrom so that frames do not overlap. See scipp/ess#770.
    wf = loki.LokiWorkflow(wavelength_from='analytical')
    wf[DiskChoppers[SampleRun]] = {}
    wf[Position[snx.NXsource, SampleRun]] = sc.vector([0, 0, 0], unit='m')
    wf[LtotalRange[SampleRun, snx.NXdetector]] = (
        sc.scalar(28.2, unit='m'),
        sc.scalar(29.2, unit='m'),
    )
    wf[SourceBounds] = dataclasses.replace(
        wf.compute(SourceBounds),
        wavelength=(sc.scalar(1.0, unit='angstrom'), sc.scalar(10.0, unit='angstrom')),
    )
    table = wf.compute(LookupTable[SampleRun, snx.NXdetector]).array
    row = table['distance', np.argmin(np.abs(table.coords['distance'].values - 28.7))]
    toa = row.coords['event_time_offset'].to(unit='ms').values

    for t, mean, stddev in (
        (14.5, 1.750, 0.123),
        (29.0, 3.745, 0.122),
        (57.9, 7.736, 0.126),
    ):
        value = row['event_time_offset', np.argmin(np.abs(toa - t))].data
        assert abs(value.value - mean) < 0.01
        assert abs(sc.stddevs(value).value / stddev - 1) < 0.05


def test_loki_workflow_reads_bool_masks_per_bank(tmp_path):
    path = tmp_path / 'masks.h5'
    mask = sc.array(dims=['tube'], values=[True, False])
    write_pixel_masks(path, {'broken': {'loki_detector_1': mask}})
    wf = loki.LokiWorkflow()
    wf[NeXusDetectorName] = 'loki_detector_1'
    wf[EmptyDetector[SampleRun]] = sc.DataArray(sc.zeros(sizes={'tube': 2}))
    wf[PixelMaskFilenames] = [path]

    assert sc.identical(wf.compute(DetectorMasks)['broken'], mask)


def test_loki_workflow_raises_if_mask_file_has_unknown_bank(tmp_path):
    path = tmp_path / 'masks.h5'
    mask = sc.array(dims=['tube'], values=[True, False])
    write_pixel_masks(path, {'broken': {'loki_detector_99': mask}})
    wf = loki.LokiWorkflow()
    wf[NeXusDetectorName] = 'loki_detector_1'
    wf[EmptyDetector[SampleRun]] = sc.DataArray(sc.zeros(sizes={'tube': 2}))
    wf[PixelMaskFilenames] = [path]

    with pytest.raises(ValueError, match='loki_detector_99'):
        wf.compute(DetectorMasks)


@pytest.mark.parametrize(
    'dims',
    [
        ('pixel', 'straw', 'tube', 'layer'),
        ('layer', 'straw'),
        ('straw', 'layer'),
    ],
    ids=['full-reversed', 'skips-dims', 'skips-dims-transposed'],
)
def test_bank_mask_gives_same_iofq_as_equivalent_id_file(
    larmor_workflow, tmp_path, dims
):
    wf = larmor_workflow(no_masks=False)
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    ids = wf.compute(EmptyDetector[SampleRun]).coords['detector_number']

    shape = [ids.sizes[dim] for dim in dims]
    mask = sc.array(
        dims=list(dims), values=np.arange(np.prod(shape)).reshape(shape) % 5 == 0
    )
    expanded = sc.broadcast(mask, sizes=ids.sizes).transpose(ids.dims)
    masked_ids = sc.array(
        dims=['detector_id'],
        values=ids.values[expanded.values],
        unit=None,
        dtype='int32',
    )
    assert 0 < masked_ids.size < ids.size

    bank_file = tmp_path / 'bank.nxs'
    write_pixel_masks(bank_file, {'m': {'larmor_detector': mask}})
    id_file = tmp_path / 'ids.h5'
    sc.io.save_hdf5(masked_ids, id_file)

    wf[PixelMaskFilenames] = [bank_file]
    from_bank_mask = wf.compute(BackgroundSubtractedIofQ)
    wf[PixelMaskFilenames] = [id_file]
    from_ids = wf.compute(BackgroundSubtractedIofQ)
    assert sc.identical(from_bank_mask, from_ids)


def test_bool_mask_file_gives_same_masks_as_id_file(larmor_workflow, tmp_path):
    wf = larmor_workflow(no_masks=False)
    wf[PixelMaskFilenames] = loki.data.loki_tutorial_mask_filenames()
    from_ids = next(iter(wf.compute(DetectorMasks).values()))

    path = tmp_path / 'masks.h5'
    write_pixel_masks(path, {'m': {'larmor_detector': from_ids}})
    wf[PixelMaskFilenames] = [path]

    assert sc.identical(wf.compute(DetectorMasks)['m'], from_ids)
