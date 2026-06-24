# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from pathlib import Path

import ess.odin.data  # noqa: F401
import pytest
import sciline as sl
import scipp as sc
from ess import odin
from ess.odin.beamline import choppers as odin_choppers

from ess.imaging import tools
from ess.imaging.types import (
    Filename,
    LookupTableFilename,
    MaskingRules,
    NeXusDetectorName,
    NXsource,
    OpenBeamRun,
    Position,
    RawDetector,
    SampleRun,
    TofDetector,
    WavelengthDetector,
)
from ess.reduce import unwrap


def _make_workflow(wavelength_from: unwrap.WavelengthLutMode) -> sl.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    wf = odin.OdinBraggEdgeWorkflow(wavelength_from=wavelength_from)
    wf[Filename[SampleRun]] = odin.data.iron_simulation_sample_small()
    wf[Filename[OpenBeamRun]] = odin.data.iron_simulation_ob_small()
    wf[NeXusDetectorName] = "event_mode_detectors/timepix3"
    if wavelength_from == "file":
        # Shortcut to set LookupTableFilename for both SampleRun and OpenBeamRun at once
        wf[LookupTableFilename] = odin.data.odin_wavelength_lookup_table()
    else:
        disk_choppers = odin_choppers(
            source_position=wf.compute(Position[NXsource, SampleRun])
        )
        # Shortcut to set DiskChoppers for both [SampleRun, NXdetector] and
        # [OpenBeamRun, NXdetector] at once
        wf[unwrap.DiskChoppers] = disk_choppers
    return wf


@pytest.mark.parametrize("run_type", [SampleRun, OpenBeamRun])
def test_can_load_detector_data(run_type):
    wf = _make_workflow("file")
    da = wf.compute(RawDetector[run_type])
    assert {
        "detector_number",
        "position",
        "x_pixel_offset",
        "y_pixel_offset",
    }.issubset(set(da.coords.keys()))
    assert da.bins is not None
    assert "event_time_offset" in da.bins.coords
    assert "event_time_zero" in da.bins.coords


@pytest.mark.parametrize("run_type", [SampleRun, OpenBeamRun])
@pytest.mark.parametrize("wavelength_mode", ["file", "analytical"])
def test_can_compute_wavelength(run_type, wavelength_mode):
    wf = _make_workflow(wavelength_mode)
    da = wf.compute(WavelengthDetector[run_type])

    assert "wavelength" in da.bins.coords


@pytest.mark.parametrize("run_type", [SampleRun, OpenBeamRun])
@pytest.mark.parametrize("wavelength_mode", ["file", "analytical"])
def test_can_compute_tof(run_type, wavelength_mode):
    wf = _make_workflow(wavelength_mode)
    wf[MaskingRules] = {}
    da = wf.compute(TofDetector[run_type])

    assert "tof" in da.bins.coords


def test_publish_reduced_scitiff(output_folder: Path):
    wf = _make_workflow("analytical")
    wf[MaskingRules] = {}
    new_sizes = {'dim_0': 64, 'dim_1': 64}
    tbins = sc.linspace('tof', 1.3e4, 1.5e5, 257, unit='us')

    sample = wf.compute(TofDetector[SampleRun]).drop_coords('detector_number')
    res_sample = tools.resample(sample, sizes=new_sizes)
    num = res_sample.hist(tof=tbins)

    openbeam = wf.compute(TofDetector[OpenBeamRun]).drop_coords('detector_number')
    res_openbeam = tools.resample(openbeam, sizes=new_sizes)
    den = res_openbeam.hist(tof=tbins)

    normed = num / den

    to_scitiff = (
        normed.assign_coords(
            x=normed.coords['x_pixel_offset'],
            y=normed.coords['y_pixel_offset'],
            t=sc.midpoints(normed.coords['tof']),
        )
        .rename_dims(dim_0='y', dim_1='x', tof='t')
        .drop_coords(['position', 'tof', 'x_pixel_offset', 'y_pixel_offset'])
    )

    from scitiff.io import save_scitiff

    save_scitiff(
        to_scitiff, output_folder / 'bragg_edge_iron_normalized_16x16x256.tiff'
    )
