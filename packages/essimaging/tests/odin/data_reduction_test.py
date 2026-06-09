# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import ess.odin.data  # noqa: F401
import pytest
import sciline as sl
from ess import odin
from ess.odin.beamline import choppers as odin_choppers

from ess.imaging.types import (
    Filename,
    LookupTableFilename,
    NeXusDetectorName,
    NXsource,
    OpenBeamRun,
    Position,
    RawDetector,
    SampleRun,
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
