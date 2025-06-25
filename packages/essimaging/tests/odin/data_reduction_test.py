# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
import sciline as sl

import ess.odin.data  # noqa: F401
from ess import odin
from ess.imaging.types import (
    DetectorData,
    DetectorTofData,
    DetectorWavelengthData,
    DiskChoppers,
    EmptyBeamRun,
    Filename,
    NeXusDetectorName,
    SampleRun,
    TimeOfFlightLookupTable,
)
from ess.reduce import time_of_flight


@pytest.fixture(scope="module")
def workflow() -> sl.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    wf = odin.OdinWorkflow(tof_lut_provider=time_of_flight.TofLutProvider.TOF)
    wf[Filename[SampleRun]] = odin.data.iron_simulation_sample_small()
    wf[Filename[EmptyBeamRun]] = odin.data.iron_simulation_ob_small()
    wf[NeXusDetectorName] = "event_mode_detectors/timepix3"
    wf[DiskChoppers[SampleRun]] = odin.beamline.choppers(
        source_position=wf.compute(DetectorData[SampleRun]).coords["source_position"]
    )
    # Cache the lookup table
    wf[TimeOfFlightLookupTable] = wf.compute(TimeOfFlightLookupTable)
    return wf


@pytest.mark.parametrize("run_type", [SampleRun, EmptyBeamRun])
def test_can_load_detector_data(workflow, run_type):
    da = workflow.compute(DetectorData[run_type])
    assert {
        "detector_number",
        "gravity",
        "position",
        "sample_position",
        "source_position",
        "x_pixel_offset",
        "y_pixel_offset",
    }.issubset(set(da.coords.keys()))
    assert da.bins is not None
    assert "event_time_offset" in da.bins.coords
    assert "event_time_zero" in da.bins.coords


@pytest.mark.parametrize("run_type", [SampleRun, EmptyBeamRun])
def test_can_compute_time_of_flight(workflow, run_type):
    da = workflow.compute(DetectorTofData[run_type])

    assert "tof" in da.bins.coords


@pytest.mark.parametrize("run_type", [SampleRun, EmptyBeamRun])
def test_can_compute_wavelength(workflow, run_type):
    da = workflow.compute(DetectorWavelengthData[run_type])

    assert "wavelength" in da.bins.coords
