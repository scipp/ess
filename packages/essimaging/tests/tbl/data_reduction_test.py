# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
import sciline as sl

import ess.tbl.data  # noqa: F401
from ess import tbl
from ess.reduce import time_of_flight
from ess.tbl.types import (
    DetectorData,
    DetectorTofData,
    DetectorWavelengthData,
    DiskChoppers,
    Filename,
    NeXusDetectorName,
    SampleRun,
    TimeOfFlightLookupTable,
)


@pytest.fixture(scope="module")
def workflow() -> sl.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    wf = tbl.TblWorkflow(tof_lut_provider=time_of_flight.TofLutProvider.TOF)
    wf[Filename[SampleRun]] = tbl.data.tutorial_sample_data()
    wf[DiskChoppers[SampleRun]] = {}
    # Cache the lookup table
    wf[TimeOfFlightLookupTable] = wf.compute(TimeOfFlightLookupTable)
    return wf


@pytest.mark.parametrize(
    "bank_name", ["ngem_detector", "he3_detector_bank0", "he3_detector_bank1"]
)
def test_can_load_detector_data(workflow, bank_name):
    workflow[NeXusDetectorName] = bank_name
    da = workflow.compute(DetectorData[SampleRun])

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


@pytest.mark.parametrize(
    "bank_name", ["ngem_detector", "he3_detector_bank0", "he3_detector_bank1"]
)
def test_can_compute_time_of_flight(workflow, bank_name):
    workflow[NeXusDetectorName] = bank_name
    da = workflow.compute(DetectorTofData[SampleRun])

    assert "tof" in da.bins.coords


@pytest.mark.parametrize(
    "bank_name", ["ngem_detector", "he3_detector_bank0", "he3_detector_bank1"]
)
def test_can_compute_wavelength(workflow, bank_name):
    workflow[NeXusDetectorName] = bank_name
    da = workflow.compute(DetectorWavelengthData[SampleRun])

    assert "wavelength" in da.bins.coords
