# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
import sciline as sl
import scipp as sc

import ess.tbl.data  # noqa: F401
from ess import tbl
from ess.imaging.types import (
    CountsWavelength,
    DetectorData,
    DetectorTofData,
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
    wf = tbl.TblWorkflow()
    wf[Filename[SampleRun]] = tbl.data.tutorial_sample_data()

    lut_wf = time_of_flight.TofLutWorkflow()
    lut_wf[time_of_flight.DiskChoppers] = {}
    lut_wf[time_of_flight.SourcePosition] = sc.vector([0, 0, 0], unit="m")
    lut_wf[time_of_flight.NumberOfSimulatedNeutrons] = 200_000
    lut_wf[time_of_flight.SimulationSeed] = 333
    lut_wf[time_of_flight.PulseStride] = 1
    lut_wf[time_of_flight.LtotalRange] = (
        sc.scalar(25.0, unit="m"),
        sc.scalar(35.0, unit="m"),
    )
    # Cache the lookup table
    wf[TimeOfFlightLookupTable] = lut_wf.compute(TimeOfFlightLookupTable)
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
    da = workflow.compute(CountsWavelength[SampleRun])

    assert "wavelength" in da.bins.coords
