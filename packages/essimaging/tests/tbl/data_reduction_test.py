# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
import sciline as sl
import scipp as sc
from ess.reduce import unwrap
from ess.reduce.nexus.types import AnyRun

import ess.tbl.data  # noqa: F401
from ess import tbl
from ess.imaging.types import (
    Filename,
    NeXusDetectorName,
    RawDetector,
    SampleRun,
    TimeOfFlightLookupTable,
    TimeOfFlightLookupTableFilename,
    TofDetector,
    WavelengthDetector,
)


@pytest.fixture(scope="module")
def tof_lookup_table() -> sl.Pipeline:
    """
    Compute tof lookup table on-the-fly.
    """

    lut_wf = unwrap.TofLookupTableWorkflow()
    lut_wf[unwrap.DiskChoppers[AnyRun]] = {}
    lut_wf[unwrap.SourcePosition] = sc.vector([0, 0, 0], unit="m")
    lut_wf[unwrap.NumberOfSimulatedNeutrons] = 200_000
    lut_wf[unwrap.SimulationSeed] = 333
    lut_wf[unwrap.PulseStride] = 1
    lut_wf[unwrap.LtotalRange] = (
        sc.scalar(25.0, unit="m"),
        sc.scalar(35.0, unit="m"),
    )
    return lut_wf.compute(TimeOfFlightLookupTable)


@pytest.fixture
def workflow() -> sl.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    wf = tbl.TblWorkflow()
    wf[Filename[SampleRun]] = tbl.data.tutorial_sample_data()
    wf[TimeOfFlightLookupTableFilename] = tbl.data.tbl_tof_lookup_table_no_choppers()
    wf[unwrap.LookupTableRelativeErrorThreshold] = {
        "ngem_detector": float('inf'),
        "he3_detector_bank0": float('inf'),
        "he3_detector_bank1": float('inf'),
    }
    return wf


@pytest.mark.parametrize(
    "bank_name", ["ngem_detector", "he3_detector_bank0", "he3_detector_bank1"]
)
def test_can_load_detector_data(workflow, bank_name):
    workflow[NeXusDetectorName] = bank_name
    da = workflow.compute(RawDetector[SampleRun])

    assert {
        "detector_number",
        "position",
        "x_pixel_offset",
        "y_pixel_offset",
    }.issubset(set(da.coords.keys()))
    assert da.bins is not None
    assert "event_time_offset" in da.bins.coords
    assert "event_time_zero" in da.bins.coords


@pytest.mark.parametrize(
    "bank_name", ["ngem_detector", "he3_detector_bank0", "he3_detector_bank1"]
)
def test_can_compute_unwrap(workflow, bank_name):
    workflow[NeXusDetectorName] = bank_name
    da = workflow.compute(TofDetector[SampleRun])

    assert "tof" in da.bins.coords


@pytest.mark.parametrize(
    "bank_name", ["ngem_detector", "he3_detector_bank0", "he3_detector_bank1"]
)
def test_can_compute_unwrap_from_custom_lut(
    workflow, tof_lookup_table, bank_name
):
    workflow[NeXusDetectorName] = bank_name
    workflow[TimeOfFlightLookupTable] = tof_lookup_table
    da = workflow.compute(TofDetector[SampleRun])

    assert "tof" in da.bins.coords


@pytest.mark.parametrize(
    "bank_name", ["ngem_detector", "he3_detector_bank0", "he3_detector_bank1"]
)
def test_can_compute_wavelength(workflow, bank_name):
    workflow[NeXusDetectorName] = bank_name
    da = workflow.compute(WavelengthDetector[SampleRun])

    assert "wavelength" in da.bins.coords
