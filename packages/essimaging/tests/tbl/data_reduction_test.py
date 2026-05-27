# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import ess.tbl.data  # noqa: F401
import pytest
import sciline as sl
from ess import tbl

from ess.imaging.types import (
    Filename,
    LookupTableFilename,
    NeXusDetectorName,
    NXdetector,
    RawDetector,
    SampleRun,
    WavelengthDetector,
)
from ess.reduce import unwrap


@pytest.fixture
def workflow() -> sl.Pipeline:
    """
    Workflow for loading NeXus data.
    """
    wf = tbl.TblWorkflow()
    wf[Filename[SampleRun]] = tbl.data.tutorial_sample_data()
    wf[LookupTableFilename[SampleRun, NXdetector]] = (
        tbl.data.tbl_wavelength_lookup_table_no_choppers()
    )
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
def test_can_compute_wavelength(workflow, bank_name):
    workflow[NeXusDetectorName] = bank_name
    da = workflow.compute(WavelengthDetector[SampleRun])

    assert "wavelength" in da.bins.coords


@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize(
    "bank_name", ["ngem_detector", "he3_detector_bank0", "he3_detector_bank1"]
)
def test_can_compute_wavelength_from_on_the_fly_lut(mode, bank_name):
    wf = tbl.TblWorkflow(mode=mode)
    wf[Filename[SampleRun]] = tbl.data.tutorial_sample_data()
    wf[unwrap.DiskChoppers[SampleRun]] = {}
    wf[NeXusDetectorName] = bank_name

    da = wf.compute(WavelengthDetector[SampleRun])
    assert "wavelength" in da.bins.coords
