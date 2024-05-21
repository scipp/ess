# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import zipfile
from io import BytesIO
from typing import Optional, Set

import numpy as np
import pytest
import sciline
import scipp as sc
import scipp.testing
from ess.dream import data, load_geant4_csv
from ess.powder.types import Filename, NeXusDetectorName, RawDetectorData, SampleRun


@pytest.fixture(scope="module")
def file_path():
    return data.get_path("data_dream0_new_hkl_Si_pwd.csv.zip")


@pytest.fixture(scope="module")
def file_path_without_sans():
    return data.get_path("data_dream_with_sectors.csv.zip")


# Load file into memory only once
@pytest.fixture(scope="module")
def load_file(file_path):
    with zipfile.ZipFile(file_path, "r") as archive:
        return archive.read(archive.namelist()[0])


# Load file into memory only once
@pytest.fixture(scope="module")
def load_file_without_sans(file_path_without_sans):
    with zipfile.ZipFile(file_path_without_sans, "r") as archive:
        return archive.read(archive.namelist()[0])


@pytest.fixture()
def file(load_file):
    return BytesIO(load_file)


@pytest.fixture()
def file_without_sans(load_file_without_sans):
    return BytesIO(load_file_without_sans)


def assert_index_coord(
    coord: sc.Variable, *, values: Optional[Set[int]] = None
) -> None:
    assert coord.ndim == 1
    assert coord.unit is None
    assert coord.dtype == "int64"
    if values is not None:
        assert set(np.unique(coord.values)) == values


def test_load_geant4_csv_loads_expected_structure(file):
    loaded = load_geant4_csv(file)
    assert isinstance(loaded, sc.DataGroup)
    assert loaded.keys() == {"instrument"}

    instrument = loaded["instrument"]
    assert isinstance(instrument, sc.DataGroup)
    assert instrument.keys() == {
        "mantle",
        "high_resolution",
        "sans",
        "endcap_forward",
        "endcap_backward",
    }


def test_load_geant4_csv_loads_expected_structure_without_sans(file_without_sans):
    loaded = load_geant4_csv(file_without_sans)
    assert isinstance(loaded, sc.DataGroup)
    assert loaded.keys() == {"instrument"}

    instrument = loaded["instrument"]
    assert isinstance(instrument, sc.DataGroup)
    assert instrument.keys() == {
        "mantle",
        "high_resolution",
        "endcap_forward",
        "endcap_backward",
    }


@pytest.mark.parametrize(
    "key", ["mantle", "high_resolution", "endcap_forward", "endcap_backward"]
)
def test_load_gean4_csv_set_weights_to_one(file, key):
    detector = load_geant4_csv(file)["instrument"][key]["events"]
    events = detector.bins.constituents["data"].data
    sc.testing.assert_identical(
        events, sc.ones(sizes=events.sizes, with_variances=True, unit="counts")
    )


def test_load_geant4_csv_mantle_has_expected_coords(file):
    # Only testing ranges that will not change in the future
    mantle = load_geant4_csv(file)["instrument"]["mantle"]["events"]
    assert_index_coord(mantle.coords["module"])
    assert_index_coord(mantle.coords["segment"])
    assert_index_coord(mantle.coords["counter"])
    assert_index_coord(mantle.coords["wire"], values=set(range(1, 33)))
    assert_index_coord(mantle.coords["strip"], values=set(range(1, 257)))
    assert "sector" not in mantle.coords

    assert "sector" not in mantle.bins.coords
    assert "tof" in mantle.bins.coords
    assert "wavelength" in mantle.bins.coords
    assert "position" in mantle.bins.coords


def test_load_geant4_csv_endcap_backward_has_expected_coords(file):
    endcap = load_geant4_csv(file)["instrument"]["endcap_backward"]["events"]
    assert_index_coord(endcap.coords["module"])
    assert_index_coord(endcap.coords["segment"])
    assert_index_coord(endcap.coords["counter"])
    assert_index_coord(endcap.coords["wire"], values=set(range(1, 17)))
    assert_index_coord(endcap.coords["strip"], values=set(range(1, 17)))
    assert "sector" not in endcap.coords

    assert "sector" not in endcap.bins.coords
    assert "tof" in endcap.bins.coords
    assert "wavelength" in endcap.bins.coords
    assert "position" in endcap.bins.coords


def test_load_geant4_csv_endcap_forward_has_expected_coords(file):
    endcap = load_geant4_csv(file)["instrument"]["endcap_forward"]["events"]
    assert_index_coord(endcap.coords["module"])
    assert_index_coord(endcap.coords["segment"])
    assert_index_coord(endcap.coords["counter"])
    assert_index_coord(endcap.coords["wire"], values=set(range(1, 17)))
    assert_index_coord(endcap.coords["strip"], values=set(range(1, 17)))
    assert "sector" not in endcap.coords

    assert "sector" not in endcap.bins.coords
    assert "tof" in endcap.bins.coords
    assert "wavelength" in endcap.bins.coords
    assert "position" in endcap.bins.coords


def test_load_geant4_csv_high_resolution_has_expected_coords(file):
    hr = load_geant4_csv(file)["instrument"]["high_resolution"]["events"]
    assert_index_coord(hr.coords["module"])
    assert_index_coord(hr.coords["segment"])
    assert_index_coord(hr.coords["counter"])
    assert_index_coord(hr.coords["wire"], values=set(range(1, 17)))
    assert_index_coord(hr.coords["strip"], values=set(range(1, 33)))
    assert_index_coord(hr.coords["sector"], values=set(range(1, 5)))

    assert "tof" in hr.bins.coords
    assert "wavelength" in hr.bins.coords
    assert "position" in hr.bins.coords


def test_load_geant4_csv_sans_has_expected_coords(file):
    sans = load_geant4_csv(file)["instrument"]["sans"]["events"]
    assert_index_coord(sans.coords["module"])
    assert_index_coord(sans.coords["segment"])
    assert_index_coord(sans.coords["counter"])

    # check ranges only if csv file contains events from SANS detectors
    if len(sans.coords["module"].values) > 0:
        assert_index_coord(sans.coords["wire"], values=set(range(1, 17)))
        assert_index_coord(sans.coords["strip"], values=set(range(1, 33)))
        assert_index_coord(sans.coords["sector"], values=set(range(1, 5)))

    assert "tof" in sans.bins.coords
    assert "wavelength" in sans.bins.coords
    assert "position" in sans.bins.coords


def test_geant4_in_pipeline(file_path, file):
    from ess.dream.io.geant4 import providers

    pipeline = sciline.Pipeline(
        providers,
        params={
            Filename[SampleRun]: file_path,
            NeXusDetectorName: NeXusDetectorName("mantle"),
        },
    )
    detector = pipeline.compute(RawDetectorData[SampleRun])
    expected = load_geant4_csv(file)["instrument"]["mantle"]["events"]
    sc.testing.assert_identical(detector, expected)
