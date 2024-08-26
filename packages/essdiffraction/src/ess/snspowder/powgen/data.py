# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Utilities for loading example data for POWGEN."""

import scipp as sc

from ess.powder.types import (
    AccumulatedProtonCharge,
    CalibrationData,
    CalibrationFilename,
    DetectorBankSizes,
    DetectorData,
    Filename,
    ProtonCharge,
    RawDataAndMetadata,
    RunType,
)

_version = "1"


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/powgen"),
        env="ESS_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/powgen/{version}/",
        version=_version,
        registry={
            # Files loadable with Mantid
            "PG3_4844_event.nxs": "md5:d5ae38871d0a09a28ae01f85d969de1e",
            "PG3_4866_event.nxs": "md5:3d543bc6a646e622b3f4542bc3435e7e",
            "PG3_5226_event.nxs": "md5:58b386ebdfeb728d34fd3ba00a2d4f1e",
            "PG3_FERNS_d4832_2011_08_24.cal": "md5:c181221ebef9fcf30114954268c7a6b6",
            # Zipped Scipp HDF5 files
            "PG3_4844_event.zip": "md5:a644c74f5e740385469b67431b690a3e",
            "PG3_4866_event.zip": "md5:5bc49def987f0faeb212a406b92b548e",
            "PG3_FERNS_d4832_2011_08_24.zip": "md5:0fef4ed5f61465eaaa3f87a18f5bb80d",
            "PG3_FERNS_d4832_2011_08_24_spectrum.h5": "md5:7aee0b40deee22d57e21558baa7a6a1a",  # noqa: E501
        },
    )


_pooch = _make_pooch()


def _get_path(name: str) -> str:
    """
    Return the path to a data file bundled with scippneutron.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    import pooch

    if name.endswith(".zip"):
        (path,) = _pooch.fetch(name, processor=pooch.Unzip())
    else:
        path = _pooch.fetch(name)
    return path


def powgen_tutorial_mantid_sample_file() -> str:
    return _get_path("PG3_4844_event.nxs")


def powgen_tutorial_mantid_vanadium_file() -> str:
    return _get_path("PG3_4866_event.nxs")


def powgen_tutorial_mantid_empty_instrument_file() -> str:
    return _get_path("PG3_5226_event.nxs")


def powgen_tutorial_mantid_calibration_file() -> str:
    return _get_path("PG3_FERNS_d4832_2011_08_24.cal")


def powgen_tutorial_sample_file() -> str:
    return _get_path("PG3_4844_event.zip")


def powgen_tutorial_vanadium_file() -> str:
    return _get_path("PG3_4866_event.zip")


def powgen_tutorial_calibration_file() -> str:
    return _get_path("PG3_FERNS_d4832_2011_08_24_spectrum.h5")


def pooch_load(filename: Filename[RunType]) -> RawDataAndMetadata[RunType]:
    """Load a file with pooch.

    If the file is a zip archive, it is extracted and a path to the contained
    file is returned.

    The loaded data holds both the events and any metadata from the file.
    """
    return RawDataAndMetadata[RunType](sc.io.load_hdf5(filename))


def pooch_load_calibration(
    filename: CalibrationFilename,
    detector_dimensions: DetectorBankSizes,
) -> CalibrationData:
    """Load the calibration data for the POWGEN test data."""
    if filename is None:
        return CalibrationFilename(None)
    ds = sc.io.load_hdf5(filename)
    ds = sc.Dataset(
        {
            key: da.fold(dim='spectrum', sizes=detector_dimensions)
            for key, da in ds.items()
        }
    )
    return CalibrationData(ds)


def extract_raw_data(
    dg: RawDataAndMetadata[RunType], sizes: DetectorBankSizes
) -> DetectorData[RunType]:
    """Return the events from a loaded data group."""
    # Remove the tof binning and dimension, as it is not needed and it gets in the way
    # of masking.
    out = dg["data"].squeeze()
    out.coords.pop("tof", None)
    out = out.fold(dim="spectrum", sizes=sizes)
    return DetectorData[RunType](out)


def extract_proton_charge(dg: RawDataAndMetadata[RunType]) -> ProtonCharge[RunType]:
    """Return the proton charge from a loaded data group."""
    return ProtonCharge[RunType](dg["proton_charge"])


def extract_accumulated_proton_charge(
    data: DetectorData[RunType],
) -> AccumulatedProtonCharge[RunType]:
    """Return the stored accumulated proton charge from a loaded data group."""
    return AccumulatedProtonCharge[RunType](data.coords["gd_prtn_chrg"])


providers = (
    pooch_load,
    pooch_load_calibration,
    extract_accumulated_proton_charge,
    extract_proton_charge,
    extract_raw_data,
)
"""Sciline Providers for loading POWGEN data."""
