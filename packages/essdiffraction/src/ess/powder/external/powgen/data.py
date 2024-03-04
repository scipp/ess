# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Utilities for loading example data for POWGEN."""

import scipp as sc

from ...types import (
    AccumulatedProtonCharge,
    CalibrationFilename,
    Filename,
    ProtonCharge,
    RawCalibrationData,
    RawData,
    RawDataAndMetadata,
    RawDataWithVariances,
    RunType,
    SampleRun,
    VanadiumRun,
)
from .types import DetectorInfo

_version = '1'

__all__ = ['get_path']


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('ess/powgen'),
        env='ESS_DATA_DIR',
        base_url='https://public.esss.dk/groups/scipp/ess/powgen/{version}/',
        version=_version,
        registry={
            # Files loadable with Mantid
            'PG3_4844_event.nxs': 'md5:d5ae38871d0a09a28ae01f85d969de1e',
            'PG3_4866_event.nxs': 'md5:3d543bc6a646e622b3f4542bc3435e7e',
            'PG3_5226_event.nxs': 'md5:58b386ebdfeb728d34fd3ba00a2d4f1e',
            'PG3_FERNS_d4832_2011_08_24.cal': 'md5:c181221ebef9fcf30114954268c7a6b6',
            # Zipped Scipp HDF5 files
            'PG3_4844_event.zip': 'md5:a644c74f5e740385469b67431b690a3e',
            'PG3_4866_event.zip': 'md5:5bc49def987f0faeb212a406b92b548e',
            'PG3_FERNS_d4832_2011_08_24.zip': 'md5:0fef4ed5f61465eaaa3f87a18f5bb80d',
        },
    )


_pooch = _make_pooch()


def get_path(name: str, unzip: bool = False) -> str:
    """
    Return the path to a data file bundled with scippneutron.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    import pooch

    return _pooch.fetch(name, processor=pooch.Unzip() if unzip else None)


def mantid_sample_file() -> str:
    return get_path('PG3_4844_event.nxs')


def mantid_vanadium_file() -> str:
    return get_path('PG3_4866_event.nxs')


def mantid_empty_instrument_file() -> str:
    return get_path('PG3_5226_event.nxs')


def mantid_calibration_file() -> str:
    return get_path('PG3_FERNS_d4832_2011_08_24.cal')


def sample_file() -> str:
    (path,) = get_path('PG3_4844_event.zip', unzip=True)
    return path


def vanadium_file() -> str:
    (path,) = get_path('PG3_4866_event.zip', unzip=True)
    return path


def calibration_file() -> str:
    (path,) = get_path('PG3_FERNS_d4832_2011_08_24.zip', unzip=True)
    return path


def pooch_load(filename: Filename[RunType]) -> RawDataAndMetadata[RunType]:
    """Load a file with pooch.

    If the file is a zip archive, it is extracted and a path to the contained
    file is returned.

    The loaded data holds both the events and any metadata from the file.
    """
    if filename.endswith('.zip'):
        (path,) = get_path(filename, unzip=True)
    else:
        path = get_path(filename)
    return RawDataAndMetadata[RunType](sc.io.load_hdf5(path))


def pooch_load_calibration(filename: CalibrationFilename) -> RawCalibrationData:
    """Load the calibration data for the POWGEN test data."""
    if filename.endswith('.zip'):
        (path,) = get_path(filename, unzip=True)
    else:
        path = get_path(filename)
    return RawCalibrationData(sc.io.load_hdf5(path))


# This can be generalized with https://github.com/scipp/sciline/issues/69.
def extract_raw_data_sample(dg: RawDataAndMetadata[SampleRun]) -> RawData[SampleRun]:
    """Return the events from a loaded data group."""
    return RawData[SampleRun](dg['data'])


def extract_raw_data_vanadium(
    dg: RawDataAndMetadata[VanadiumRun],
) -> RawDataWithVariances[VanadiumRun]:
    """Return the events from a loaded data group."""
    return RawDataWithVariances[VanadiumRun](dg['data'])


def extract_detector_info(dg: RawDataAndMetadata[SampleRun]) -> DetectorInfo:
    """Return the detector info from a loaded data group."""
    return DetectorInfo(dg['detector_info'])


def extract_proton_charge(dg: RawDataAndMetadata[RunType]) -> ProtonCharge[RunType]:
    """Return the proton charge from a loaded data group."""
    return ProtonCharge[RunType](dg['proton_charge'])


def extract_accumulated_proton_charge(
    data: RawData[RunType],
) -> AccumulatedProtonCharge[RunType]:
    """Return the stored accumulated proton charge from a loaded data group."""
    return AccumulatedProtonCharge[RunType](data.coords['gd_prtn_chrg'])


providers = (
    pooch_load,
    pooch_load_calibration,
    extract_accumulated_proton_charge,
    extract_detector_info,
    extract_proton_charge,
    extract_raw_data_sample,
    extract_raw_data_vanadium,
)
"""Sciline Providers for loading POWGEN data."""
