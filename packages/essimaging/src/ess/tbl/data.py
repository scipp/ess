# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with TBL."""

import pathlib

from ess.reduce.data import Entry, make_registry

_registry = make_registry(
    'ess/tbl',
    version="2",
    files={
        "tbl_sample_data_2025-03.hdf": "md5:12db6bc06721278b3abe47992eac3e77",
        "TBL-wavelength-lookup-table-no-choppers-5m-35m.h5": "md5:e28793b7e1c12986ee63a1df68723268",  # noqa: E501
        'tbl-orca-focussing.hdf.zip': Entry(
            alg='md5', chk='f365acd9ea45dd205c0b9398d163cfa4', unzip=True
        ),
        "ymir_lego_dark_run.hdf": "md5:c0ed089dd7663986042e29fb47514130",
        "ymir_lego_openbeam_run.hdf": "md5:00375becd54d2ed3be096413dc30883c",
        "ymir_lego_sample_run.hdf": "md5:ae56a335cf3d4e87ef090ec4e51da69c",
    },
)


def tutorial_sample_data() -> pathlib.Path:
    """ """
    return _registry.get_path("tbl_sample_data_2025-03.hdf")


def tbl_wavelength_lookup_table_no_choppers() -> pathlib.Path:
    """
    TBL wavelength lookup table without choppers.
    This file is used to convert the neutron arrival time to wavelength.

    This table was computed using `Create a wavelength lookup table for TBL
    <../../tbl/tbl-make-wavelength-lookup-table.rst>`_
    with ``NumberOfSimulatedNeutrons = 5_000_000``.
    """
    return _registry.get_path("TBL-wavelength-lookup-table-no-choppers-5m-35m.h5")


def tbl_orca_focussing_data() -> pathlib.Path:
    """
    Return the path to the TBL ORCA HDF5 file used for camera focussing.
    Note that the images in this file have been resized from 2048x2048 to 512x512
    to reduce the file size.
    """

    return _registry.get_path('tbl-orca-focussing.hdf.zip')


def tbl_lego_dark_run() -> pathlib.Path:
    """
    Return the path to the TBL LEGO dark run HDF5 file, created from the YMIR data.
    This file was created using the tools/make-tbl-images-from-ymir.ipynb notebook.
    A TBL file (857127_00000212.hdf) was used as a template for the NeXus structure.
    The dark run data was extracted from the YMIR LEGO run (first 5 frames).
    """

    return _registry.get_path("ymir_lego_dark_run.hdf")


def tbl_lego_openbeam_run() -> pathlib.Path:
    """
    Return the path to the TBL LEGO open beam run HDF5 file, created from the YMIR data.
    This file was created using the tools/make-tbl-images-from-ymir.ipynb notebook.
    A TBL file (857127_00000212.hdf) was used as a template for the NeXus structure.
    The open beam run data was extracted from the YMIR LEGO run (frames 5 to 10).
    """

    return _registry.get_path("ymir_lego_openbeam_run.hdf")


def tbl_lego_sample_run() -> pathlib.Path:
    """
    Return the path to the TBL LEGO sample run HDF5 file, created from the YMIR data.
    This file was created using the tools/make-tbl-images-from-ymir.ipynb notebook.
    A TBL file (857127_00000212.hdf) was used as a template for the NeXus structure.
    The sample run data was extracted from the YMIR LEGO run (frames 10 and onward).
    """

    return _registry.get_path("ymir_lego_sample_run.hdf")
