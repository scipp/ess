# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with TBL."""

import pathlib

from ess.reduce.data import Entry, make_registry

_registry = make_registry(
    'ess/tbl',
    version="1",
    files={
        "tbl_sample_data_2025-03.hdf": "md5:12db6bc06721278b3abe47992eac3e77",
        "TBL-tof-lookup-table-no-choppers.h5": "md5:8bc98fac0ee64fc8f5decf509c75bafe",
        'tbl-orca-focussing.hdf.zip': Entry(
            alg='md5', chk='f365acd9ea45dd205c0b9398d163cfa4', unzip=True
        ),
    },
)


def tutorial_sample_data() -> pathlib.Path:
    """ """
    return _registry.get_path("tbl_sample_data_2025-03.hdf")


def tbl_tof_lookup_table_no_choppers() -> pathlib.Path:
    """
    TBL TOF lookup table without choppers.
    This file is used to convert the neutron arrival time to time-of-flight.

    This table was computed using `Create a time-of-flight lookup table for TBL
    <../../user-guide/tbl/tbl-make-tof-lookup-table.rst>`_
    with ``NumberOfSimulatedNeutrons = 2_000_000``.
    """
    return _registry.get_path("TBL-tof-lookup-table-no-choppers.h5")


def tbl_orca_focussing_data() -> pathlib.Path:
    """
    Return the path to the TBL ORCA HDF5 file used for camera focussing.
    Note that the images in this file have been resized from 2048x2048 to 512x512
    to reduce the file size.
    """

    return _registry.get_path('tbl-orca-focussing.hdf.zip')
