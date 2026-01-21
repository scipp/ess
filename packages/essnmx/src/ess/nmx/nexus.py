# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import io
import pathlib
import warnings
from typing import Any

import h5py
import numpy as np
import scipp as sc
import scippnexus as snx

from .configurations import Compression
from .types import (
    NMXDetectorMetadata,
    NMXMonitorMetadata,
    NMXSampleMetadata,
    NMXSourceMetadata,
)


def _check_file(
    filename: str | pathlib.Path | io.BytesIO, overwrite: bool
) -> pathlib.Path | io.BytesIO:
    if isinstance(filename, str | pathlib.Path):
        filename = pathlib.Path(filename)
        if filename.exists() and not overwrite:
            raise FileExistsError(
                f"File '{filename}' already exists. Use `overwrite=True` to overwrite."
            )
    return filename


def _create_dataset_from_var(
    *,
    root_entry: h5py.Group,
    var: sc.Variable,
    name: str,
    long_name: str | None = None,
    compression: str | None = None,
    compression_opts: int | tuple[int, int] | None = None,
    chunks: tuple[int, ...] | int | bool | None = None,
    dtype: Any = None,
) -> h5py.Dataset:
    compression_options = {}
    if compression is not None:
        compression_options["compression"] = compression
    if compression_opts is not None:
        compression_options["compression_opts"] = compression_opts

    dataset = root_entry.create_dataset(
        name,
        data=var.values if dtype is None else var.values.astype(dtype, copy=False),
        chunks=chunks,
        **compression_options,
    )
    if var.unit is not None:
        dataset.attrs["units"] = str(var.unit)
    if long_name is not None:
        dataset.attrs["long_name"] = long_name
    return dataset


def _retrieve_compression_arguments(compress_mode: Compression) -> dict:
    """Returns compression filter and opts arguments for the ``compress_mode``.

    Returns an empty dictionary if an unimplemented compression mode
    or `NONE` compression mode is selected.

    """
    if compress_mode == Compression.BITSHUFFLE_LZ4:
        try:
            import bitshuffle.h5

            compression_filter = bitshuffle.h5.H5FILTER
            compression_opts = (0, bitshuffle.h5.H5_COMPRESS_LZ4)
        except ImportError:
            warnings.warn(
                UserWarning(
                    "Could not find the bitshuffle.h5 module from bitshuffle package. "
                    "The bitshuffle package is not installed properly. "
                    "Trying with gzip compression instead..."
                ),
                stacklevel=2,
            )
            compression_filter = "gzip"
            compression_opts = 4
    elif compress_mode == Compression.GZIP:
        compression_filter = "gzip"
        compression_opts = 4
    elif compress_mode == Compression.NONE:
        return {}
    else:
        warnings.warn(
            UserWarning(
                f"Compression Mode {compress_mode} is not implemented yet. "
                "Not Compressing the dataset... "
                "Try `GZIP` or `BITSHUFFLE_LZ4` if compression is needed."
            ),
            stacklevel=2,
        )
        return {}

    return {"compression": compression_filter, "compression_opts": compression_opts}


def _add_arbitrary_metadata(
    nx_entry: h5py.Group, **arbitrary_metadata: sc.Variable
) -> None:
    if not arbitrary_metadata:
        return

    metadata_group = nx_entry.create_group("metadata")
    for key, value in arbitrary_metadata.items():
        if not isinstance(value, sc.Variable):
            import warnings

            msg = f"Skipping metadata key '{key}' as it is not a scipp.Variable."
            warnings.warn(UserWarning(msg), stacklevel=2)
            continue
        else:
            _create_dataset_from_var(
                name=key,
                root_entry=metadata_group,
                var=value,
            )


def _set_default_instrument(nx_entry: snx.Group) -> snx.Group:
    """Return NXinstrument group.

    If 'instrument' exists in the NXentry group, it returns the existing one.
    Otherwise, new NXinstrument group is created and returned.
    The default NXinstrument group has a field 'name' with the instrument name, 'NMX'.
    """
    if "instrument" not in nx_entry:
        nx_instrument = nx_entry.create_class("instrument", 'NXinstrument')
        nx_instrument.create_field(key='name', value='NMX')
    else:
        nx_instrument = nx_entry["instrument"]

    return nx_instrument


def export_static_metadata_as_nxlauetof(
    *,
    sample_metadata: NMXSampleMetadata,
    source_metadata: NMXSourceMetadata,
    output_file: str | pathlib.Path | io.BytesIO,
    overwrite: bool = False,
    **arbitrary_metadata: sc.Variable,
) -> None:
    """Export the metadata to a NeXus file with the LAUE_TOF application definition.

    ``Metadata`` in this context refers to the information
    that is not part of the reduced detector counts itself,
    but is necessary for the interpretation of the reduced data.
    Since NMX can have arbitrary number of detectors,
    this function can take multiple detector metadata objects.

    Parameters
    ----------
    sample_metadata:
        Sample metadata object.
    source_metadata:
        Source metadata object.
    monitor_metadata:
        Monitor metadata object.
    output_file:
        Output file path.
    arbitrary_metadata:
        Arbitrary metadata that does not fit into the existing metadata objects.

    """
    _check_file(output_file, overwrite=overwrite)
    with snx.File(output_file, "w") as f:
        f._group.attrs["NX_class"] = "NXlauetof"
        nx_entry = f.create_class(name='entry', class_name='NXlauetof')
        nx_entry.create_field('definitions', value='NXlauetof')
        nx_entry['sample'] = sample_metadata

        nx_instrument = _set_default_instrument(nx_entry)
        nx_instrument['source'] = source_metadata
        _add_arbitrary_metadata(nx_entry._group, **arbitrary_metadata)


def export_monitor_metadata_as_nxlauetof(
    monitor_metadata: NMXMonitorMetadata,
    output_file: str | pathlib.Path | io.BytesIO,
    append_mode: bool = True,
) -> None:
    """Export the detector specific metadata to a NeXus file.

    Since NMX can have arbitrary number of detectors,
    this function can take multiple detector metadata objects.

    Parameters
    ----------
    monitor_metadata:
        Monitor metadata object.
    output_file:
        Output file path.

    """
    if not append_mode:
        raise NotImplementedError("Only append mode is supported for now.")

    with snx.File(output_file, "r+") as f:
        nx_entry = f["entry"]
        nx_entry["control"] = monitor_metadata


def export_detector_metadata_as_nxlauetof(
    detector_metadata: NMXDetectorMetadata,
    output_file: str | pathlib.Path | io.BytesIO,
    append_mode: bool = True,
) -> None:
    """Export the detector specific metadata to a NeXus file.

    Since NMX can have arbitrary number of detectors,
    this function can take multiple detector metadata objects.

    Parameters
    ----------
    detector_metadatas:
        Detector metadata objects.
    output_file:
        Output file path.

    """

    if not append_mode:
        raise NotImplementedError("Only append mode is supported for now.")

    with snx.File(output_file, "r+") as f:
        nx_entry: snx.Group = f["entry"]
        nx_instrument = _set_default_instrument(nx_entry)
        nx_instrument[detector_metadata.detector_name] = detector_metadata


def export_reduced_data_as_nxlauetof(
    detector_name: str,
    da: sc.DataArray,
    output_file: str | pathlib.Path | io.BytesIO,
    *,
    append_mode: bool = True,
    compress_mode: Compression = Compression.BITSHUFFLE_LZ4,
) -> None:
    """Export the reduced data to a NeXus file with the LAUE_TOF application definition.

    Even though this function only exports
    reduced data(detector counts and its coordinates),
    the input should contain all the necessary metadata
    for minimum sanity check.

    Parameters
    ----------
    dg:
        Reduced data and metadata.
    output_file:
        Output file path.
    append_mode:
        If ``True``, the file is opened in append mode.
        If ``False``, the file is opened in None-append mode.
        > None-append mode is not supported for now.
        > Only append mode is supported for now.
    compress_mode:
        The detector counts are compressed using the ``compress_mode``.
        It is because only the detector counts are expected to be large.
        If ``Compression.BITSHUFFLE_LZ4`` is selected
        but the bitshuffle is not supported for the environment,
        it will fall back to ``Compression.GZIP``.
        Select ``Compression.NONE`` if compression is not needed.

    """
    if not append_mode:
        raise NotImplementedError("Only append mode is supported for now.")

    with h5py.File(output_file, "r+") as f:
        nx_detector: h5py.Group = f[f"entry/instrument/{detector_name}"]
        # Data - shape: [n_x_pixels, n_y_pixels, n_tof_bins]
        # The actual application definition defines it as integer,
        # so we overwrite the dtype here.

        compression_args = _retrieve_compression_arguments(compress_mode)
        if compress_mode != Compression.NONE:  # Calculate the chunk sizes
            num_x, num_y = da.sizes['x_pixel_offset'], da.sizes['y_pixel_offset']
            compression_args['chunks'] = (num_x, num_y, 1)  # Chunk along tof axis

        data_dset = _create_dataset_from_var(
            name="data",
            root_entry=nx_detector,
            var=da.data,
            dtype=np.uint,
            **compression_args,
        )

        data_dset.attrs["signal"] = 1

        if 'tof' in da.coords:
            _create_dataset_from_var(
                name='time_of_flight',
                root_entry=nx_detector,
                var=sc.midpoints(da.coords['tof'], dim='tof'),
            )
        elif 'event_time_offset' in da.coords:
            _create_dataset_from_var(
                name='event_time_offset',
                root_entry=nx_detector,
                var=sc.midpoints(
                    da.coords['event_time_offset'], dim='event_time_offset'
                ),
            )
        else:
            raise ValueError("Could not find time-related bin edges to store.")
