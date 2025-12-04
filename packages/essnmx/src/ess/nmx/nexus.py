# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import io
import pathlib
import warnings
from typing import Any

import h5py
import numpy as np
import scipp as sc

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


def _create_dataset_from_string(*, root_entry: h5py.Group, name: str, var: str) -> None:
    root_entry.create_dataset(name, dtype=h5py.string_dtype(), data=var)


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
    else:
        compression_filter = None
        compression_opts = None

    return {"compression": compression_filter, "compression_opts": compression_opts}


def _create_lauetof_data_entry(file_obj: h5py.File) -> h5py.Group:
    nx_entry = file_obj.create_group("entry")
    nx_entry.attrs["NX_class"] = "NXentry"
    return nx_entry


def _add_lauetof_definition(nx_entry: h5py.Group) -> None:
    _create_dataset_from_string(root_entry=nx_entry, name="definition", var="NXlauetof")


def _add_lauetof_instrument(nx_entry: h5py.Group) -> h5py.Group:
    nx_instrument = nx_entry.create_group("instrument")
    nx_instrument.attrs["NX_class"] = "NXinstrument"
    _create_dataset_from_string(root_entry=nx_instrument, name="name", var="NMX")
    return nx_instrument


def _add_lauetof_source_group(
    source_position: sc.Variable, nx_instrument: h5py.Group
) -> None:
    nx_source = nx_instrument.create_group("source")
    nx_source.attrs["NX_class"] = "NXsource"
    _create_dataset_from_string(
        root_entry=nx_source, name="name", var="European Spallation Source"
    )
    _create_dataset_from_string(root_entry=nx_source, name="short_name", var="ESS")
    _create_dataset_from_string(
        root_entry=nx_source, name="type", var="Spallation Neutron Source"
    )
    _create_dataset_from_var(
        root_entry=nx_source, name="distance", var=sc.norm(source_position)
    )
    # Legacy probe information.
    _create_dataset_from_string(root_entry=nx_source, name="probe", var="neutron")


def _add_lauetof_detector_group(
    *,
    detector_name: str,
    x_pixel_size: sc.Variable,
    y_pixel_size: sc.Variable,
    origin_position: sc.Variable,
    fast_axis: sc.Variable,
    slow_axis: sc.Variable,
    distance: sc.Variable,
    polar_angle: sc.Variable,
    azimuthal_angle: sc.Variable,
    nx_instrument: h5py.Group,
) -> None:
    nx_det = nx_instrument.create_group(detector_name)  # Detector name
    nx_det.attrs["NX_class"] = "NXdetector"
    _create_dataset_from_var(name="polar_angle", root_entry=nx_det, var=polar_angle)
    _create_dataset_from_var(
        name="azimuthal_angle", root_entry=nx_det, var=azimuthal_angle
    )
    _create_dataset_from_var(name="x_pixel_size", root_entry=nx_det, var=x_pixel_size)
    _create_dataset_from_var(name="y_pixel_size", root_entry=nx_det, var=y_pixel_size)
    _create_dataset_from_var(name="distance", root_entry=nx_det, var=distance)
    # Legacy geometry information until we have a better way to store it
    _create_dataset_from_var(name="origin", root_entry=nx_det, var=origin_position)
    # Fast axis, along where the pixel ID increases by 1
    _create_dataset_from_var(root_entry=nx_det, name="fast_axis", var=fast_axis)
    # Slow axis, along where the pixel ID increases
    # by the number of pixels in the fast axis
    _create_dataset_from_var(root_entry=nx_det, name="slow_axis", var=slow_axis)


def _add_lauetof_sample_group(
    *,
    crystal_rotation: sc.Variable,
    sample_name: str | sc.Variable,
    sample_orientation_matrix: sc.Variable,
    sample_unit_cell: sc.Variable,
    nx_entry: h5py.Group,
) -> None:
    nx_sample = nx_entry.create_group("sample")
    nx_sample.attrs["NX_class"] = "NXsample"
    _create_dataset_from_var(
        root_entry=nx_sample,
        var=crystal_rotation,
        name='crystal_rotation',
        long_name='crystal rotation in Phi (XYZ)',
    )
    _create_dataset_from_string(
        root_entry=nx_sample,
        name='name',
        var=sample_name if isinstance(sample_name, str) else sample_name.value,
    )
    _create_dataset_from_var(
        name='orientation_matrix', root_entry=nx_sample, var=sample_orientation_matrix
    )
    _create_dataset_from_var(
        name='unit_cell',
        root_entry=nx_sample,
        var=sample_unit_cell,
    )


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


def export_static_metadata_as_nxlauetof(
    sample_metadata: NMXSampleMetadata,
    source_metadata: NMXSourceMetadata,
    output_file: str | pathlib.Path | io.BytesIO,
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
    _check_file(output_file, overwrite=True)
    with h5py.File(output_file, "w") as f:
        f.attrs["NX_class"] = "NXlauetof"
        nx_entry = _create_lauetof_data_entry(f)
        _add_lauetof_definition(nx_entry)
        _add_lauetof_sample_group(
            crystal_rotation=sample_metadata.crystal_rotation,
            sample_name=sample_metadata.sample_name,
            sample_orientation_matrix=sample_metadata.sample_orientation_matrix,
            sample_unit_cell=sample_metadata.sample_unit_cell,
            nx_entry=nx_entry,
        )
        nx_instrument = _add_lauetof_instrument(nx_entry)
        _add_lauetof_source_group(source_metadata.source_position, nx_instrument)
        # Skipping ``NXdata``(name) field with data link
        # Add arbitrary metadata
        _add_arbitrary_metadata(nx_entry, **arbitrary_metadata)


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

    with h5py.File(output_file, "r+") as f:
        nx_entry = f["entry"]
        # Placeholder for ``monitor`` group
        _add_lauetof_monitor_group(
            tof_bin_coord=monitor_metadata.tof_bin_coord,
            monitor_histogram=monitor_metadata.monitor_histogram,
            nx_entry=nx_entry,
        )


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

    with h5py.File(output_file, "r+") as f:
        nx_entry = f["entry"]
        if "instrument" not in nx_entry:
            nx_instrument = _add_lauetof_instrument(f["entry"])
        else:
            nx_instrument = nx_entry["instrument"]

        # Add detector group metadata
        _add_lauetof_detector_group(
            detector_name=detector_metadata.detector_name,
            x_pixel_size=detector_metadata.x_pixel_size,
            y_pixel_size=detector_metadata.y_pixel_size,
            origin_position=detector_metadata.origin_position,
            fast_axis=detector_metadata.fast_axis,
            slow_axis=detector_metadata.slow_axis,
            distance=detector_metadata.distance,
            polar_angle=detector_metadata.polar_angle,
            azimuthal_angle=detector_metadata.azimuthal_angle,
            nx_instrument=nx_instrument,
        )


def _add_lauetof_monitor_group(
    *,
    tof_bin_coord: str,
    monitor_histogram: sc.DataArray,
    nx_entry: h5py.Group,
) -> None:
    nx_monitor = nx_entry.create_group("control")
    nx_monitor.attrs["NX_class"] = "NXmonitor"
    _create_dataset_from_string(root_entry=nx_monitor, name='mode', var='monitor')
    nx_monitor["preset"] = 0.0  # Check if this is the correct value
    data_dset = _create_dataset_from_var(
        name='data',
        root_entry=nx_monitor,
        var=monitor_histogram.data,
    )
    data_dset.attrs["signal"] = 1
    data_dset.attrs["primary"] = 1

    _create_dataset_from_var(
        name='time_of_flight',
        root_entry=nx_monitor,
        var=monitor_histogram.coords[tof_bin_coord],
    )


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
    compress_counts:
        If ``True``, the detector counts are compressed using bitshuffle.
        It is because only the detector counts are expected to be large.

    """
    if not append_mode:
        raise NotImplementedError("Only append mode is supported for now.")

    with h5py.File(output_file, "r+") as f:
        nx_detector: h5py.Group = f[f"entry/instrument/{detector_name}"]
        # Data - shape: [n_x_pixels, n_y_pixels, n_tof_bins]
        # The actual application definition defines it as integer,
        # so we overwrite the dtype here.
        num_x, num_y = da.sizes['x_pixel_offset'], da.sizes['y_pixel_offset']

        if compress_mode != Compression.NONE:
            compression_args = _retrieve_compression_arguments(compress_mode)
            data_dset = _create_dataset_from_var(
                name="data",
                root_entry=nx_detector,
                var=da.data,
                chunks=(num_x, num_y, 1),  # Chunk along tof axis
                dtype=np.uint,
                **compression_args,
            )
        else:
            data_dset = _create_dataset_from_var(
                name="data", root_entry=nx_detector, var=da.data, dtype=np.uint
            )

        data_dset.attrs["signal"] = 1
        _create_dataset_from_var(
            name='time_of_flight',
            root_entry=nx_detector,
            var=sc.midpoints(da.coords['tof'], dim='tof'),
        )
