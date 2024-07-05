# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import io
import pathlib
from functools import partial

import h5py
import scipp as sc


def _create_dataset_from_var(
    *,
    root_entry: h5py.Group,
    var: sc.Variable,
    name: str,
    long_name: str | None = None,
    compression: str | None = None,
    compression_opts: int | None = None,
) -> h5py.Dataset:
    compression_options = {}
    if compression is not None:
        compression_options["compression"] = compression
    if compression_opts is not None:
        compression_options["compression_opts"] = compression_opts

    dataset = root_entry.create_dataset(
        name,
        data=var.values,
        **compression_options,
    )
    dataset.attrs["units"] = str(var.unit)
    if long_name is not None:
        dataset.attrs["long_name"] = long_name
    return dataset


_create_compressed_dataset = partial(
    _create_dataset_from_var,
    compression="gzip",
    compression_opts=4,
)


def _create_root_data_entry(file_obj: h5py.File) -> h5py.Group:
    nx_entry = file_obj.create_group("NMX_data")
    nx_entry.attrs["NX_class"] = "NXentry"
    nx_entry.attrs["default"] = "data"
    nx_entry.attrs["name"] = "NMX"
    nx_entry["name"] = "NMX"
    nx_entry["definition"] = "TOFRAW"
    return nx_entry


def _create_sample_group(data: sc.DataGroup, nx_entry: h5py.Group) -> h5py.Group:
    nx_sample = nx_entry.create_group("NXsample")
    nx_sample["name"] = data['sample_name'].value
    _create_dataset_from_var(
        root_entry=nx_sample,
        var=data['crystal_rotation'],
        name='crystal_rotation',
        long_name='crystal rotation in Phi (XYZ)',
    )
    return nx_sample


def _create_instrument_group(data: sc.DataGroup, nx_entry: h5py.Group) -> h5py.Group:
    nx_instrument = nx_entry.create_group("NXinstrument")
    nx_instrument.create_dataset("proton_charge", data=data['proton_charge'].values)

    nx_detector_1 = nx_instrument.create_group("detector_1")
    # Detector counts
    _create_compressed_dataset(
        root_entry=nx_detector_1,
        name="counts",
        var=data['counts'],
    )
    # Time of arrival bin edges
    _create_dataset_from_var(
        root_entry=nx_detector_1,
        var=data['counts'].coords['t'],
        name="t_bin",
        long_name="t_bin TOF (ms)",
    )
    # Pixel IDs
    _create_compressed_dataset(
        root_entry=nx_detector_1,
        name="pixel_id",
        var=data['counts'].coords['id'],
        long_name="pixel ID",
    )
    return nx_instrument


def _create_detector_group(data: sc.DataGroup, nx_entry: h5py.Group) -> h5py.Group:
    nx_detector = nx_entry.create_group("NXdetector")
    # Position of the first pixel (lowest ID) in the detector
    _create_compressed_dataset(
        root_entry=nx_detector,
        name="origin",
        var=data['origin_position'],
    )
    # Fast axis, along where the pixel ID increases by 1
    _create_dataset_from_var(
        root_entry=nx_detector, var=data['fast_axis'], name="fast_axis"
    )
    # Slow axis, along where the pixel ID increases
    # by the number of pixels in the fast axis
    _create_dataset_from_var(
        root_entry=nx_detector, var=data['slow_axis'], name="slow_axis"
    )
    return nx_detector


def _create_source_group(data: sc.DataGroup, nx_entry: h5py.Group) -> h5py.Group:
    nx_source = nx_entry.create_group("NXsource")
    nx_source["name"] = "European Spallation Source"
    nx_source["short_name"] = "ESS"
    nx_source["type"] = "Spallation Neutron Source"
    nx_source["distance"] = sc.norm(data['source_position']).value
    nx_source["probe"] = "neutron"
    nx_source["target_material"] = "W"
    return nx_source


def export_as_nexus(
    data: sc.DataGroup, output_file: str | pathlib.Path | io.BytesIO
) -> None:
    """Export the reduced data to a NeXus file.

    Currently exporting step is not expected to be part of sciline pipelines.
    """
    with h5py.File(output_file, "w") as f:
        f.attrs["default"] = "NMX_data"
        nx_entry = _create_root_data_entry(f)
        _create_sample_group(data, nx_entry)
        _create_instrument_group(data, nx_entry)
        _create_detector_group(data, nx_entry)
        _create_source_group(data, nx_entry)
