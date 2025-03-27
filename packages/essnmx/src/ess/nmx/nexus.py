# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import io
import pathlib
import warnings
from collections.abc import Callable, Generator
from functools import partial, wraps
from typing import Any, TypeVar

import h5py
import numpy as np
import sciline as sl
import scipp as sc

from .types import (
    DetectorIndex,
    DetectorName,
    FilePath,
    NMXDetectorMetadata,
    NMXExperimentMetadata,
    NMXReducedDataGroup,
)


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


@wraps(_create_dataset_from_var)
def _create_compressed_dataset(*args, **kwargs):
    """Create dataset with compression options.

    It will try to use ``bitshuffle`` for compression if available.
    Otherwise, it will fall back to ``gzip`` compression.

    [``Bitshuffle/LZ4``](https://github.com/kiyo-masui/bitshuffle)
    is used for convenience.
    Since ``Dectris`` uses it for their Nexus file compression,
    it is compatible with DIALS.
    ``Bitshuffle/LZ4`` tends to give similar results to
    GZIP and other compression algorithms with better performance.
    A naive implementation of bitshuffle/LZ4 compression,
    shown in [issue #124](https://github.com/scipp/essnmx/issues/124),
    led to 80% file reduction (365 MB vs 1.8 GB).

    """
    try:
        import bitshuffle.h5

        compression_filter = bitshuffle.h5.H5FILTER
        default_compression_opts = (0, bitshuffle.h5.H5_COMPRESS_LZ4)
    except ImportError:
        warnings.warn(
            UserWarning(
                "Could not find the bitshuffle.h5 module from bitshuffle package. "
                "The bitshuffle package is not installed or only partially installed. "
                "Exporting to NeXus files with bitshuffle compression is not possible."
            ),
            stacklevel=2,
        )
        compression_filter = "gzip"
        default_compression_opts = 4

    return _create_dataset_from_var(
        *args,
        **kwargs,
        compression=compression_filter,
        compression_opts=default_compression_opts,
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
    warnings.warn(
        DeprecationWarning(
            "Exporting to custom NeXus format will be deprecated in the near future."
            "Please use ``export_as_nxlauetof`` instead."
        ),
        stacklevel=2,
    )
    with h5py.File(output_file, "w") as f:
        f.attrs["default"] = "NMX_data"
        nx_entry = _create_root_data_entry(f)
        _create_sample_group(data, nx_entry)
        _create_instrument_group(data, nx_entry)
        _create_detector_group(data, nx_entry)
        _create_source_group(data, nx_entry)


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
    dg: NMXExperimentMetadata, nx_instrument: h5py.Group
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
        root_entry=nx_source, name="distance", var=sc.norm(dg["source_position"])
    )
    # Legacy probe information.
    _create_dataset_from_string(root_entry=nx_source, name="probe", var="neutron")


def _add_lauetof_detector_group(dg: sc.DataGroup, nx_instrument: h5py.Group) -> None:
    nx_detector = nx_instrument.create_group(dg["detector_name"].value)  # Detector name
    nx_detector.attrs["NX_class"] = "NXdetector"
    _create_dataset_from_var(
        name="polar_angle",
        root_entry=nx_detector,
        var=sc.scalar(0, unit='deg'),  # TODO: Add real data
    )
    _create_dataset_from_var(
        name="azimuthal_angle",
        root_entry=nx_detector,
        var=sc.scalar(0, unit='deg'),  # TODO: Add real data
    )
    _create_dataset_from_var(
        name="x_pixel_size", root_entry=nx_detector, var=dg["x_pixel_size"]
    )
    _create_dataset_from_var(
        name="y_pixel_size", root_entry=nx_detector, var=dg["y_pixel_size"]
    )
    _create_dataset_from_var(
        name="distance",
        root_entry=nx_detector,
        var=sc.scalar(0, unit='m'),  # TODO: Add real data
    )
    # Legacy geometry information until we have a better way to store it
    _create_dataset_from_var(
        name="origin", root_entry=nx_detector, var=dg['origin_position']
    )
    # Fast axis, along where the pixel ID increases by 1
    _create_dataset_from_var(
        root_entry=nx_detector, var=dg['fast_axis'], name="fast_axis"
    )
    # Slow axis, along where the pixel ID increases
    # by the number of pixels in the fast axis
    _create_dataset_from_var(
        root_entry=nx_detector, var=dg['slow_axis'], name="slow_axis"
    )


def _add_lauetof_sample_group(dg: NMXExperimentMetadata, nx_entry: h5py.Group) -> None:
    nx_sample = nx_entry.create_group("sample")
    nx_sample.attrs["NX_class"] = "NXsample"
    _create_dataset_from_var(
        root_entry=nx_sample,
        var=dg['crystal_rotation'],
        name='crystal_rotation',
        long_name='crystal rotation in Phi (XYZ)',
    )
    _create_dataset_from_string(
        root_entry=nx_sample,
        name='name',
        var=dg['sample_name'].value,
    )
    _create_dataset_from_var(
        name='orientation_matrix',
        root_entry=nx_sample,
        var=sc.array(
            dims=['i', 'j'],
            values=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            unit="dimensionless",
        ),  # TODO: Add real data, the sample orientation matrix
    )
    _create_dataset_from_var(
        name='unit_cell',
        root_entry=nx_sample,
        var=sc.array(
            dims=['i'],
            values=[1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
            unit="dimensionless",  # TODO: Add real data,
            # a, b, c, alpha, beta, gamma
        ),
    )


def _add_lauetof_monitor_group(data: sc.DataGroup, nx_entry: h5py.Group) -> None:
    nx_monitor = nx_entry.create_group("control")
    nx_monitor.attrs["NX_class"] = "NXmonitor"
    _create_dataset_from_string(root_entry=nx_monitor, name='mode', var='monitor')
    nx_monitor["preset"] = 0.0  # Check if this is the correct value
    data_dset = _create_dataset_from_var(
        name='data',
        root_entry=nx_monitor,
        var=sc.array(
            dims=['tof'], values=[1, 1, 1], unit="counts"
        ),  # TODO: Add real data, bin values
    )
    data_dset.attrs["signal"] = 1
    data_dset.attrs["primary"] = 1
    _create_dataset_from_var(
        name='time_of_flight',
        root_entry=nx_monitor,
        var=sc.array(
            dims=['tof'], values=[1, 1, 1], unit="s"
        ),  # TODO: Add real data, bin edges
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


def _export_static_metadata_as_nxlauetof(
    experiment_metadata: NMXExperimentMetadata,
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
    experiment_metadata:
        Experiment metadata object.
    output_file:
        Output file path.
    arbitrary_metadata:
        Arbitrary metadata that does not fit into the existing metadata objects.

    """
    with h5py.File(output_file, "w") as f:
        f.attrs["NX_class"] = "NXlauetof"
        nx_entry = _create_lauetof_data_entry(f)
        _add_lauetof_definition(nx_entry)
        _add_lauetof_sample_group(experiment_metadata, nx_entry)
        nx_instrument = _add_lauetof_instrument(nx_entry)
        _add_lauetof_source_group(experiment_metadata, nx_instrument)
        # Placeholder for ``monitor`` group
        _add_lauetof_monitor_group(experiment_metadata, nx_entry)
        # Skipping ``NXdata``(name) field with data link
        # Add arbitrary metadata
        _add_arbitrary_metadata(nx_entry, **arbitrary_metadata)


def _export_detector_metadata_as_nxlauetof(
    *detector_metadatas: NMXDetectorMetadata,
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
        for detector_metadata in detector_metadatas:
            _add_lauetof_detector_group(detector_metadata, nx_instrument)


def _export_reduced_data_as_nxlauetof(
    dg: NMXReducedDataGroup,
    output_file: str | pathlib.Path | io.BytesIO,
    *,
    append_mode: bool = True,
    compress_counts: bool = True,
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
        nx_detector: h5py.Group = f[f"entry/instrument/{dg['detector_name'].value}"]
        # Data - shape: [n_x_pixels, n_y_pixels, n_tof_bins]
        # The actual application definition defines it as integer,
        # but we keep the original data type for now
        num_x, num_y = dg["detector_shape"].value  # Probably better way to do this
        if compress_counts:
            data_dset = _create_compressed_dataset(
                name="data",
                root_entry=nx_detector,
                var=sc.fold(
                    dg['counts'].data, dim='id', sizes={'x': num_x, 'y': num_y}
                ),
                chunks=(num_x, num_y, 1),
                dtype=np.uint,
            )
        else:
            data_dset = _create_dataset_from_var(
                name="data",
                root_entry=nx_detector,
                var=sc.fold(
                    dg['counts'].data, dim='id', sizes={'x': num_x, 'y': num_y}
                ),
                dtype=np.uint,
            )
        data_dset.attrs["signal"] = 1
        _create_dataset_from_var(
            name='time_of_flight',
            root_entry=nx_detector,
            var=sc.midpoints(dg['counts'].coords['t'], dim='t'),
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


T = TypeVar("T", bound=sc.DataArray)


class NXLauetofWriter:
    def __init__(
        self,
        *,
        output_filename: str | pathlib.Path | io.BytesIO,
        workflow: sl.Pipeline,
        chunk_generator: Callable[[FilePath, DetectorName], Generator[T, None, None]],
        chunk_insert_key: type[T],
        extra_meta: dict[str, sc.Variable] | None = None,
        compress_counts: bool = True,
        overwrite: bool = False,
    ) -> None:
        from ess.reduce.streaming import EternalAccumulator, StreamProcessor

        from .types import FilePath, NMXReducedCounts

        self.compress_counts = compress_counts
        self._chunk_generator = chunk_generator
        self._chunk_insert_key = chunk_insert_key
        self._workflow = workflow
        self._output_filename = _check_file(output_filename, overwrite)
        self._input_filename = workflow.compute(FilePath)
        self._final_stream_processor = partial(
            StreamProcessor,
            dynamic_keys=(chunk_insert_key,),
            target_keys=(NMXReducedDataGroup,),
            accumulators={NMXReducedCounts: EternalAccumulator},
        )
        self._detector_metas: dict[DetectorName, NMXDetectorMetadata] = {}
        self._detector_reduced: dict[DetectorName, NMXReducedDataGroup] = {}
        _export_static_metadata_as_nxlauetof(
            experiment_metadata=self._workflow.compute(NMXExperimentMetadata),
            output_file=self._output_filename,
            **(extra_meta or {}),
        )

    def add_panel(
        self, *, detector_id: DetectorIndex | DetectorName
    ) -> NMXReducedDataGroup:
        from .types import PixelIds

        temp_wf = self._workflow.copy()
        if isinstance(detector_id, int):
            temp_wf[DetectorIndex] = detector_id
        elif isinstance(detector_id, str):
            temp_wf[DetectorName] = detector_id
        else:
            raise TypeError(
                f"Expected detector_id to be an int or str, got {type(detector_id)}"
            )

        _export_detector_metadata_as_nxlauetof(
            temp_wf.compute(NMXDetectorMetadata),
            output_file=self._output_filename,
        )
        # First compute static information
        detector_name = temp_wf.compute(DetectorName)
        temp_wf[PixelIds] = temp_wf.compute(PixelIds)
        processor = self._final_stream_processor(temp_wf)
        # Then iterate over the chunks
        for da in self._chunk_generator(self._input_filename, detector_name):
            if any(da.sizes.values()) == 0:
                continue
            else:
                results = processor.add_chunk({self._chunk_insert_key: da})

        _export_reduced_data_as_nxlauetof(
            results[NMXReducedDataGroup],
            self._output_filename,
            compress_counts=self.compress_counts,
        )
        return results[NMXReducedDataGroup]
