# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import io
import pathlib
from typing import Any, Callable, Optional, Sequence, Union

import h5py
import numpy as np
import sciline
import scipp as sc

from .const import DETECTOR_DIM, PIXEL_DIM, TOF_DIM
from .mcstas_xml import McStasInstrument
from .types import DetectorIndex, DetectorName, TimeBinSteps


class _SharedFields(sc.DataGroup):
    """All shared fields between NMXData and NMXReducedData.

    ``weights`` is only present in NMXData due to potential memory issues.
    """

    @property
    def origin_position(self) -> sc.Variable:
        """Position of the first pixel (lowest ID) in the detector.

        Relative position from the sample."""
        return self['origin_position']

    @property
    def crystal_rotation(self) -> sc.Variable:
        """Rotation of the crystal."""

        return self['crystal_rotation']

    @property
    def sample_name(self) -> sc.Variable:
        return self['sample_name']

    @property
    def fast_axis(self) -> sc.Variable:
        """Fast axis, along where the pixel ID increases by 1."""
        return self['fast_axis']

    @property
    def slow_axis(self) -> sc.Variable:
        """Slow axis, along where the pixel ID increases by > 1.

        The pixel ID increases by the number of pixels in the fast axis."""
        return self['slow_axis']

    @property
    def proton_charge(self) -> sc.Variable:
        """Accumulated number of protons during the measurement."""
        return self['proton_charge']

    @property
    def source_position(self) -> sc.Variable:
        """Relative position of the source from the sample."""
        return self['source_position']

    @property
    def sample_position(self) -> sc.Variable:
        """Relative position of the sample from the sample. (0, 0, 0)"""
        return self['sample_position']


class NMXData(_SharedFields, sc.DataGroup):
    @property
    def weights(self) -> sc.DataArray:
        """Event data grouped by pixel id."""
        return self['weights']


class NMXReducedData(_SharedFields, sc.DataGroup):
    @property
    def counts(self) -> sc.DataArray:
        """Binned time of arrival data from flattened event data."""
        return self['counts']

    def _create_dataset_from_var(
        self,
        *,
        root_entry: h5py.Group,
        var: sc.Variable,
        name: str,
        long_name: Optional[str] = None,
        compression: Optional[str] = None,
        compression_opts: Optional[int] = None,
    ) -> h5py.Dataset:
        compression_options = dict()
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

    def _create_compressed_dataset(
        self,
        *,
        root_entry: h5py.Group,
        name: str,
        var: sc.Variable,
        long_name: Optional[str] = None,
    ) -> h5py.Dataset:
        return self._create_dataset_from_var(
            root_entry=root_entry,
            var=var,
            name=name,
            long_name=long_name,
            compression="gzip",
            compression_opts=4,
        )

    def _create_root_data_entry(self, file_obj: h5py.File) -> h5py.Group:
        nx_entry = file_obj.create_group("NMX_data")
        nx_entry.attrs["NX_class"] = "NXentry"
        nx_entry.attrs["default"] = "data"
        nx_entry.attrs["name"] = "NMX"
        nx_entry["name"] = "NMX"
        nx_entry["definition"] = "TOFRAW"
        return nx_entry

    def _create_sample_group(self, nx_entry: h5py.Group) -> h5py.Group:
        nx_sample = nx_entry.create_group("NXsample")
        nx_sample["name"] = self.sample_name.value
        # Crystal rotation
        self._create_dataset_from_var(
            root_entry=nx_sample,
            var=self.crystal_rotation,
            name='crystal_rotation',
            long_name='crystal rotation in Phi (XYZ)',
        )
        return nx_sample

    def _create_instrument_group(self, nx_entry: h5py.Group) -> h5py.Group:
        nx_instrument = nx_entry.create_group("NXinstrument")
        nx_instrument.create_dataset("proton_charge", data=self.proton_charge.values)

        nx_detector_1 = nx_instrument.create_group("detector_1")
        # Detector counts
        self._create_compressed_dataset(
            root_entry=nx_detector_1,
            name="counts",
            var=self.counts,
        )
        # Time of arrival bin edges
        self._create_dataset_from_var(
            root_entry=nx_detector_1,
            var=self.counts.coords[TOF_DIM],
            name="t_bin",
            long_name="t_bin TOF (ms)",
        )
        # Pixel IDs
        self._create_compressed_dataset(
            root_entry=nx_detector_1,
            name="pixel_id",
            var=self.counts.coords[PIXEL_DIM],
            long_name="pixel ID",
        )
        return nx_instrument

    def _create_detector_group(self, nx_entry: h5py.Group) -> h5py.Group:
        nx_detector = nx_entry.create_group("NXdetector")
        # Position of the first pixel (lowest ID) in the detector
        self._create_compressed_dataset(
            root_entry=nx_detector,
            name="origin",
            var=self.origin_position,
        )
        # Fast axis, along where the pixel ID increases by 1
        self._create_dataset_from_var(
            root_entry=nx_detector, var=self.fast_axis, name="fast_axis"
        )
        # Slow axis, along where the pixel ID increases
        # by the number of pixels in the fast axis
        self._create_dataset_from_var(
            root_entry=nx_detector, var=self.slow_axis, name="slow_axis"
        )
        return nx_detector

    def _create_source_group(self, nx_entry: h5py.Group) -> h5py.Group:
        nx_source = nx_entry.create_group("NXsource")
        nx_source["name"] = "European Spallation Source"
        nx_source["short_name"] = "ESS"
        nx_source["type"] = "Spallation Neutron Source"
        nx_source["distance"] = sc.norm(self.source_position).value
        nx_source["probe"] = "neutron"
        nx_source["target_material"] = "W"
        return nx_source

    def export_as_nexus(
        self, output_file_base: Union[str, pathlib.Path, io.BytesIO]
    ) -> None:
        """Export the reduced data to a NeXus file.

        Currently exporting step is not expected to be part of sciline pipelines.
        """
        if isinstance(output_file_base, (str, pathlib.Path)):
            file_base = pathlib.Path(output_file_base)
            if file_base.suffix not in (".h5", ".nxs"):
                raise ValueError("Output file name must end with .h5 or .nxs")
        else:
            file_base = output_file_base

        with h5py.File(file_base, "w") as out_file:
            out_file.attrs["default"] = "NMX_data"
            # Root Data Entry
            nx_entry = self._create_root_data_entry(out_file)
            # Sample
            self._create_sample_group(nx_entry)
            # Instrument
            self._create_instrument_group(nx_entry)
            # Detector
            self._create_detector_group(nx_entry)
            # Source
            self._create_source_group(nx_entry)


def bin_time_of_arrival(
    nmx_data: sciline.Series[DetectorIndex, NMXData],
    detector_name: sciline.Series[DetectorIndex, DetectorName],
    instrument: McStasInstrument,
    time_bin_step: TimeBinSteps,
) -> NMXReducedData:
    """Bin time of arrival data into ``time_bin_step`` bins."""

    nmx_data = list(nmx_data.values())
    nmx_data = sc.concat(nmx_data, DETECTOR_DIM)
    counts = nmx_data.pop('weights').hist(t=time_bin_step)
    new_coords = instrument.to_coords(*detector_name.values())
    new_coords.pop('pixel_id')

    return NMXReducedData(
        counts=counts,
        **{**nmx_data, **new_coords},
    )


def _apply_elem_wise(
    func: Callable, var: sc.Variable, *, result_dtype: Any = None
) -> sc.Variable:
    """Apply a function element-wise to the variable values.

    This helper is only for vector-dtype variables.
    Use ``numpy.vectorize`` for other types.

    Parameters
    ----------
    func:
        The function to apply.
    var:
        The variable to apply the function to.
    result_dtype:
        The dtype of the resulting variable.
        It is needed especially when the function returns a vector.

    """

    def apply_func(val: Sequence, _cur_depth: int = 0) -> list:
        if _cur_depth == len(var.dims):
            return func(val)
        return [apply_func(v, _cur_depth + 1) for v in val]

    if result_dtype is None:
        return sc.Variable(
            dims=var.dims,
            values=apply_func(var.values),
        )
    return sc.Variable(
        dims=var.dims,
        values=apply_func(var.values),
        dtype=result_dtype,
    )


def _detour_group(
    da: sc.DataArray, group_name: str, detour_func: Callable
) -> sc.DataArray:
    """Group the data array by a hash of a coordinate.

    It uses index of each unique hash value
    for grouping instead of hash value itself
    to avoid overflow issues.

    """
    from uuid import uuid4

    copied = da.copy(deep=False)

    # Temporary coords for grouping
    detour_idx_coord_name = uuid4().hex + "hash_idx"

    # Create a temporary detoured coordinate
    detour_var = _apply_elem_wise(detour_func, da.coords[group_name])
    # Create a temporary hash-index of each unique value
    unique_hashes = np.unique(detour_var.values)
    hash_to_idx = {hash_val: idx for idx, hash_val in enumerate(unique_hashes)}
    copied.coords[detour_idx_coord_name] = _apply_elem_wise(
        lambda idx: hash_to_idx[idx], detour_var
    )

    # Group by the hash-index
    grouped = copied.group(detour_idx_coord_name)

    # Restore the original values
    idx_to_detour = {idx: hash_val for hash_val, idx in hash_to_idx.items()}
    detour_to_var = {
        hash_val: var
        for var, hash_val in zip(da.coords[group_name].values, detour_var.values)
    }
    idx_to_var = {
        idx: detour_to_var[hash_val] for idx, hash_val in idx_to_detour.items()
    }
    grouped.coords[group_name] = _apply_elem_wise(
        lambda idx: idx_to_var[idx],
        grouped.coords[detour_idx_coord_name],
        result_dtype=da.coords[group_name].dtype,
    )
    # Rename dims back to group_name and drop the temporary hash-index coordinate
    return grouped.rename_dims({detour_idx_coord_name: group_name}).drop_coords(
        [detour_idx_coord_name]
    )


def _group(da: sc.DataArray, /, *args: str, **group_detour_func_map) -> sc.DataArray:
    """Group the data array by the given coordinates.

    Parameters
    ----------
    da:
        The data array to group.
    args:
        The coordinates to group by.
    group_hash_func_map:
        The hash functions for each coordinate.

    Returns
    -------
    sc.DataArray
        The grouped data array.

    """
    grouped = da
    for group_name in args:
        if group_name in group_detour_func_map:
            grouped = _detour_group(
                grouped, group_name, group_detour_func_map[group_name]
            )
        else:
            try:
                grouped = sc.group(grouped, group_name)
            except Exception:
                grouped = _detour_group(
                    grouped, group_name, group_detour_func_map.get(group_name, hash)
                )

    return grouped
