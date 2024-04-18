# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import io
import pathlib
from typing import Optional, Union

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


def _join_variables(*vars: sc.Variable, splitter: str = " ") -> sc.Variable:
    # Check if all variables are integer
    if not all(var.dtype == int for var in vars):
        raise ValueError("All variables must be integer type.")
    # Check if all variables have the same dimensions
    dims = set(var.dim for var in vars)
    if len(dims) != 1:
        raise ValueError("All variables must have the same dimensions.")
    # Check if all variables have the same length
    lengths = set(len(var.values) for var in vars)
    if len(lengths) != 1:
        raise ValueError("All variables must have the same length.")

    return sc.array(
        dims=dims,
        values=[
            splitter.join(str(val) for val in row)
            for row in zip(*(var.values for var in vars))
        ],
    )


def _split_variable(var: sc.Variable, splitter: str = " ") -> tuple[sc.Variable, ...]:
    if var.dtype != str:
        raise ValueError("The variable must be string type.")
    separated = [val.split(splitter) for val in var.values]
    # Check if all rows have the same length
    lengths = set(len(row) for row in separated)
    if len(lengths) != 1:
        raise ValueError("All rows must have the same length.")

    return tuple(
        sc.array(dims=var.dims, values=[int(row[i]) for row in separated], dtype=int)
        for i in range(lengths.pop())
    )


def _zip_and_group(da: sc.DataArray, /, *args: str | sc.Variable) -> sc.DataArray:
    """Group the data array by the given coordinates.

    It is a temporary solution before we fix the issue in scipp.
    It should be replaced with the scipp group function once it's possible
    to group by string-type coordinates or ``tuple[int]`` type of coordinates.
    See [#3046](https://github.com/scipp/scipp/issues/3046) and
    [#3425](https://github.com/scipp/scipp/issues/3425) for more details.

    Parameters
    ----------
    da:
        The data array to group.
    args:
        The coordinates to group by.
    group_detour_func_map:
        The conversion functions.

    Returns
    -------
    sc.DataArray
        The grouped data array.

    """
    copied = da.copy(deep=False)
    group_coord_names = tuple(arg if isinstance(arg, str) else arg.dim for arg in args)
    tmp_str_coord_name = "".join(group_coord_names)
    group_coords = tuple(
        copied.coords[name] for name in group_coord_names
    )  # Must keep the order

    tmp_coord = _join_variables(*group_coords)
    copied.coords[tmp_str_coord_name] = tmp_coord

    if all(isinstance(arg, str) for arg in args):
        group_var = sc.array(
            dims=[tmp_str_coord_name],
            values=np.unique(copied.coords[tmp_str_coord_name].values),
        )
    elif all(isinstance(arg, sc.Variable) for arg in args):
        group_var = _join_variables(
            *[arg.rename_dims({arg.dim: tmp_str_coord_name}) for arg in args]
        )
    else:
        raise ValueError("All coordinates must be either str or sc.Variable.")

    grouped = copied.group(group_var)
    real_coords = _split_variable(grouped.coords[tmp_str_coord_name])
    for i_coord, name in enumerate(group_coord_names):
        grouped.coords[name] = real_coords[i_coord].rename_dims(
            {real_coords[i_coord].dim: grouped.dim}
        )

    return grouped.drop_coords([tmp_str_coord_name])
