# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pathlib
from typing import NewType, Union

import h5py
import scipp as sc

TimeBinSteps = NewType("TimeBinSteps", int)


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
    def proton_charge(self) -> float:
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
        """Event data grouped by pixel id and panel."""
        return self['weights']


class NMXReducedData(_SharedFields, sc.DataGroup):
    @property
    def counts(self) -> sc.DataArray:
        """Binned time of arrival data from flattened event data."""
        return self['counts']

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
        crystal_rotation = nx_sample.create_dataset(
            'crystal_rotation', data=self.crystal_rotation.values
        )
        crystal_rotation.attrs["units"] = str(self.crystal_rotation.unit)
        crystal_rotation.attrs["long_name"] = 'crystal rotation in Phi (XYZ)'

        return nx_sample

    def _create_compressed_dataset(
        self, nx_entry: h5py.Group, name: str, var: sc.Variable, *, long_name: str
    ) -> h5py.Dataset:
        dataset = nx_entry.create_dataset(
            name,
            data=var.values,
            compression="gzip",
            compression_opts=4,
        )
        dataset.attrs["units"] = str(var.unit)
        dataset.attrs["long_name"] = name
        return dataset

    def _create_instrument_group(self, nx_entry: h5py.Group) -> h5py.Group:
        nx_instrument = nx_entry.create_group("NXinstrument")
        nx_instrument.attrs["nr_detector"] = self.origin_position.sizes['panel']
        nx_instrument.create_dataset("proton_charge", data=self.proton_charge)

        nx_detector_1 = nx_instrument.create_group("detector_1")
        counts = nx_detector_1.create_dataset(
            "counts", data=[self.counts.values], compression="gzip", compression_opts=4
        )
        counts.attrs["units"] = "counts"
        t_spectrum = nx_detector_1.create_dataset(
            "t_bin",
            data=self.counts.coords['t'].values,
            compression="gzip",
            compression_opts=4,
        )
        t_spectrum.attrs["units"] = "s"
        t_spectrum.attrs["long_name"] = "t_bin TOF (ms)"
        pixel_id = nx_detector_1.create_dataset(
            "pixel_id",
            data=self.counts.coords['id'].values,
            compression="gzip",
            compression_opts=4,
        )
        pixel_id.attrs["units"] = ""
        pixel_id.attrs["long_name"] = "pixel ID"
        return nx_instrument

    def _create_detector_group(self, nx_entry: h5py.Group) -> h5py.Group:
        nx_detector = nx_entry.create_group("NXdetector")
        # Position of the first pixel (lowest ID) in the detector
        detector_origins = nx_detector.create_dataset(
            "origin",
            data=self.origin_position.values,
            compression="gzip",
            compression_opts=4,
        )
        detector_origins.attrs["units"] = "m"
        # Fast axis, along where the pixel ID increases by 1
        nx_detector.create_dataset("fast_axis", data=self.fast_axis.values)
        # Slow axis, along where the pixel ID increases
        # by the number of pixels in the fast axis
        nx_detector.create_dataset("slow_axis", data=self.slow_axis.values)
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

    def export_as_nexus(self, output_file_name: Union[str, pathlib.Path]) -> None:
        import h5py

        file_name = pathlib.Path(output_file_name)
        if file_name.suffix not in (".h5", ".nxs"):
            raise ValueError("Output file name must end with .h5 or .nxs")

        with h5py.File(file_name, "w") as out_file:
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
    nmx_data: NMXData, time_bin_step: TimeBinSteps
) -> NMXReducedData:
    """Bin time of arrival data into ``time_bin_step`` bins."""

    return NMXReducedData(
        counts=nmx_data.weights.flatten(dims=['panel', 'id'], to='id').hist(
            t=time_bin_step
        ),
        **{key: nmx_data[key] for key in nmx_data.keys() if key != 'weights'},
    )
