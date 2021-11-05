# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Andrew R. McCluskey (arm61)

import numpy as np
import scipp as sc
import scippneutron as scn
from ess.reflectometry import corrections, resolution, binning, orso, write


def load(
        filename: str,
        sample_angle_offset: sc.Variable = 0 * sc.units.deg,
        gravity: bool = True,
        beam_size: sc.Variable = 0.001 * sc.units.m,
        sample_size: sc.Variable = 0.01 * sc.units.m,
        detector_spatial_resolution: sc.Variable = 0.0025 * sc.units.m) -> sc.DataArray:
    """
    The general reflectometry data loader.

    :param filename: The path to the file to be reduced.
    :param sample_angle_offset: Correction for omega or possibly misalignment of sample
        (Optional). Default is `0 degrees of arc`.
    :param gravity: Should gravity be accounted for (Optional). Default is `True`.
    :param beam_size: Size of the beam perpendicular to the scattering surface
        (Optional). Default is `0.001 m`.
    :param sample_size: Size of the sample in direction of the beam (Optional).
        Default is `0.01 m`.
    :param detector_spatial_resolution: Spatial resolution of the detector (Optional).
        Default is `2.5 mm`.
    """

    da = scn.load_nexus(filename)
    da.attrs["filename"] = sc.scalar(filename)

    # if isinstance(data, str):
    #     self.data_file = data
    #     self.data =
    # else:
    #     self.data_file = data_file
    #     self.data = data
    # self.data.bins.constituents["data"].variances = np.ones_like(
    #     self.data.bins.constituents["data"].values)
    da.attrs["sample_angle_offset"] = sample_angle_offset
    # da.attrs["gravity"] = sc.scalar(gravity)
    da.attrs["beam_size"] = beam_size
    da.attrs["sample_size"] = sample_size
    da.attrs["detector_spatial_resolution"] = detector_spatial_resolution
    # self.orso = orso.Orso(orso.Creator(name='scipp'), orso.DataSource(),
    #                       orso.Reduction(), [])
    # experiment = orso.Experiment(self.data.attrs['instrument_name'].value, 'neutron')
    # self.orso.data_source.experiment = experiment
    return da
