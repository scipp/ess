# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ...choppers import make_chopper


def make_beamline() -> dict:
    """
    V20 chopper cascade and component positions.
    Chopper opening angles taken from Woracek et al. (2016)
    https://doi.org/10.1016/j.nima.2016.09.034

    The +15.0 increments added to the angles correspond to an offset between the
    zero angle and the chopper top-dead center.
    """

    dim = 'frame'

    beamline = {
        "source_pulse_length": sc.scalar(2.86e+03, unit='us'),
        "source_pulse_t_0": sc.scalar(140.0, unit='us'),
        "source_position": sc.vector(value=[0.0, 0.0, -25.3], unit='m')
    }

    beamline["chopper_wfm_1"] = sc.scalar(
        make_chopper(
            frequency=sc.scalar(70.0, unit="Hz"),
            phase=sc.scalar(47.10, unit='deg'),
            position=beamline["source_position"] +
            sc.vector(value=[0, 0, 6.6], unit='m'),
            cutout_angles_begin=sc.array(
                dims=[dim],
                values=np.array([83.71, 140.49, 193.26, 242.32, 287.91, 330.3]) + 15.0,
                unit='deg'),
            cutout_angles_end=sc.array(
                dims=[dim],
                values=np.array([94.7, 155.79, 212.56, 265.33, 314.37, 360.0]) + 15.0,
                unit='deg'),
            kind=sc.scalar('wfm')))

    beamline["chopper_wfm_2"] = sc.scalar(
        make_chopper(
            frequency=sc.scalar(70.0, unit="Hz"),
            phase=sc.scalar(76.76, unit='deg'),
            position=beamline["source_position"] +
            sc.vector(value=[0, 0, 7.1], unit='m'),
            cutout_angles_begin=sc.array(
                dims=[dim],
                values=np.array([65.04, 126.1, 182.88, 235.67, 284.73, 330.32]) + 15.0,
                unit='deg'),
            cutout_angles_end=sc.array(
                dims=[dim],
                values=np.array([76.03, 141.4, 202.18, 254.97, 307.74, 360.0]) + 15.0,
                unit='deg'),
            kind=sc.scalar('wfm')))

    beamline["chopper_foc_1"] = sc.scalar(
        make_chopper(
            frequency=sc.scalar(56.0, unit="Hz"),
            phase=sc.scalar(62.40, unit='deg'),
            position=beamline["source_position"] +
            sc.vector(value=[0, 0, 8.8], unit='m'),
            cutout_angles_begin=sc.array(
                dims=[dim],
                values=np.array([64.35, 125.05, 183.41, 236.4, 287.04, 335.53]) + 15.0,
                unit='deg'),
            cutout_angles_end=sc.array(
                dims=[dim],
                values=np.array([84.99, 148.29, 205.22, 254.27, 302.8, 360.0]) + 15.0,
                unit='deg'),
            kind=sc.scalar("frame_overlap")))
    beamline["chopper_foc_2"] = sc.scalar(
        make_chopper(
            frequency=sc.scalar(28.0, unit="Hz"),
            phase=sc.scalar(12.27, unit='deg'),
            position=beamline["source_position"] +
            sc.vector(value=[0, 0, 15.9], unit='m'),
            cutout_angles_begin=sc.array(
                dims=[dim],
                values=np.array([79.78, 136.41, 191.73, 240.81, 287.13, 330.89]) + 15.0,
                unit='deg'),
            cutout_angles_end=sc.array(
                dims=[dim],
                values=np.array([116.38, 172.47, 221.94, 267.69, 311.69, 360.0]) + 15.0,
                unit='deg'),
            kind=sc.scalar("frame_overlap")))

    return beamline
