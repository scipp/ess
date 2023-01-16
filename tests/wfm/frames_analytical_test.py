# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp import constants

import ess.choppers as ch
import ess.wfm as wfm


def _frames_from_slopes(data):
    detector_pos_norm = sc.norm(data.meta["position"])

    # Get the number of WFM frames
    choppers = {key: data.meta[key].value for key in ch.find_chopper_keys(data)}
    nframes = ch.cutout_angles_begin(choppers["chopper_wfm_1"]).sizes["frame"]

    # Now find frame boundaries
    frames = sc.Dataset()
    frames["time_min"] = sc.zeros(dims=["frame"], shape=[nframes], unit=sc.units.us)
    frames["time_max"] = sc.zeros_like(frames["time_min"])
    frames["delta_time_min"] = sc.zeros_like(frames["time_min"])
    frames["delta_time_max"] = sc.zeros_like(frames["time_min"])
    frames["wavelength_min"] = sc.zeros(dims=["frame"],
                                        shape=[nframes],
                                        unit=sc.units.angstrom)
    frames["wavelength_max"] = sc.zeros_like(frames["wavelength_min"])
    frames["delta_wavelength_min"] = sc.zeros_like(frames["wavelength_min"])
    frames["delta_wavelength_max"] = sc.zeros_like(frames["wavelength_min"])

    frames["time_correction"] = sc.zeros(dims=["frame"],
                                         shape=[nframes],
                                         unit=sc.units.us)

    near_wfm_chopper = choppers["chopper_wfm_1"]
    far_wfm_chopper = choppers["chopper_wfm_2"]

    # Distance between WFM choppers
    dz_wfm = sc.norm(far_wfm_chopper["position"].data -
                     near_wfm_chopper["position"].data)
    # Mid-point between WFM choppers
    z_wfm = 0.5 * (near_wfm_chopper["position"].data + far_wfm_chopper["position"].data)
    # Distance between detector positions and wfm chopper mid-point
    zdet_minus_zwfm = sc.norm(data.meta["position"] - z_wfm)
    # Neutron mass to Planck constant ratio
    alpha = sc.to_unit(constants.m_n / constants.h, 'us/m/angstrom')

    near_t_open = ch.time_open(near_wfm_chopper)
    near_t_close = ch.time_closed(near_wfm_chopper)
    far_t_open = ch.time_open(far_wfm_chopper)

    for i in range(nframes):
        dt_lambda_max = near_t_close["frame", i] - near_t_open["frame", i]
        slope_lambda_max = dz_wfm / dt_lambda_max
        intercept_lambda_max = sc.norm(near_wfm_chopper["position"].data
                                       ) - slope_lambda_max * near_t_close["frame", i]
        t_lambda_max = (detector_pos_norm - intercept_lambda_max) / slope_lambda_max

        slope_lambda_min = sc.norm(near_wfm_chopper["position"].data) / (
            near_t_close["frame", i] -
            (data.meta["source_pulse_length"] + data.meta["source_pulse_t_0"]))
        intercept_lambda_min = sc.norm(far_wfm_chopper["position"].data
                                       ) - slope_lambda_min * far_t_open["frame", i]
        t_lambda_min = (detector_pos_norm - intercept_lambda_min) / slope_lambda_min

        t_lambda_min_plus_dt = (
            detector_pos_norm -
            (sc.norm(near_wfm_chopper["position"].data) -
             slope_lambda_min * near_t_close["frame", i])) / slope_lambda_min
        dt_lambda_min = t_lambda_min_plus_dt - t_lambda_min

        # Compute wavelength information
        lambda_min = (t_lambda_min + 0.5 * dt_lambda_min -
                      far_t_open["frame", i]) / (alpha * zdet_minus_zwfm)
        lambda_max = (t_lambda_max - 0.5 * dt_lambda_max -
                      far_t_open["frame", i]) / (alpha * zdet_minus_zwfm)
        dlambda_min = dz_wfm * lambda_min / zdet_minus_zwfm
        dlambda_max = dz_wfm * lambda_max / zdet_minus_zwfm

        frames["time_min"]["frame", i] = t_lambda_min
        frames["delta_time_min"]["frame", i] = dt_lambda_min
        frames["time_max"]["frame", i] = t_lambda_max
        frames["delta_time_max"]["frame", i] = dt_lambda_max
        frames["wavelength_min"]["frame", i] = lambda_min
        frames["wavelength_max"]["frame", i] = lambda_max
        frames["delta_wavelength_min"]["frame", i] = dlambda_min
        frames["delta_wavelength_max"]["frame", i] = dlambda_max
        frames["time_correction"]["frame", i] = far_t_open["frame", i]

    frames["wfm_chopper_mid_point"] = z_wfm
    return frames


def _check_against_reference(ds, frames):
    reference = _frames_from_slopes(ds)
    for key in frames:
        assert sc.allclose(reference[key].data, frames[key].data)
    for i in range(frames.sizes['frame'] - 1):
        assert sc.allclose(frames["delta_time_max"]["frame", i].data,
                           frames["delta_time_min"]["frame", i + 1].data)


def test_frames_analytical():
    ds = sc.Dataset(coords=wfm.make_fake_beamline())
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_large_dz_wfm():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        chopper_wfm_1_position=sc.vector(value=[0.0, 0.0, 6.0], unit='m'),
        chopper_wfm_2_position=sc.vector(value=[0.0, 0.0, 8.0], unit='m')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_short_pulse():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        pulse_length=sc.to_unit(sc.scalar(1.86e+03, unit='us'), 's')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_large_t_0():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        pulse_t_0=sc.to_unit(sc.scalar(300., unit='us'), 's')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_6_frames():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(nframes=6))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)


def test_frames_analytical_short_lambda_min():
    ds = sc.Dataset(coords=wfm.make_fake_beamline(
        lambda_min=sc.scalar(0.5, unit='angstrom')))
    frames = wfm.get_frames(ds)
    _check_against_reference(ds, frames)
