# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from scipp.constants import h, m_n

from ..choppers import make_chopper


def make_fake_beamline(chopper_wfm_1_position=None,
                       chopper_wfm_2_position=None,
                       frequency=None,
                       lambda_min=None,
                       pulse_length=None,
                       pulse_t_0=None,
                       nframes=2):
    """
    Fake chopper cascade with 2 optically blind WFM choppers.
    Based on mathematical description in Schmakat et al. (2020);
    https://www.sciencedirect.com/science/article/pii/S0168900220308640
    """
    kwargs = dict(
        chopper_wfm_1_position=chopper_wfm_1_position,
        chopper_wfm_2_position=chopper_wfm_2_position,
        frequency=frequency,
        lambda_min=lambda_min,
        pulse_length=pulse_length,
        pulse_t_0=pulse_t_0,
    )
    if kwargs['chopper_wfm_1_position'] is None:
        kwargs['chopper_wfm_1_position'] = sc.vector(value=[0.0, 0.0, 6.775], unit='m')
    if kwargs['chopper_wfm_2_position'] is None:
        kwargs['chopper_wfm_2_position'] = sc.vector(value=[0.0, 0.0, 7.225], unit='m')
    if kwargs['frequency'] is None:
        kwargs['frequency'] = sc.scalar(56.0, unit=sc.units.one / sc.units.s)
    if kwargs['lambda_min'] is None:
        kwargs['lambda_min'] = sc.scalar(1.0, unit='angstrom')
    if kwargs['pulse_length'] is None:
        kwargs['pulse_length'] = sc.scalar(2.86e-03, unit='s')
    if kwargs['pulse_t_0'] is None:
        kwargs['pulse_t_0'] = sc.scalar(1.3e-4, unit='s')
    return _make_fake_beamline_impl(nframes=nframes, **kwargs)


def _make_fake_beamline_impl(chopper_wfm_1_position, chopper_wfm_2_position, frequency,
                             lambda_min, pulse_length, pulse_t_0, nframes):
    dim = 'frame'
    # Neutron mass to Planck constant ratio
    alpha = sc.to_unit(m_n / h, 's/m/angstrom')
    omega = (2.0 * np.pi * sc.units.rad) * frequency

    cutout_angles_center_wfm_1 = sc.empty(dims=[dim], shape=[nframes], unit='rad')
    cutout_angles_center_wfm_2 = sc.empty_like(cutout_angles_center_wfm_1)
    cutout_angles_width = sc.empty_like(cutout_angles_center_wfm_1)

    for i in range(nframes):
        # Equation (3) in Schmakat et al. (2020)
        lambda_max = (pulse_length +
                      alpha * lambda_min * sc.norm(chopper_wfm_1_position)) / (
                          alpha * sc.norm(chopper_wfm_2_position))
        # Equation (4) in Schmakat et al. (2020)
        theta = omega * (pulse_length + alpha *
                         (lambda_min - lambda_max) * sc.norm(chopper_wfm_1_position))
        # Equation (5) in Schmakat et al. (2020)
        phi_wfm_1 = omega * (
            pulse_t_0 + 0.5 * pulse_length + 0.5 * alpha *
            (lambda_min + lambda_max) * sc.norm(chopper_wfm_1_position))
        # Equation (6) in Schmakat et al. (2020)
        phi_wfm_2 = omega * (pulse_t_0 + 1.5 * pulse_length + 0.5 * alpha * (
            (3.0 * lambda_min) - lambda_max) * sc.norm(chopper_wfm_1_position))

        cutout_angles_width[dim, i] = theta
        cutout_angles_center_wfm_1[dim, i] = phi_wfm_1
        cutout_angles_center_wfm_2[dim, i] = phi_wfm_2

        lambda_min = lambda_max

    return {
        "chopper_wfm_1":
        sc.scalar(
            make_chopper(frequency=frequency,
                         phase=sc.scalar(0.0, unit='deg'),
                         position=chopper_wfm_1_position,
                         cutout_angles_center=cutout_angles_center_wfm_1,
                         cutout_angles_width=cutout_angles_width,
                         kind=sc.scalar('wfm'))),
        "chopper_wfm_2":
        sc.scalar(
            make_chopper(frequency=frequency,
                         phase=sc.scalar(0.0, unit='deg'),
                         position=chopper_wfm_2_position,
                         cutout_angles_center=cutout_angles_center_wfm_2,
                         cutout_angles_width=cutout_angles_width,
                         kind=sc.scalar('wfm'))),
        'position':
        sc.vector(value=[0., 0., 60.], unit='m'),
        "source_pulse_length":
        sc.to_unit(pulse_length, 'us'),
        "source_pulse_t_0":
        sc.to_unit(pulse_t_0, 'us'),
        "source_position":
        sc.vector(value=[0.0, 0.0, 0.0], unit='m')
    }
