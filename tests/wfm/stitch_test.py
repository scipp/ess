# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import ess.wfm as wfm
import ess.choppers as ch
import scipp as sc
from scipp import constants
import scippneutron as scn
import pytest


def test_basic_stitching():
    frames = sc.Dataset()
    shift = -5.0
    frames['time_min'] = sc.array(dims=['frame'], values=[0.0], unit=sc.units.us)
    frames['time_max'] = sc.array(dims=['frame'], values=[10.0], unit=sc.units.us)
    frames['time_correction'] = sc.array(dims=['frame'],
                                         values=[shift],
                                         unit=sc.units.us)
    frames["wfm_chopper_mid_point"] = sc.vector(value=[0., 0., 2.0], unit='m')

    data = sc.DataArray(data=sc.ones(dims=['t'], shape=[100], unit=sc.units.counts),
                        coords={
                            't':
                            sc.linspace(dim='t',
                                        start=0.0,
                                        stop=10.0,
                                        num=101,
                                        unit=sc.units.us),
                            'source_position':
                            sc.vector(value=[0., 0., 0.], unit='m')
                        })

    nbins = 10
    stitched = wfm.stitch(data=data, dim='t', frames=frames, bins=nbins)
    # Note dimension change to TOF as well as shift
    assert sc.identical(
        sc.values(stitched),
        sc.DataArray(data=sc.ones(dims=['tof'], shape=[nbins], unit=sc.units.counts) *
                     nbins,
                     coords={
                         'tof':
                         sc.linspace(dim='tof',
                                     start=0.0 - shift,
                                     stop=10.0 - shift,
                                     num=nbins + 1,
                                     unit=sc.units.us),
                         'source_position':
                         sc.vector(value=[0., 0., 2.], unit='m')
                     }))


def _do_stitching_on_beamline(wavelengths, dim, event_mode=False):
    # Make beamline parameters for 6 frames
    coords = wfm.make_fake_beamline(nframes=6)

    # They are all created half-way through the pulse.
    # Compute their arrival time at the detector.
    alpha = sc.to_unit(constants.m_n / constants.h, 's/m/angstrom')
    dz = sc.norm(coords['position'] - coords['source_position'])
    arrival_times = sc.to_unit(
        alpha * dz * wavelengths,
        'us') + coords['source_pulse_t_0'] + (0.5 * coords['source_pulse_length'])
    coords[dim] = arrival_times

    # Make a data array that contains the beamline and the time coordinate
    tmin = sc.min(arrival_times)
    tmax = sc.max(arrival_times)
    dt = 0.1 * (tmax - tmin)

    if event_mode:
        num = 2
    else:
        num = 2001
    time_binning = sc.linspace(dim=dim,
                               start=(tmin - dt).value,
                               stop=(tmax + dt).value,
                               num=num,
                               unit=dt.unit)
    events = sc.DataArray(data=sc.ones(dims=['event'],
                                       shape=arrival_times.shape,
                                       unit=sc.units.counts,
                                       with_variances=True),
                          coords=coords)
    if event_mode:
        da = events.bin({dim: time_binning})
    else:
        da = events.hist({dim: time_binning})

    # Find location of frames
    frames = wfm.get_frames(da)

    stitched = wfm.stitch(frames=frames, data=da, dim=dim, bins=2001)

    wav = scn.convert(stitched, origin='tof', target='wavelength', scatter=False)
    if event_mode:
        out = wav
    else:
        out = wav.rebin(wavelength=sc.linspace(
            dim='wavelength', start=1.0, stop=10.0, num=1001, unit='angstrom'))

    choppers = {key: da.meta[key].value for key in ch.find_chopper_keys(da)}
    # Distance between WFM choppers
    dz_wfm = sc.norm(choppers["chopper_wfm_2"]["position"].data -
                     choppers["chopper_wfm_1"]["position"].data)
    # Delta_lambda  / lambda
    dlambda_over_lambda = dz_wfm / sc.norm(coords['position'] -
                                           frames['wfm_chopper_mid_point'].data)

    return out, dlambda_over_lambda


def _check_lambda_inside_resolution(lam,
                                    dlam_over_lam,
                                    data,
                                    event_mode=False,
                                    check_value=True):
    dlam = 0.5 * dlam_over_lam * lam
    if event_mode:
        sum_in_range = data.bin(
            wavelength=sc.array(dims=['wavelength'],
                                values=[(lam - dlam).value, (lam + dlam).value],
                                unit=lam.unit)).hist().data['wavelength', 0]
    else:
        sum_in_range = sc.sum(data['wavelength', lam - dlam:lam + dlam]).data
    assert sc.isclose(sum_in_range, 1.0 * sc.units.counts).value is check_value


@pytest.mark.parametrize("dim", ['time', 'tof'])
@pytest.mark.parametrize("event_mode", [False, True])
def test_stitching_on_beamline(event_mode, dim):
    wavelengths = sc.array(dims=['event'],
                           values=[1.75, 3.2, 4.5, 6.0, 7.0, 8.25],
                           unit='angstrom')
    stitched, dlambda_over_lambda = _do_stitching_on_beamline(wavelengths,
                                                              dim=dim,
                                                              event_mode=event_mode)

    for i in range(len(wavelengths)):
        _check_lambda_inside_resolution(wavelengths['event', i],
                                        dlambda_over_lambda,
                                        stitched,
                                        event_mode=event_mode)


@pytest.mark.parametrize("dim", ['time', 'tof'])
@pytest.mark.parametrize("event_mode", [False, True])
def test_stitching_on_beamline_bad_wavelength(event_mode, dim):
    # Create 6 neutrons. The first wavelength is in this case too short to pass through
    # the WFM choppers.
    wavelengths = sc.array(dims=['event'],
                           values=[1.5, 3.2, 4.5, 6.0, 7.0, 8.25],
                           unit='angstrom')
    stitched, dlambda_over_lambda = _do_stitching_on_beamline(wavelengths,
                                                              dim=dim,
                                                              event_mode=event_mode)

    # The first wavelength should fail the check, since anything not passing through
    # the choppers won't satisfy the dlambda/lambda condition.
    _check_lambda_inside_resolution(wavelengths['event', 0],
                                    dlambda_over_lambda,
                                    stitched,
                                    check_value=False,
                                    event_mode=event_mode)
    for i in range(1, len(wavelengths)):
        _check_lambda_inside_resolution(wavelengths['event', i],
                                        dlambda_over_lambda,
                                        stitched,
                                        event_mode=event_mode)
