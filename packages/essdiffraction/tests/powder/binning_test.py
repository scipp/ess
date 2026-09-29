# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import scipp as sc
from ess import powder
from ess.powder.binning import (
    dspacing_bins_from_wavelength_and_two_theta,
    wavelength_range_from_chopper_frames,
)
from ess.powder.conversion import powder_coordinate_transformation_graph
from ess.powder.types import (
    DetectorTwoTheta,
    DspacingNBins,
    SampleRun,
    WavelengthRange,
)
from scippneutron.tof import chopper_cascade


def test_automatic_dspacing_binning_is_opt_in_for_powder_workflows():
    assert set(powder.binning.providers).isdisjoint(powder.providers)


def test_wavelength_range_uses_frame_after_last_chopper():
    source = chopper_cascade.FrameSequence.from_source_pulse(
        time_min=sc.scalar(0.0, unit='ms'),
        time_max=sc.scalar(3.0, unit='ms'),
        wavelength_min=sc.scalar(0.5, unit='angstrom'),
        wavelength_max=sc.scalar(5.0, unit='angstrom'),
    )
    last_subframe = chopper_cascade.Subframe(
        time=sc.array(dims=['vertex'], values=[1.0, 2.0], unit='ms'),
        wavelength=sc.array(dims=['vertex'], values=[1.2, 3.4], unit='angstrom'),
    )
    frames = chopper_cascade.FrameSequence(
        [
            *source.frames,
            chopper_cascade.Frame(
                distance=sc.scalar(10.0, unit='m'), subframes=[last_subframe]
            ),
        ]
    )

    wavelength_range = wavelength_range_from_chopper_frames(frames)

    assert sc.identical(
        wavelength_range,
        sc.array(dims=['bound'], values=[1.2, 3.4], unit='angstrom'),
    )


def test_dspacing_bins_span_envelope_of_wavelength_and_two_theta():
    wavelength_range = WavelengthRange[SampleRun](
        sc.array(dims=['bound'], values=[1.0, 4.0], unit='angstrom')
    )
    two_theta = DetectorTwoTheta[SampleRun](
        sc.array(dims=['pixel'], values=[30.0, 60.0, 90.0], unit='deg')
    )
    graph = powder_coordinate_transformation_graph(
        source_position=sc.vector([0.0, 0.0, -1.0], unit='m'),
        sample_position=sc.vector([0.0, 0.0, 0.0], unit='m'),
        gravity=sc.vector([0.0, -9.81, 0.0], unit='m/s^2'),
    )
    bins = dspacing_bins_from_wavelength_and_two_theta(
        wavelength_range, two_theta, graph, DspacingNBins(4)
    )

    assert bins.sizes == {'dspacing': 5}
    assert sc.allclose(
        bins[[0, -1]],
        sc.array(
            dims=['dspacing'],
            values=[
                1.0 / (2**0.5),
                4.0 / (2 * sc.sin(15.0 * sc.Unit('deg')).value),
            ],
            unit='angstrom',
        ),
    )
