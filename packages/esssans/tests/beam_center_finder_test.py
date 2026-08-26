# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import sciline
import scipp as sc
from ess.sans.beam_center_finder import beam_center_from_center_of_mass
from ess.sans.common import gravity_vector
from ess.sans.conversions import ElasticCoordTransformGraph, sans_elastic
from ess.sans.types import (
    CorrectForGravity,
    CorrectedDetector,
    Numerator,
    Position,
    SampleRun,
)

import scippnexus as snx


def make_detector(
    *,
    center: sc.Variable,
    distance: sc.Variable,
    size: int = 41,
    extent: float = 1.0,
) -> sc.DataArray:
    """
    Flat square detector with a Gaussian intensity distribution centered on ``center``.

    The detector is normal to the Z axis. The pattern is narrow enough to be contained
    well within the detector, so that its center of mass is unbiased.
    """
    x = sc.linspace('x', -extent, extent, size, unit='m')
    y = sc.linspace('y', -extent, extent, size, unit='m')
    position = sc.spatial.as_vectors(
        x.broadcast(sizes={'y': size, 'x': size}),
        y.broadcast(sizes={'y': size, 'x': size}),
        distance.broadcast(sizes={'y': size, 'x': size}),
    )
    r2 = (position.fields.x - center.fields.x) ** 2 + (
        position.fields.y - center.fields.y
    ) ** 2
    sigma = sc.scalar(0.2, unit='m')
    counts = sc.exp(-r2 / (2 * sigma**2))
    # A wavelength dim exercises the summing of non-position dims.
    counts = counts.broadcast(sizes={**counts.sizes, 'wavelength': 3}).copy()
    return sc.DataArray(
        data=counts,
        coords={
            'position': position,
            'wavelength': sc.linspace('wavelength', 1.0, 3.0, 3, unit='angstrom'),
        },
    )


def make_workflow(
    *,
    detector: sc.DataArray,
    sample_position: sc.Variable,
    source_position: sc.Variable | None = None,
) -> sciline.Pipeline:
    if source_position is None:
        source_position = sample_position - sc.vector([0, 0, 20.0], unit='m')
    workflow = sciline.Pipeline(())
    workflow[CorrectedDetector[SampleRun, Numerator]] = detector
    workflow[ElasticCoordTransformGraph[SampleRun]] = sans_elastic(
        CorrectForGravity(False),
        sample_position=Position[snx.NXsample, SampleRun](sample_position),
        source_position=Position[snx.NXsource, SampleRun](source_position),
        gravity=gravity_vector(),
    )
    return workflow


def test_center_of_mass_finds_center_of_symmetric_pattern() -> None:
    center = sc.vector([0.1, -0.07, 0.0], unit='m')
    workflow = make_workflow(
        detector=make_detector(center=center, distance=sc.scalar(5.0, unit='m')),
        sample_position=sc.vector([0, 0, 0], unit='m'),
    )
    result = beam_center_from_center_of_mass(workflow)
    assert sc.allclose(result, center, atol=sc.scalar(1e-3, unit='m'))


def test_center_of_mass_is_measured_relative_to_the_sample_position() -> None:
    # Pattern is centered on the beam axis through the sample, so there is no offset
    # of the beam center, despite the sample being off-axis.
    sample_position = sc.vector([0.1, -0.07, 0.0], unit='m')
    workflow = make_workflow(
        detector=make_detector(
            center=sample_position, distance=sc.scalar(5.0, unit='m')
        ),
        sample_position=sample_position,
    )
    result = beam_center_from_center_of_mass(workflow)
    assert sc.allclose(
        result, sc.vector([0, 0, 0], unit='m'), atol=sc.scalar(1e-3, unit='m')
    )
