# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import scipp as sc
from scipp import array, scalar, sqrt, vector
from scipp.spatial import rotations_from_rotvecs

from ess.spectroscopy.indirect import kf as secondary


def vectors_close(a, b, tol=None):
    from scipp import scalar

    if tol is None:
        tol = scalar(1e-8, unit=a.unit)
    difference = a - b
    return sc.norm(difference) < tol


def all_vectors_close(a, b, tol=None):
    if len(a.shape) > 1 and len(b.shape) > 1:
        for d in a.dims:
            if d not in b.dims:
                return False
            if a.sizes[d] != b.sizes[d]:
                return False
        if a.shape != b.shape:
            return False
        return all_vectors_close(a.flatten(to='one'), b.flatten(to='one'), tol)
    if len(a.shape) and len(b.shape):
        if a.shape != b.shape:
            return False
        for x, y in zip(a, b, strict=True):
            if not vectors_close(x, y, tol):
                return False
    elif len(a.shape):
        for x in a:
            if not vectors_close(x, b, tol):
                return False
    elif len(b.shape):
        for y in b:
            if not vectors_close(a, y, tol):
                return False
    else:
        return vectors_close(a, b, tol)
    return True


def test_back_scattering_sample_analyzer_vector():
    sample_position = vector([0.0, 0.0, 0.0], unit='m')

    # The analyzer orientation defines the scattering plane coordinate system:
    analyzer_transform = rotations_from_rotvecs(vector([0.0, 0.0, 0.0], unit='degree'))

    x = analyzer_transform * vector([1.0, 0.0, 0.0], unit='1')
    y = analyzer_transform * vector([0.0, 1.0, 0.0], unit='1')
    z = analyzer_transform * vector([0.0, 0.0, 1.0], unit='1')
    # sample-analyzer-vector must be along the local z axis
    sample_analyzer_vec = scalar(1.0, unit='m') * z
    analyzer_position = sample_analyzer_vec + sample_position

    # analyzer orientation of 90 degrees ->
    # scattering angle of 180 degrees: back scattering
    analyzer_orientation = rotations_from_rotvecs(scalar(90.0, unit='degree') * y)

    detector_y = analyzer_orientation * analyzer_orientation * y
    assert vectors_close(detector_y, y)
    # The detector positions are defined in the coordinate system,
    # then rotated around the analyzer position
    tubes = array(values=[-0.1, 0.0, 0.1], unit='m', dims=['tube']) * x
    detector_positions = scalar(1.0, unit='m') * z + tubes
    detector_positions = (
        analyzer_position
        + analyzer_orientation * analyzer_orientation * detector_positions
    )
    calculated = secondary.sample_analyzer_vector(
        sample_position, analyzer_position, analyzer_orientation, detector_positions
    )

    # This would be 'right' for fixed tau direction, but we utilize the mosaic and
    # insist on scattering from the center line of the analyzer
    # offset = sample_analyzer_vec - tubes / 2.0
    # (minus because the detector has been rotated 180 degrees)
    assert all_vectors_close(calculated, sample_analyzer_vec)

    # Offsetting along the analyzer scattering plane normal moves the analyzer
    # intersection point by half
    wires = array(values=[-0.1, 0.0, 0.1], unit='m', dims=['wire']) * y
    detector_positions = scalar(1.0, unit='m') * z + wires
    detector_positions = (
        analyzer_position
        + analyzer_orientation * analyzer_orientation * detector_positions
    )
    calculated = secondary.sample_analyzer_vector(
        sample_position, analyzer_position, analyzer_orientation, detector_positions
    )
    assert all_vectors_close(calculated, sample_analyzer_vec + wires / 2.0)

    # Offsetting in both transverse directions at once
    detector_positions = scalar(1.0, unit='m') * z + wires + tubes
    detector_positions = (
        analyzer_position
        + analyzer_orientation * analyzer_orientation * detector_positions
    )
    calculated = secondary.sample_analyzer_vector(
        sample_position, analyzer_position, analyzer_orientation, detector_positions
    )
    # as the x displacement increases, the y displacement should decrease
    frac = wires / (1.0 + sqrt(1.0 + (sc.norm(tubes) / scalar(1.0, unit='m')) ** 2))
    assert all_vectors_close(calculated, sample_analyzer_vec + frac)


def test_sample_analyzer_vector():
    # TODO Randomize the sample position
    # TODO Randomize the analyzer transformation (a3 angle)
    # TODO Randomize the analyzer distance
    # TODO Randomize the detector positions

    from random import random

    sample_position = vector([0.0, 0.0, 0.0], unit='m')

    # The analyzer orientation defines the scattering plane coordinate system:
    analyzer_transform = rotations_from_rotvecs(vector([0.0, 0.0, 0.0], unit='degree'))

    x = analyzer_transform * vector([1.0, 0.0, 0.0], unit='1')
    y = analyzer_transform * vector([0.0, 1.0, 0.0], unit='1')
    z = analyzer_transform * vector([0.0, 0.0, 1.0], unit='1')
    # sample-analyzer-vector must be along the local z axis
    sample_analyzer_vec = scalar(0.5 + random(), unit='m') * z  # noqa: S311
    sample_analyzer_vec /= sc.norm(sample_analyzer_vec).value
    analyzer_position = sample_analyzer_vec + sample_position

    # Avoid sin(theta) values near 0 or 1
    analyzer_orientation = rotations_from_rotvecs(scalar(45.0, unit='degree') * y)

    detector_y = analyzer_orientation * analyzer_orientation * y
    assert vectors_close(detector_y, y)
    # The detector positions are defined in the coordinate system,
    # then rotated around the analyzer position
    tubes = array(values=[-0.1, 0.0, 0.1], unit='m', dims=['tube']) * x
    detector_positions = scalar(1.0, unit='m') * z + tubes
    detector_positions = (
        analyzer_position
        + analyzer_orientation * analyzer_orientation * detector_positions
    )
    calculated = secondary.sample_analyzer_vector(
        sample_position, analyzer_position, analyzer_orientation, detector_positions
    )
    offset = calculated - sample_analyzer_vec
    assert all_vectors_close(offset, 0 * tubes)

    # Offsetting along the analyzer scattering plane normal moves
    # the analyzer intersection point by half
    wires = array(values=[-0.1, 0.0, 0.1], unit='m', dims=['wire']) * y
    detector_positions = scalar(1.0, unit='m') * z + wires
    detector_positions = (
        analyzer_position
        + analyzer_orientation * analyzer_orientation * detector_positions
    )
    calculated = secondary.sample_analyzer_vector(
        sample_position, analyzer_position, analyzer_orientation, detector_positions
    )
    assert all_vectors_close(calculated, sample_analyzer_vec + wires / 2.0)

    # Offsetting in both transverse directions at once
    detector_positions = scalar(1.0, unit='m') * z + wires + tubes
    detector_positions = (
        analyzer_position
        + analyzer_orientation * analyzer_orientation * detector_positions
    )
    calculated = secondary.sample_analyzer_vector(
        sample_position, analyzer_position, analyzer_orientation, detector_positions
    )
    # as the x displacement increases, the y displacement should decrease
    frac = wires / (1.0 + sqrt(1.0 + (sc.norm(tubes) / scalar(1.0, unit='m')) ** 2))
    assert all_vectors_close(calculated, sample_analyzer_vec + frac)
