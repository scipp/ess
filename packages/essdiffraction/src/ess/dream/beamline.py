# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Beamline tools for DREAM."""

from ess.powder.types import DetectorDimensions, RawDetectorData, SampleRun


def dream_detector_dimensions(data: RawDetectorData[SampleRun]) -> DetectorDimensions:
    """Logical dimensions used by a DREAM detector.

    The dimensions differ between simulated data loaded from GEANT4 CSV files
    and measured data loaded from NeXus files.
    The dimensions returned by this function match the dimensions found
    in the ``data`` argument.

    Parameters
    ----------
    data:
        Dimensions are deduced based on this data.

    Returns
    -------
    :
        Logical dimensions used by the given DREAM detector.
    """
    geant4_dims = {
        'module',
        'segment',
        'counter',
        'wire',
        'strip',
        'sector',
    }
    nexus_dims = {
        'wire',
        'mounting_unit',
        'cassette',
        'counter',
        'strip',
        'sector',
        'sumo_cass_ctr',
        'other',
    }
    dims = (geant4_dims | nexus_dims) & set(data.dims)
    return DetectorDimensions(tuple(dims))


providers = (dream_detector_dimensions,)
"""Sciline providers for DREAM detector handling."""
