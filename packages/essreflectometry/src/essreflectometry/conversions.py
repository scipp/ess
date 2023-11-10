# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import h, m_n, pi
from scippneutron._utils import elem_dtype, elem_unit
from scippneutron.conversion.graph import beamline, tof

# from . import orso
from .types import (
    FootprintCorrectedData,
    HistogrammedQData,
    QBins,
    QData,
    RawData,
    Run,
    SpecularReflectionCoordTransformGraph,
    ThetaData,
    WavelengthData,
)


def theta(
    gravity: sc.Variable,
    wavelength: sc.Variable,
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    sample_rotation: sc.Variable,
) -> sc.Variable:
    """
    Compute the theta angle, including gravity correction,
    This is similar to the theta calculation in SANS (see
    https://docs.mantidproject.org/nightly/algorithms/Q1D-v2.html#q-unit-conversion),
    but we ignore the horizontal `x` component.
    See the schematic in Fig 5 of doi: 10.1016/j.nima.2016.03.007.

    Parameters
    ----------
    gravity:
        The three-dimensional vector describing gravity.
    wavelength:
        Wavelength values for the events.
    incident_beam:
        Vector for incident beam.
    scatter_beam:
        Vector for scattered beam.
    sample_rotation:
        Rotation of sample wrt to incoming beam.

    Returns
    -------
    :
        Theta values, accounting for gravity.
    """
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)
    y = sc.dot(scattered_beam, gravity) / grav
    y_correction = sc.to_unit(wavelength, elem_unit(L2), copy=True)
    y_correction *= y_correction
    drop = L2**2
    drop *= grav * (m_n**2 / (2 * h**2))
    # Optimization when handling either the dense or the event coord of binned data:
    # - For the event coord, both operands have same dims, and we can multiply in place
    # - For the dense coord, we need to broadcast using non in-place operation
    if set(drop.dims).issubset(set(y_correction.dims)):
        y_correction *= drop
    else:
        y_correction = y_correction * drop
    y_correction += y
    out = sc.abs(y_correction, out=y_correction)
    out /= L2
    out = sc.asin(out, out=out)
    out -= sc.to_unit(sample_rotation, 'rad')
    return out


def reflectometry_q(wavelength: sc.Variable, theta: sc.Variable) -> sc.Variable:
    """
    Compute the Q vector from the theta angle computed as the difference
    between gamma and omega.
    Note that this is identical the 'normal' Q defined in scippneutron, except that
    the `theta` angle is given as an input instead of `two_theta`.

    Parameters
    ----------
    wavelength:
        Wavelength values for the events.
    theta:
        Theta values, accounting for gravity.

    Returns
    -------
    :
        Q-values.
    """
    dtype = elem_dtype(wavelength)
    c = (4 * pi).astype(dtype)
    return c * sc.sin(theta.astype(dtype, copy=False)) / wavelength


def specular_reflection() -> SpecularReflectionCoordTransformGraph:
    """
    Generate a coordinate transformation graph for specular reflection reflectometry.

    Returns
    -------
    :
        Specular reflectometry graph.
    """
    graph = {
        **beamline.beamline(scatter=True),
        **tof.elastic_wavelength("tof"),
        "theta": theta,
        "Q": reflectometry_q,
    }
    return SpecularReflectionCoordTransformGraph(graph)


def tof_to_wavelength(
    data_array: RawData[Run],
    graph: SpecularReflectionCoordTransformGraph,
) -> WavelengthData[Run]:
    """
    Use :code:`transform_coords` to convert from ToF to wavelength, cutoff high and
    low limits for wavelength, and add necessary ORSO metadata.

    Parameters
    ----------
    data_array:
        Data array to convert.
    graph:
        Graph for :code:`transform_coords`.

    Returns
    -------
    :
        New data array with wavelength dimension.
    """
    data_array_wav = data_array.transform_coords(["wavelength"], graph=graph)
    # TODO
    # try:
    #    from orsopy import fileio

    #    unit = data_array_wav.coords['wavelength'].unit
    #    # This insures that when the unit is Å it is written as
    #    # angstrom in the ORSO object.
    #    if unit == 'angstrom':
    #        unit = 'angstrom'
    #    orso_measurement = data_array_wav.attrs['orso'].value.data_source.measurement
    #    orso_measurement.instrument_settings.wavelength = fileio.base.ValueRange(
    #        float(data_array_wav.coords['wavelength'].min().value),
    #        float(data_array_wav.coords['wavelength'].max().value),
    #        unit,
    #    )
    # except ImportError:
    #     orso.not_found_warning()
    return WavelengthData[Run](data_array_wav)


def wavelength_to_theta(
    data_array: WavelengthData[Run],
    graph: SpecularReflectionCoordTransformGraph,
) -> ThetaData[Run]:
    """
    Use :code:`transform_coords` to find the theta values for the events and
    potentially add ORSO metadata.

    Parameters
    ----------
    data_array:
        Data array to convert.
    graph:
        Graph for :code:`transform_coords`.

    Returns
    -------
    :
        New data array with theta coordinate.
    """
    data_array_theta = data_array.transform_coords(['theta'], graph=graph)
    # TODO
    # try:
    #    from orsopy import fileio

    #    orso_measurement = data_array_theta.attrs['orso'].value.data_source.measurement
    #    orso_measurement.instrument_settings.incident_angle = fileio.base.ValueRange(
    #        float(data_array_theta.coords['theta'].min().value),
    #        float(data_array_theta.coords['theta'].max().value),
    #        data_array_theta.bins.coords['theta'].min().unit,
    #    )
    #    import inspect

    #    # Determine if 'gravity' is in the graph and if to add the gravity correction
    #    if any(
    #        [
    #            'gravity' in i.parameters.keys()
    #            for i in map(inspect.signature, graph.values())
    #        ]
    #    ):
    #        data_array_theta.attrs['orso'].value.reduction.corrections += [
    #            'gravity correction'
    #        ]
    # except ImportError:
    #    orso.not_found_warning()
    return ThetaData[Run](data_array_theta)


def theta_to_q(
    data_array: FootprintCorrectedData[Run],
    q_bins: QBins,
    graph: SpecularReflectionCoordTransformGraph,
) -> QData[Run]:
    """
    Convert from theta to Q and if necessary bin in Q.

    Parameters
    ----------
    data_array:
        Data array to convert.
    q_edges:
        The lower and upper limits for the Q.
    graph:
        Graph for :code:`transform_coords`.

    Returns
    -------
    :
        New data array with theta coordinate.
    """
    data_array = data_array.transform_coords(["Q"], graph=graph)
    data_array = data_array.bin({q_bins.dim: q_bins})
    return QData[Run](data_array)


def histogram(data_array: QData[Run]) -> HistogrammedQData[Run]:
    """
    Sum the event bins.

    Parameters
    ----------
    data_array:
        Data array to be summed.

    Returns
    -------
    :
        Summed data array.
    """
    return HistogrammedQData[Run](data_array.hist())


providers = [
    tof_to_wavelength,
    wavelength_to_theta,
    theta_to_q,
    histogram,
]
