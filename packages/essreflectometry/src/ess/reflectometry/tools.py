# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, NewType

import numpy as np
import sciline
import scipp as sc
import scipy.optimize as opt

# from ess.reflectometry import orso
from ess.reflectometry.types import (
    Filename,
    QBins,
    ReflectivityOverQ,
    SampleRun,
    ScalingFactorForOverlap,
    UnscaledReducibleData,
)
from ess.reflectometry.workflow import with_filenames

_STD_TO_FWHM = sc.scalar(2.0) * sc.sqrt(sc.scalar(2.0) * sc.log(sc.scalar(2.0)))


def fwhm_to_std(fwhm: sc.Variable) -> sc.Variable:
    """
    Convert from full-width half maximum to standard deviation.

    Parameters
    ----------
    fwhm:
        Full-width half maximum.

    Returns
    -------
    :
        Standard deviation.
    """
    # Enables the conversion from full width half
    # maximum to standard deviation
    return fwhm / _STD_TO_FWHM


def linlogspace(
    dim: str,
    edges: list | np.ndarray,
    scale: list | str,
    num: list | int,
    unit: str | None = None,
) -> sc.Variable:
    """
    Generate a 1d array of bin edges with a mixture of linear and/or logarithmic
    spacings.

    Examples:

    - Create linearly spaced edges (equivalent to `sc.linspace`):
        linlogspace(dim='x', edges=[0.008, 0.08], scale='linear', num=50, unit='m')
    - Create logarithmically spaced edges (equivalent to `sc.geomspace`):
        linlogspace(dim='x', edges=[0.008, 0.08], scale='log', num=50, unit='m')
    - Create edges with a linear and a logarithmic part:
        linlogspace(dim='x', edges=[1, 3, 8], scale=['linear', 'log'], num=[16, 20])

    Parameters
    ----------
    dim:
        The dimension of the output Variable.
    edges:
        The edges for the different parts of the mesh.
    scale:
        A string or list of strings specifying the scaling for the different
        parts of the mesh. Possible values for the scaling are `"linear"` and `"log"`.
        If a list is supplied, the length of the list must be one less than the length
        of the `edges` parameter.
    num:
        An integer or a list of integers specifying the number of points to use
        in each part of the mesh. If a list is supplied, the length of the list must be
        one less than the length of the `edges` parameter.
    unit:
        The unit of the output Variable.

    Returns
    -------
    :
        Lin-log spaced Q-bin edges.
    """
    if not isinstance(scale, list):
        scale = [scale]
    if not isinstance(num, list):
        num = [num]
    if len(scale) != len(edges) - 1:
        raise ValueError(
            "Sizes do not match. The length of edges should be one greater than scale."
        )

    funcs = {"linear": sc.linspace, "log": sc.geomspace}
    grids = []
    for i in range(len(edges) - 1):
        # Skip the leading edge in the piece when concatenating
        start = int(i > 0)
        mesh = funcs[scale[i]](
            dim=dim, start=edges[i], stop=edges[i + 1], num=num[i] + start, unit=unit
        )
        grids.append(mesh[dim, start:])

    return sc.concat(grids, dim)


def _sort_by(a, by):
    return [x for x, _ in sorted(zip(a, by, strict=True), key=lambda x: x[1])]


def _find_interval_overlaps(intervals):
    '''Returns the intervals where at least
    two or more of the provided intervals
    are overlapping.'''
    edges = list(chain.from_iterable(intervals))
    is_start_edge = list(chain.from_iterable((True, False) for _ in intervals))
    edges_sorted = sorted(edges)
    is_start_edge_sorted = _sort_by(is_start_edge, edges)

    number_overlapping = 0
    overlap_intervals = []
    for x, is_start in zip(edges_sorted, is_start_edge_sorted, strict=True):
        if number_overlapping == 1 and is_start:
            start = x
        if number_overlapping == 2 and not is_start:
            overlap_intervals.append((start, x))
        if is_start:
            number_overlapping += 1
        else:
            number_overlapping -= 1
    return overlap_intervals


def _searchsorted(a, v):
    for i, e in enumerate(a):
        if e > v:
            return i
    return len(a)


def _create_qgrid_where_overlapping(qgrids):
    '''Given a number of Q-grids, construct a new grid
    covering the regions where (any two of the) provided grids overlap.'''
    pieces = []
    for start, end in _find_interval_overlaps([(q.min(), q.max()) for q in qgrids]):
        interval_sliced_from_qgrids = [
            q[max(_searchsorted(q, start) - 1, 0) : _searchsorted(q, end) + 1]
            for q in qgrids
        ]
        densest_grid_in_interval = max(interval_sliced_from_qgrids, key=len)
        pieces.append(densest_grid_in_interval)
    return sc.concat(pieces, dim='Q')


def _same_dtype(arrays):
    return [arr.to(dtype='float64') for arr in arrays]


def _interpolate_on_qgrid(curves, grid):
    return sc.concat(
        _same_dtype([sc.lookup(c, grid.dim)[sc.midpoints(grid)] for c in curves]),
        dim='curves',
    )


CriticalEdgeKey = NewType('CriticalEdgeKey', None)
"""A unique key used to store a 'fake' critical edge in a workflow collection."""


def scale_reflectivity_curves_to_overlap(
    wf_collection: Sequence[sc.DataArray],
    critical_edge_interval: tuple[sc.Variable, sc.Variable] | None = None,
    cache_intermediate_results: bool = True,
) -> tuple[list[sc.DataArray], list[sc.Variable]]:
    '''
    Set the ``ScalingFactorForOverlap`` parameter on the provided workflows
    in a way that would makes the 1D reflectivity curves overlap.

    If :code:`critical_edge_interval` is not provided, all workflows are scaled except
    the data with the lowest Q-range, which is considered to be the reference curve.
    The scaling factors are determined by a maximum likelihood estimate
    (assuming the errors are normal distributed).

    If :code:`critical_edge_interval` is provided then all data are scaled.

    All reflectivity curves must be have the same unit for data and the Q-coordinate.

    Parameters
    ---------
    wf_collection:
        The collection of workflows that can compute the ``ReflectivityOverQ``.
    critical_edge_interval:
        A tuple denoting an interval that is known to belong
        to the critical edge, i.e. where the reflectivity is
        known to be 1.
    cache_intermediate_results:
        If ``True`` the intermediate results ``UnscaledReducibleData`` will be cached
        (this is the base for all types that are downstream of the scaling factor).

    Returns
    ---------
    :
        A list of scaled reflectivity curves and a list of the scaling factors.
    '''
    if critical_edge_interval is not None:
        # Find q bins with the lowest Q start point
        q = min(
            (wf.compute(QBins) for wf in wf_collection.values()),
            key=lambda q_: q_.min(),
        )
        N = (
            ((q >= critical_edge_interval[0]) & (q < critical_edge_interval[1]))
            .sum()
            .value
        )
        edge = sc.DataArray(
            data=sc.ones(dims=('Q',), shape=(N,), with_variances=True),
            coords={'Q': sc.linspace('Q', *critical_edge_interval, N + 1)},
        )
        wfc = wf_collection.copy()
        underlying_wf = next(iter(wfc.values()))
        edge_wf = underlying_wf.copy()
        edge_wf[ReflectivityOverQ] = edge
        wfc.add(CriticalEdgeKey, edge_wf)
        return scale_reflectivity_curves_to_overlap(
            wfc, cache_intermediate_results=cache_intermediate_results
        )

    wfc = wf_collection.copy()
    if cache_intermediate_results:
        wfc[UnscaledReducibleData[SampleRun]] = wfc.compute(
            UnscaledReducibleData[SampleRun]
        )

    reflectivities = wfc.compute(ReflectivityOverQ)

    # First sort the dict of reflectivities by the Q min value
    curves = {
        k: v.hist() if v.bins is not None else v
        for k, v in sorted(
            reflectivities.items(), key=lambda item: item[1].coords['Q'].min().value
        )
    }
    # Now place the critical edge at the beginning, if it exists
    if CriticalEdgeKey in curves.keys():
        curves = {CriticalEdgeKey: curves[CriticalEdgeKey]} | curves

    if len({c.data.unit for c in curves.values()}) != 1:
        raise ValueError('The reflectivity curves must have the same unit')
    if len({c.coords['Q'].unit for c in curves.values()}) != 1:
        raise ValueError('The Q-coordinates must have the same unit for each curve')

    qgrid = _create_qgrid_where_overlapping([c.coords['Q'] for c in curves.values()])

    r = _interpolate_on_qgrid(map(sc.values, curves.values()), qgrid).values
    v = _interpolate_on_qgrid(map(sc.variances, curves.values()), qgrid).values

    def cost(scaling_factors):
        scaling_factors = np.concatenate([[1.0], scaling_factors])[:, None]
        r_scaled = scaling_factors * r
        v_scaled = scaling_factors**2 * v
        v_scaled[v_scaled == 0] = np.nan
        inv_v_scaled = 1 / v_scaled
        r_avg = np.nansum(r_scaled * inv_v_scaled, axis=0) / np.nansum(
            inv_v_scaled, axis=0
        )
        return np.nansum((r_scaled - r_avg) ** 2 * inv_v_scaled)

    sol = opt.minimize(cost, [1.0] * (len(curves) - 1))
    scaling_factors = (1.0, *map(float, sol.x))

    wfc[ScalingFactorForOverlap[SampleRun]] = dict(
        zip(curves.keys(), scaling_factors, strict=True)
    )

    if CriticalEdgeKey in wfc.keys():
        wfc.remove(CriticalEdgeKey)

    return wfc


def combine_curves(
    curves: Sequence[sc.DataArray],
    q_bin_edges: sc.Variable | None = None,
) -> sc.DataArray:
    '''Combines the given curves by interpolating them
    on a 1d grid defined by :code:`q_bin_edges` and averaging
    over the provided reflectivity curves.

    The averaging is done using a weighted mean where the weights
    are proportional to the variances.

    Unless the curves are already scaled correctly they might
    need to be scaled using :func:`scale_reflectivity_curves_to_overlap`
    before calling this function.

    All curves must be have the same unit for data and the Q-coordinate.

    Parameters
    ----------
    curves:
        the reflectivity curves that should be combined
    q_bin_edges:
        the Q bin edges of the resulting combined reflectivity curve

    Returns
    ---------
    :
        A data array representing the combined reflectivity curve
    '''
    if len({c.data.unit for c in curves}) != 1:
        raise ValueError('The reflectivity curves must have the same unit')
    if len({c.coords['Q'].unit for c in curves}) != 1:
        raise ValueError('The Q-coordinates must have the same unit for each curve')

    r = _interpolate_on_qgrid(map(sc.values, curves), q_bin_edges).values
    v = _interpolate_on_qgrid(map(sc.variances, curves), q_bin_edges).values

    v[v == 0] = np.nan
    inv_v = 1.0 / v
    r_avg = np.nansum(r * inv_v, axis=0) / np.nansum(inv_v, axis=0)
    v_avg = 1 / np.nansum(inv_v, axis=0)
    return sc.DataArray(
        data=sc.array(
            dims='Q',
            values=r_avg,
            variances=v_avg,
            unit=next(iter(curves)).data.unit,
        ),
        coords={'Q': q_bin_edges},
    )


class WorkflowCollection:
    """
    A collection of sciline workflows that can be used to compute multiple
    targets from multiple workflows.
    It can also be used to set parameters for all workflows in a single shot.
    """

    def __init__(self, workflows: Mapping[str, sciline.Pipeline]):
        self._workflows = {name: pl.copy() for name, pl in workflows.items()}

    def __setitem__(self, key: type, value: Any | Mapping[type, Any]):
        if hasattr(value, 'items'):
            for name, v in value.items():
                self._workflows[name][key] = v
        else:
            for pl in self._workflows.values():
                pl[key] = value

    def __getitem__(self, name: str) -> sciline.Pipeline:
        """
        Returns a single workflow from the collection given by its name.
        """
        return self._workflows[name]

    def compute(self, target: type | Sequence[type], **kwargs) -> Mapping[str, Any]:
        return {
            name: pl.compute(target, **kwargs) for name, pl in self._workflows.items()
        }

    def copy(self) -> 'WorkflowCollection':
        return self.__class__(self._workflows)

    def keys(self) -> Sequence[str]:
        return self._workflows.keys()

    def values(self) -> Sequence[sciline.Pipeline]:
        return self._workflows.values()

    def items(self) -> Sequence[tuple[str, sciline.Pipeline]]:
        return self._workflows.items()

    def add(self, name: str, workflow: sciline.Pipeline):
        """
        Adds a new workflow to the collection.
        """
        self._workflows[name] = workflow.copy()

    def remove(self, name: str):
        """
        Removes a workflow from the collection by its name.
        """
        del self._workflows[name]


def batch_processor(
    workflow: sciline.Pipeline, runs: Mapping[Any, Mapping[type, Any]]
) -> WorkflowCollection:
    """
    Creates a collection of sciline workflows from the provided runs.

    Runs can be provided as a mapping of names to parameters or as a sequence
    of mappings of parameters and values.

    Parameters
    ----------
    workflow:
        The sciline workflow used to compute the targets for each of the runs.
    runs:
        The sciline parameters to be used for each run.
        TODO: explain how grouping works depending on the type of `runs`.
    """
    workflows = {}
    for name, parameters in runs.items():
        wf = workflow.copy()
        for tp, value in parameters.items():
            if tp is Filename[SampleRun]:
                continue
            wf[tp] = value

        if Filename[SampleRun] in parameters:
            if isinstance(parameters[Filename[SampleRun]], list | tuple):
                wf = with_filenames(
                    wf,
                    SampleRun,
                    parameters[Filename[SampleRun]],
                )
            else:
                wf[Filename[SampleRun]] = parameters[Filename[SampleRun]]
        workflows[name] = wf
    return WorkflowCollection(workflows)
