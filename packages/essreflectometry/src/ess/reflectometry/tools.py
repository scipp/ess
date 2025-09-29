# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any

import numpy as np
import sciline as sl
import scipp as sc
import scipy.optimize as opt

from ess.reflectometry import orso
from ess.reflectometry.types import (
    Filename,
    ReducibleData,
    ReflectivityOverQ,
    SampleRun,
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


class MultiGraphViz:
    """
    A dummy class to concatenate multiple graphviz visualizations into a single repr
    output for Jupyter notebooks.
    This combines the SVG representations of multiple graphs vertically with a small gap
    in between.
    """

    def __init__(self, graphs: Sequence):
        self.graphs = graphs

    def _repr_mimebundle_(self, include=None, exclude=None):
        gap = 10
        parsed = []
        for svg in [g._repr_image_svg_xml() for g in self.graphs]:
            # extract width, height, and inner <g> content
            m = re.search(r'width="([\d.]+)pt".*?height="([\d.]+)pt"', svg, re.S)
            w, h = float(m.group(1)), float(m.group(2))
            inner = re.search(r'<svg[^>]*>(.*)</svg>', svg, re.S).group(1)
            parsed.append((w, h, inner))

        # vertical shift
        total_width = max(w for w, _, _ in parsed)
        total_height = sum(h for _, h, _ in parsed) + gap * (len(parsed) - 1)

        pieces = []
        offset_x = offset_y = 0
        for _, h, inner in parsed:
            pieces.append(
                f'<g transform="translate({offset_x},{offset_y})">{inner}</g>'
            )
            offset_y += h + gap

        # TODO: for some reason, combining the svgs seems to scale them down. This
        # then means that the computed bounding box is too large. For now, we
        # apply a fudge factor of 0.75 to the width and height. It is unclear where
        # exactly this comes from.
        combined = f'''
        <svg xmlns="http://www.w3.org/2000/svg"
            width="{total_width * 0.75}pt" height="{total_height * 0.75}pt">
        {''.join(pieces)}
        </svg>
        '''
        return {"image/svg+xml": combined}


class BatchProcessor:
    """
    A collection of sciline workflows that can be used to compute multiple
    targets from multiple workflows.
    It can also be used to set parameters for all workflows in a single shot.
    """

    def __init__(self, workflows: Mapping[str, sl.Pipeline]):
        self.workflows = workflows

    def __setitem__(self, key: type, value: Mapping[str, Any]) -> None:
        """
        A mapping (dict or DataGroup) should be supplied as the value. The keys
        of the mapping should correspond to the names of the workflows in the
        collection. The node matching the key will be set to the corresponding value for
        each of the workflows.
        """
        for name, v in value.items():
            self.workflows[name][key] = v

    def __getitem__(self, name: str) -> BatchProcessor:
        """
        Get a new BatchProcessor where the workflows are the sub-workflows that lead to
        the node with the given name.
        """
        return BatchProcessor({k: wf[name] for k, wf in self.workflows.items()})

    def compute(self, targets: type | Sequence[type], **kwargs) -> Mapping[str, Any]:
        """
        Compute the given target(s) for all workflows in the collection.

        Parameters
        ----------
        targets:
            The target type(s) to compute.
        **kwargs:
            Additional keyword arguments passed to `sciline.Pipeline.compute`.
        """
        if not isinstance(targets, list | tuple):
            targets = [targets]
        out = {}
        for t in targets:
            out[t] = sc.DataGroup()
            for name, wf in self.workflows.items():
                try:
                    out[t][name] = wf.compute(t, **kwargs)
                except sl.UnsatisfiedRequirement as e:
                    try:
                        out[t][name] = sl.compute_mapped(
                            wf, t, **kwargs
                        ).values.tolist()
                    except (sl.UnsatisfiedRequirement, ValueError):
                        # ValueError is raised when the requested type is not mapped
                        raise e from e
        return next(iter(out.values())) if len(out) == 1 else out

    def copy(self) -> BatchProcessor:
        """
        Create a copy of the workflow collection.
        """
        return BatchProcessor({k: wf.copy() for k, wf in self.workflows.items()})

    def visualize(self, targets: type | Sequence[type], **kwargs) -> MultiGraphViz:
        """
        Visualize all workflows in the collection.

        Parameters
        ----------
        targets : type | Sequence[type]
            The target type(s) to visualize.
        **kwargs:
            Additional keyword arguments passed to `sciline.Pipeline.visualize`.
        """
        from graphviz import Digraph

        # Place all the graphviz Digraphs side by side into a single one.
        if not isinstance(targets, list | tuple):
            targets = [targets]
        graphs = []
        for key, wf in self.workflows.items():
            v = wf.visualize(targets, **kwargs)
            g = Digraph(
                graph_attr=v.graph_attr, node_attr=v.node_attr, edge_attr=v.edge_attr
            )
            with g.subgraph(name=f"cluster_{key}") as c:
                c.attr(label=key, style="rounded", color="black")
                c.body.extend(v.body)

            graphs.append(g)

        return MultiGraphViz(graphs)


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


def scale_for_reflectivity_overlap(
    reflectivities: sc.DataArray | Mapping[str, sc.DataArray] | sc.DataGroup,
    critical_edge_interval: tuple[sc.Variable, sc.Variable]
    | list[sc.Variable]
    | None = None,
) -> sc.DataArray | sc.DataGroup:
    '''
    Compute a scaling for 1D reflectivity curves in a way that would makes the curves
    overlap.
    One can supply either a single curve or a collection/DataGroup of curves.

    If :code:`critical_edge_interval` is not provided, all curves are scaled except
    the data with the lowest Q-range, which is considered to be the reference curve.
    The scaling factors are determined by a maximum likelihood estimate
    (assuming the errors are normal distributed).

    If :code:`critical_edge_interval` is provided then all data are scaled.

    All reflectivity curves must be have the same unit for data and the Q-coordinate.

    Parameters
    ---------
    reflectivities:
        The reflectivity curves that should be scaled.
    critical_edge_interval:
        A tuple denoting an interval that is known to belong
        to the critical edge, i.e. where the reflectivity is
        known to be 1.

    Returns
    ---------
    :
        A DataGroup with the same keys as the input containing the
        scaling factors for each reflectivity curve.
    '''
    only_one_curve = isinstance(reflectivities, sc.DataArray)
    if only_one_curve:
        reflectivities = {"": reflectivities}

    # First sort the dict of reflectivities by the Q min value
    curves = {
        k: v.hist() if v.bins is not None else v
        for k, v in sorted(
            reflectivities.items(), key=lambda item: item[1].coords['Q'].min().value
        )
    }

    critical_edge_key = uuid.uuid4().hex
    if critical_edge_interval is not None:
        q = {key: c.coords['Q'] for key, c in curves.items()}
        q = min(q.values(), key=lambda q_: q_.min())
        # TODO: This is slightly different from before: it extracts the bins from the
        # QBins variable that cover the critical edge interval. This means that the
        # resulting curve will not necessarily begin and end exactly at the values
        # specified, but rather at the closest bin edges.
        edge = sc.DataArray(
            data=sc.ones(sizes={q.dim: q.sizes[q.dim] - 1}, with_variances=True),
            coords={q.dim: q},
        )[q.dim, critical_edge_interval[0] : critical_edge_interval[1]]
        # Now place the critical edge at the beginning
        curves = {critical_edge_key: edge} | curves

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

    out = sc.DataGroup(
        {
            k: v
            for k, v in zip(curves.keys(), scaling_factors, strict=True)
            if k != critical_edge_key
        }
    )

    return out[""] if only_one_curve else out


def combine_curves(
    curves: Sequence[sc.DataArray] | sc.DataGroup | Mapping[str, sc.DataArray],
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
    if hasattr(curves, 'items'):
        curves = list(curves.values())
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


def batch_processor(
    workflow: sl.Pipeline, runs: Mapping[Any, Mapping[type, Any]]
) -> BatchProcessor:
    """
    Creates a collection of sciline workflows from the provided runs.

    Example:

    ```
    from ess.reflectometry import amor, tools

    workflow = amor.AmorWorkflow()

    runs = {
        '608': {
            SampleRotationOffset[SampleRun]: sc.scalar(0.05, unit='deg'),
            Filename[SampleRun]: "file_608.hdf",
        },
        '609': {
            SampleRotationOffset[SampleRun]: sc.scalar(0.05, unit='deg'),
            Filename[SampleRun]: "file_609.hdf",
        },
        '610': {
            SampleRotationOffset[SampleRun]: sc.scalar(0.05, unit='deg'),
            Filename[SampleRun]: "file_610.hdf",
        },
        '611': {
            SampleRotationOffset[SampleRun]: sc.scalar(0.05, unit='deg'),
            Filename[SampleRun]: "file_611.hdf",
        },
    }

    batch = tools.batch_processor(workflow, runs)

    results = batch.compute(ReflectivityOverQ)
    ```

    Additionally, if a list of filenames is provided for
    ``Filename[SampleRun]``, the events from the files will be concatenated
    into a single event list before processing.

    Example:

    ```
    runs = {
        '608': {
            Filename[SampleRun]: "file_608.hdf",
        },
        '609+610': {
            Filename[SampleRun]: ["file_609.hdf", "file_610.hdf"],
        },
    }
    ```

    Parameters
    ----------
    workflow:
        The sciline workflow used to compute the targets for each of the runs.
    runs:
        The sciline parameters to be used for each run.
        Should be a mapping where the keys are the names of the runs
        and the values are mappings of type to value pairs.
        In addition, if one of the values for ``Filename[SampleRun]``
        is a list or a tuple, then the events from the files
        will be concatenated into a single event list.
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
                wf = with_filenames(wf, SampleRun, parameters[Filename[SampleRun]])
            else:
                wf[Filename[SampleRun]] = parameters[Filename[SampleRun]]
        workflows[name] = wf
    return BatchProcessor(workflows)


def batch_compute(
    workflow: sl.Pipeline,
    runs: Sequence[Mapping[type, Any]] | Mapping[Any, Mapping[type, Any]],
    target: type | Sequence[type] = orso.OrsoIofQDataset,
    *,
    scale_to_overlap: bool
    | tuple[sc.Variable, sc.Variable]
    | list[sc.Variable] = False,
) -> list | Mapping:
    '''
    Computes requested target(s) from a supplied workflow for a number of runs.
    Each entry of :code:`runs` is a mapping of parameters and
    values needed to produce the targets.

    This is an alternative to using :func:`batch_processor`: instead of returning a
    BatchProcessor object which can operate on multiple workflows at once,
    this function directly computes the requested targets, reducing the risk of
    accidentally compromizing the workflows in the collection.

    It also provides the option to scale the reflectivity curves so that they overlap
    in the regions where they have the same Q-value.

    Beginners should prefer this function over :func:`batch_processor` unless
    they need the extra flexibility of the latter (caching intermediate results,
    quickly exploring results, etc).

    Example:

    ```
    from ess.reflectometry import amor, tools

    workflow = amor.AmorWorkflow()

    runs = {
        '608': {
            SampleRotationOffset[SampleRun]: sc.scalar(0.05, unit='deg'),
            Filename[SampleRun]: "file_608.hdf",
        },
        '609': {
            SampleRotationOffset[SampleRun]: sc.scalar(0.05, unit='deg'),
            Filename[SampleRun]: "file_609.hdf",
        },
        '610': {
            SampleRotationOffset[SampleRun]: sc.scalar(0.05, unit='deg'),
            Filename[SampleRun]: "file_610.hdf",
        },
        '611': {
            SampleRotationOffset[SampleRun]: sc.scalar(0.05, unit='deg'),
            Filename[SampleRun]: "file_611.hdf",
        },
    }

    r_of_q = tools.batch_compute(workflow, runs, target=ReflectivityOverQ)
    ```

    Additionally, if a list of filenames is provided for
    ``Filename[SampleRun]``, the events from the files will be concatenated
    into a single event list before processing.

    Example:

    ```
    runs = {
        '608': {
            Filename[SampleRun]: "file_608.hdf",
        },
        '609+610': {
            Filename[SampleRun]: ["file_609.hdf", "file_610.hdf"],
        },
    }
    ```

    Parameters
    -----------
    workflow:
        The sciline workflow used to compute `ReflectivityOverQ` for each of the runs.

    runs:
        The sciline parameters to be used for each run.

    target:
        The domain type(s) to compute for each run.

    scale_to_overlap:
        If ``True`` the loaded data will be scaled so that the computed reflectivity
        curves to overlap.
        If a tuple is provided, it is interpreted as a critical edge interval where
        the reflectivity is known to be 1.
    '''
    batch = batch_processor(workflow=workflow, runs=runs)

    if scale_to_overlap:
        results = batch.compute((ReflectivityOverQ, ReducibleData[SampleRun]))
        scale_factors = scale_for_reflectivity_overlap(
            results[ReflectivityOverQ].hist(),
            critical_edge_interval=scale_to_overlap
            if isinstance(scale_to_overlap, tuple | list)
            else None,
        )
        batch[ReducibleData[SampleRun]] = (
            scale_factors * results[ReducibleData[SampleRun]]
        )
        batch[ReflectivityOverQ] = scale_factors * results[ReflectivityOverQ]

    return batch.compute(target)
