# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""This module provides tools for running workflows in a streaming fashion."""

from abc import ABC, abstractmethod
from collections import ChainMap
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import networkx as nx
import sciline
import scipp as sc

if TYPE_CHECKING:
    import graphviz

T = TypeVar('T')


def maybe_hist(value: T) -> T:
    """
    Convert value to a histogram if it is not already a histogram.

    This is the default pre-processing used by accumulators.

    Parameters
    ----------
    value:
        Value to be converted to a histogram.

    Returns
    -------
    :
        Histogram.
    """
    if not isinstance(value, sc.Variable | sc.DataArray):
        return value
    return value if value.bins is None else value.hist()


class Accumulator(ABC, Generic[T]):
    """
    Abstract base class for accumulators.

    Accumulators are used to accumulate values over multiple chunks.
    """

    def __init__(self, preprocess: Callable[[T], T] | None = maybe_hist) -> None:
        """
        Parameters
        ----------
        preprocess:
            Preprocessing function to be applied to pushed values prior to accumulation.
        """
        self._preprocess = preprocess

    def push(self, value: T) -> None:
        """
        Push a value to the accumulator.

        Parameters
        ----------
        value:
            Value to be pushed to the accumulator.
        """
        if self._preprocess is not None:
            value = self._preprocess(value)
        self._do_push(value)

    @abstractmethod
    def _do_push(self, value: T) -> None: ...

    @property
    def is_empty(self) -> bool:
        """
        Check if the accumulator is empty.

        Returns
        -------
        :
            True if the accumulator is empty, False otherwise.
        """
        return False

    @property
    def value(self) -> T:
        """
        Get the accumulated value.

        Returns
        -------
        :
            Accumulated value.

        Raises
        ------
        ValueError
            If the accumulator is empty.
        """
        if self.is_empty:
            raise ValueError("Cannot get value from empty accumulator")
        return self._get_value()

    @abstractmethod
    def _get_value(self) -> T:
        """Return the accumulated value, assuming it exists."""

    @abstractmethod
    def clear(self) -> None:
        """
        Clear the accumulator, resetting it to its initial state.
        """

    def on_finalize(self) -> None:
        """
        Called after finalize retrieves value.

        Override this method to perform custom cleanup after each finalize cycle.
        The default implementation does nothing.
        """


class EternalAccumulator(Accumulator[T]):
    """
    Simple accumulator that adds pushed values immediately.

    Does not support event data.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._value: T | None = None

    @property
    def is_empty(self) -> bool:
        return self._value is None

    def _get_value(self) -> T:
        return deepcopy(self._value)

    def _do_push(self, value: T) -> None:
        if self._value is None:
            self._value = deepcopy(value)
        else:
            self._value += value

    def clear(self) -> None:
        """Clear the accumulated value."""
        self._value = None


class MeanAccumulator(EternalAccumulator[T]):
    """
    Accumulator that computes the mean of pushed values.

    Does not support event data.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._count = 0

    def _do_push(self, value: T) -> None:
        super()._do_push(value)
        self._count += 1

    def _get_value(self) -> T:
        return super()._get_value() / self._count

    def clear(self) -> None:
        """Clear the accumulated value and count."""
        super().clear()
        self._count = 0


class RollingAccumulator(Accumulator[T]):
    """
    Accumulator that adds pushed values to a rolling window.

    Does not support event data.
    """

    def __init__(self, window: int = 10, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        window:
            Size of the rolling window.
        """
        super().__init__(**kwargs)
        self._window = window
        self._values: list[T] = []

    @property
    def is_empty(self) -> bool:
        return len(self._values) == 0

    def _get_value(self) -> T:
        # Naive and potentially slow implementation if values and/or window are large!
        return sc.reduce(self._values).sum()

    def _do_push(self, value: T) -> None:
        self._values.append(value)
        if len(self._values) > self._window:
            self._values.pop(0)

    def clear(self) -> None:
        """Clear the accumulated values."""
        self._values = []


class MinAccumulator(Accumulator):
    """Keeps the minimum value seen so far.

    Only supports scalar values.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cur_min: sc.Variable | None = None

    def _do_push(self, value: sc.Variable) -> None:
        if self._cur_min is None:
            self._cur_min = value
        else:
            self._cur_min = min(self._cur_min, value)

    @property
    def is_empty(self) -> bool:
        """Check if the accumulator has collected a minimum value."""
        return self._cur_min is None

    def _get_value(self) -> Any:
        return self._cur_min

    def clear(self) -> None:
        """Clear the accumulated minimum value."""
        self._cur_min = None


class MaxAccumulator(Accumulator):
    """Keeps the maximum value seen so far.

    Only supports scalar values.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cur_max: sc.Variable | None = None

    @property
    def is_empty(self) -> bool:
        """Check if the accumulator has collected a maximum value."""
        return self._cur_max is None

    def _do_push(self, value: sc.Variable) -> None:
        if self._cur_max is None:
            self._cur_max = value
        else:
            self._cur_max = max(self._cur_max, value)

    def _get_value(self) -> sc.Variable | None:
        return self._cur_max

    def clear(self) -> None:
        """Clear the accumulated maximum value."""
        self._cur_max = None


class StreamProcessor:
    """
    Wrap a base workflow for streaming processing of chunks.

    Note that this class can not determine if the workflow is valid for streamed
    processing based on the input keys. In particular, it is the responsibility of the
    user to ensure that the workflow is "linear" with respect to the dynamic keys up to
    the accumulation keys.

    Similarly, the stream processor cannot determine from the workflow structure whether
    context updates are compatible with the accumulated data. Accumulators are not
    cleared automatically. This is best illustrated with an example:

    - If the context is the detector rotation angle, and we accumulate I(Q) (or a
      prerequisite of I(Q)), then updating the detector angle context is compatible with
      previous data, assuming Q for each new chunk is computed based on the angle.
    - If the context is the sample temperature, and we accumulate I(Q), then updating
      the temperature context is not compatible with previous data. Accumulating I(Q, T)
      could be compatible in this case.

    Since the correctness cannot be determined from the workflow structure, we recommend
    implementing processing steps in a way to catch such problems. For example, adding
    the temperature as a coordinate to the I(Q) data array should allow for
    automatically raising in the accumulator if the temperature changes.

    The workflow is split into :py:class:`sciline.Stage` objects, built together so
    that everything depending on neither context nor chunks is computed once, at
    construction:

    - The context stage computes, from the context keys, what the other stages read
      that depends on the context. It runs when the context changes, and its results
      are held until the next change.
    - One chunk stage per group of accumulators that depend on the same dynamic keys.
      It runs for each chunk with these keys, and its results are pushed to the
      accumulators.
    - The finalize stage computes the targets from the accumulated values.

    Targets that are accumulation keys, or that depend on the context only, are taken
    from the accumulators and from the results of the context stage.
    """

    def __init__(
        self,
        base_workflow: sciline.Pipeline,
        *,
        dynamic_keys: tuple[sciline.typing.Key, ...],
        context_keys: tuple[sciline.typing.Key, ...] = (),
        target_keys: tuple[sciline.typing.Key, ...],
        accumulators: dict[sciline.typing.Key, Accumulator | Callable[..., Accumulator]]
        | tuple[sciline.typing.Key, ...],
        allow_bypass: bool = False,
    ) -> None:
        """
        Create a stream processor.

        Parameters
        ----------
        base_workflow:
            Workflow to be used for processing chunks.
        dynamic_keys:
            Keys that are expected to be updated with each chunk. These keys cannot
            depend on each other or on context_keys.
        context_keys:
            Keys that define context for processing chunks and may change occasionally.
            These keys cannot overlap with dynamic_keys or depend on each other or on
            dynamic_keys.
        target_keys:
            Keys to be computed and returned.
        accumulators:
            Keys at which to accumulate values and their accumulators. If a tuple is
            passed, :py:class:`EternalAccumulator` is used for all keys. Otherwise, a
            dict mapping keys to accumulator instances can be passed. If a dict value is
            a callable, base_workflow.bind_and_call(value) is used to make an instance.
        allow_bypass:
            If True, allow bypassing accumulators for keys that are not in the
            accumulators dict. This is useful for dynamic keys that are not "terminated"
            in any accumulator. USE WITH CARE! This will lead to incorrect results
            unless the values for these keys are valid for all chunks comprised in the
            final accumulators at the point where :py:meth:`finalize` is called.

        Raises
        ------
        ValueError
            If the dynamic and context keys overlap or depend on each other, or if
            a target depends on a dynamic key other than through an accumulator and
            ``allow_bypass`` is False.
        """
        self._dynamic_keys = set(dynamic_keys)
        self._context_keys = set(context_keys)
        self._target_keys = target_keys
        _validate_streaming_keys(base_workflow, self._dynamic_keys, self._context_keys)
        special_keys = self._dynamic_keys | self._context_keys
        self._accumulators = _make_accumulators(
            base_workflow,
            accumulators,
            _used_keys(
                base_workflow, [k for k in target_keys if k not in special_keys]
            ),
        )
        (
            self._context_stage,
            self._chunk_stages,
            self._finalize_stage,
        ) = _prepare_stages(
            base_workflow,
            dynamic_keys=self._dynamic_keys,
            context_keys=self._context_keys,
            target_keys=target_keys,
            accumulator_keys=tuple(self._accumulators),
            allow_bypass=allow_bypass,
        )
        self._bypassed_keys = self._dynamic_keys.intersection(
            self._finalize_stage.inputs
        )
        # Values kept between calls: the context, what the context stage computed from
        # it, and the latest chunk of each bypassed dynamic key.
        self._held: dict[sciline.typing.Key, Any] = {}

    def set_context(self, context: dict[sciline.typing.Key, Any]) -> None:
        """
        Set the context for processing chunks.

        Context keys not given keep their value. Everything derived from the context
        is recomputed, unless the given keys are only read directly.

        Parameters
        ----------
        context:
            Context to be set.
        """
        for key in context:
            if key not in self._context_keys:
                raise ValueError(f"Key '{key}' is not a context key")
        self._held.update(context)
        stage = self._context_stage
        # Until every context key the stage needs has been set, it cannot run.
        # Stages reading its results then fail, naming the keys that are not set.
        if not context.keys().isdisjoint(stage.inputs) and self._held.keys() >= set(
            stage.inputs
        ):
            self._held.update(stage.run(self._values_for(stage.inputs, self._held)))

    def add_chunk(
        self, chunks: dict[sciline.typing.Key, Any]
    ) -> dict[sciline.typing.Key, Any]:
        """
        Legacy interface for accumulating values from chunks and finalizing the result.

        It is recommended to use :py:meth:`accumulate` and :py:meth:`finalize` instead.

        Parameters
        ----------
        chunks:
            Chunks to be processed.

        Returns
        -------
        :
            Finalized result.
        """
        self.accumulate(chunks)
        return self.finalize()

    def accumulate(self, chunks: dict[sciline.typing.Key, Any]) -> None:
        """
        Accumulate values from chunks without finalizing the result.

        Parameters
        ----------
        chunks:
            Chunks to be processed.

        Raises
        ------
        ValueError
            If non-dynamic keys are provided in chunks.
            If accumulator computation requires dynamic keys not provided in chunks.
        """
        non_dynamic = set(chunks) - self._dynamic_keys
        if non_dynamic:
            raise ValueError(
                f"Can only update dynamic keys. Got non-dynamic keys: {non_dynamic}"
            )

        stages = []
        for stage in self._chunk_stages:
            needed = self._dynamic_keys.intersection(stage.inputs)
            if needed.isdisjoint(chunks):
                continue
            if not needed.issubset(chunks):
                raise ValueError(
                    f"Accumulators {stage.outputs} require dynamic keys "
                    f"{needed - chunks.keys()} not provided in the current chunk."
                )
            stages.append(stage)

        # Only what finalize() reads directly outlives the call.
        self._held.update({k: v for k, v in chunks.items() if k in self._bypassed_keys})
        values = ChainMap(chunks, self._held)
        results = [
            stage.run(self._values_for(stage.inputs, values)) for stage in stages
        ]
        for result in results:
            for key, value in result.items():
                self._accumulators[key].push(value)

    def finalize(self) -> dict[sciline.typing.Key, Any]:
        """
        Get the final result by computing the target keys based on accumulated values.

        Returns
        -------
        :
            Finalized result.
        """
        stage = self._finalize_stage
        accumulated = {key: acc.value for key, acc in self._accumulators.items()}
        values = ChainMap(accumulated, self._held)
        computed = stage.run(self._values_for(stage.inputs, values))
        result = self._values_for(self._target_keys, values.new_child(computed))
        for accumulator in self._accumulators.values():
            accumulator.on_finalize()
        return result

    def clear(self) -> None:
        """
        Clear all accumulators, resetting them to their initial state.

        This is useful for restarting a streaming computation without
        creating a new StreamProcessor instance.
        """
        for accumulator in self._accumulators.values():
            accumulator.clear()

    def _values_for(
        self,
        keys: Iterable[sciline.typing.Key],
        values: Mapping[sciline.typing.Key, Any],
    ) -> dict[sciline.typing.Key, Any]:
        if missing := [key for key in keys if key not in values]:
            unset = [k for k in self._context_stage.inputs if k not in self._held]
            hint = f' Context keys {unset} have not been set.' if unset else ''
            raise ValueError(f'No value was fed for {missing}.{hint}')
        return {key: values[key] for key in keys}

    def visualize(
        self,
        compact: bool = False,
        mode: Literal['data', 'task', 'both'] = 'data',
        cluster_generics: bool = True,
        cluster_color: str | None = '#f0f0ff',
        show_legend: bool = True,
        show_static_dependencies: bool = True,
        **kwargs: Any,
    ) -> 'graphviz.Digraph':
        """
        Visualize the workflow, styling nodes by when they are computed.

        - Static nodes (gray): computed once, at construction.
        - Static cached nodes (gray, thick border): static values that are kept.
        - Context keys (blue, thick border): input entry points for context.
        - Context-dependent nodes (light blue): computed when the context changes.
        - Dynamic keys (green, thick border): input entry points for chunks.
        - Dynamic nodes (light green): computed for each chunk.
        - Accumulator keys (orange cylinder): aggregation points.
        - Finalize nodes (plum): computed from accumulators during finalize.
        - Target keys (double border): final outputs.

        Parameters
        ----------
        compact:
            If True, parameter-table-dependent branches are collapsed.
        mode:
            'data' shows only data nodes, 'task' shows task nodes, 'both' shows all.
        cluster_generics:
            If True, generic products are grouped into clusters.
        cluster_color:
            Background color of clusters. If None, clusters are dotted.
        show_legend:
            If True, add a legend explaining the node styles.
        show_static_dependencies:
            If True (default), show all static nodes including dependencies of cached
            nodes. If False, hide the ancestors of cached nodes to simplify the graph.
        **kwargs:
            Additional arguments passed to graphviz.Digraph.

        Returns
        -------
        :
            A graphviz.Digraph with styled nodes.
        """
        from sciline.visualize import (
            CACHED_STYLE,
            INPUT_STYLE,
            ONCE_STYLE,
            OUTPUT_STYLE,
            PER_CALL_STYLE,
        )

        stages = (self._context_stage, *self._chunk_stages, self._finalize_stage)
        cached = {k for s in stages for k in s.cached}
        # Where a key is in several groups, later groups take precedence.
        groups = {
            'Static': (
                ONCE_STYLE,
                {k for s in stages for k in s.keys if k not in s.per_call} - cached,
            ),
            'Static (cached)': (CACHED_STYLE, cached),
            'Context-dependent': (
                {'style': 'filled', 'fillcolor': '#d4e8f4'},
                _computed_per_call(self._context_stage),
            ),
            'Context key (input)': (
                {**INPUT_STYLE, 'fillcolor': '#87CEEB'},
                self._context_keys,
            ),
            'Dynamic (per chunk)': (
                PER_CALL_STYLE,
                _computed_per_call(*self._chunk_stages),
            ),
            'Dynamic key (input)': (INPUT_STYLE, self._dynamic_keys),
            'Accumulator': (
                {'style': 'filled', 'fillcolor': '#FFB347', 'shape': 'cylinder'},
                self._accumulators,
            ),
            'Finalize (from accum.)': (
                {'style': 'filled', 'fillcolor': '#DDA0DD'},
                _computed_per_call(self._finalize_stage),
            ),
            'Target (output)': (OUTPUT_STYLE, self._target_keys),
        }
        return sciline.visualize_stages(
            *stages,
            groups=groups,
            show_legend=show_legend,
            show_cached_ancestors=show_static_dependencies,
            compact=compact,
            mode=mode,
            cluster_generics=cluster_generics,
            cluster_color=cluster_color,
            **kwargs,
        )


def _validate_streaming_keys(
    base_workflow: sciline.Pipeline,
    dynamic_keys: set[sciline.typing.Key],
    context_keys: set[sciline.typing.Key],
) -> None:
    """Raise unless dynamic and context keys are disjoint and mutually independent."""
    if overlap := dynamic_keys & context_keys:
        raise ValueError(f"Keys cannot be both dynamic and context: {overlap}")
    graph = base_workflow.underlying_graph
    special_keys = dynamic_keys | context_keys
    for key in special_keys:
        if key not in graph:
            continue
        if special_ancestors := nx.ancestors(graph, key) & special_keys:
            downstream = 'Dynamic' if key in dynamic_keys else 'Context'
            raise ValueError(
                f"{downstream} key '{key}' depends on other dynamic/context keys: "
                f"{special_ancestors}. This is not supported."
            )


def _used_keys(
    workflow: sciline.Pipeline, keys: Iterable[sciline.typing.Key]
) -> frozenset[sciline.typing.Key]:
    """Keys that computing ``keys`` uses, with the workflow cut nowhere.

    This builds a stage only to inspect it, so nothing is computed.
    """
    return sciline.Stage(workflow, outputs=keys, inputs=()).keys


def _prepare_stages(
    workflow: sciline.Pipeline,
    *,
    dynamic_keys: set[sciline.typing.Key],
    context_keys: set[sciline.typing.Key],
    target_keys: tuple[sciline.typing.Key, ...],
    accumulator_keys: tuple[sciline.typing.Key, ...],
    allow_bypass: bool,
) -> tuple[sciline.Stage, tuple[sciline.Stage, ...], sciline.Stage]:
    """
    Build the context stage, the chunk stages, and the finalize stage.

    The chunk and finalize stages are first built on their own, cut only at the
    dynamic and accumulation keys, to find what they cache that depends on the
    context: this is what the context stage computes. These stages are only
    inspected, so nothing is computed until all stages are built together.
    """
    # Accumulators reading the same dynamic keys share a chunk stage. An accumulator
    # that reads none is never pushed to.
    groups: dict[frozenset[sciline.typing.Key], list[sciline.typing.Key]] = {}
    for key in accumulator_keys:
        if dynamic := _used_keys(workflow, (key,)) & dynamic_keys:
            groups.setdefault(dynamic, []).append(key)
    chunk_probes = [
        sciline.Stage(workflow, outputs=keys, inputs=dynamic)
        for dynamic, keys in groups.items()
    ]

    special_keys = dynamic_keys | context_keys
    finalized = tuple(
        k for k in target_keys if k not in accumulator_keys and k not in special_keys
    )
    used = _used_keys(workflow, finalized)
    accumulated = tuple(k for k in accumulator_keys if k in used)
    finalize_probe = sciline.Stage(workflow, outputs=finalized, inputs=accumulated)
    if bypassed := tuple(k for k in dynamic_keys if k in finalize_probe.keys):
        if not allow_bypass:
            raise ValueError(
                f"Targets depend on dynamic keys {bypassed} other than through an "
                "accumulator. Pass allow_bypass=True to compute them from the "
                "latest chunk."
            )
        finalize_probe = sciline.Stage(
            workflow, outputs=finalized, inputs=(*accumulated, *bypassed)
        )

    probes = (*chunk_probes, finalize_probe)
    context = tuple(k for k in context_keys if any(k in p.keys for p in probes))
    cached = tuple(dict.fromkeys(k for p in probes for k in p.cached))
    forwarded: tuple[sciline.typing.Key, ...] = ()
    if context:
        derived = sciline.Stage(workflow, outputs=cached, inputs=context)
        # Context keys that the stages read directly come from the held context, so
        # the context stage only computes what is derived from them.
        forwarded = tuple(k for k in derived.per_call_outputs if k not in context_keys)
    context_stage, *chunk_stages, finalize_stage = sciline.prepare_stages(
        workflow,
        inputs=dict.fromkeys(
            (
                *context,
                *forwarded,
                *(k for dynamic in groups for k in dynamic),
                *bypassed,
                *accumulated,
            )
        ),
        outputs=[
            forwarded,
            *groups.values(),
            tuple(k for k in finalized if k not in forwarded),
        ],
    )
    return context_stage, tuple(chunk_stages), finalize_stage


def _make_accumulators(
    base_workflow: sciline.Pipeline,
    accumulators: dict[sciline.typing.Key, Accumulator | Callable[..., Accumulator]]
    | tuple[sciline.typing.Key, ...],
    used: frozenset[sciline.typing.Key],
) -> dict[sciline.typing.Key, Accumulator]:
    """
    Instantiate the accumulators that the target keys actually depend on.

    Accumulators that are not used given the target keys are dropped. Callables are
    bound and called on the workflow, which allows for initializing an accumulator
    with arguments that depend on the workflow, such as bin edges, which would
    otherwise be hard to obtain.
    """
    if not isinstance(accumulators, dict):
        accumulators = {key: EternalAccumulator() for key in accumulators}
    return {
        key: value
        if isinstance(value, Accumulator)
        else base_workflow.bind_and_call(value)
        for key, value in accumulators.items()
        if key in used
    }


def _computed_per_call(*stages: sciline.Stage) -> set[sciline.typing.Key]:
    """Keys that the stages compute on each call, excluding their inputs."""
    return {k for s in stages for k in s.per_call if k not in s.inputs}
