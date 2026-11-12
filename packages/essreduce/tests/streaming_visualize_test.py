# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import NewType

import graphviz
import sciline

from ess.reduce import streaming

# Test types for building workflows
DynamicA = NewType('DynamicA', float)
DynamicB = NewType('DynamicB', float)
StaticA = NewType('StaticA', float)
StaticB = NewType('StaticB', float)
Context1 = NewType('Context1', float)
ContextB = NewType('ContextB', float)
AccumA = NewType('AccumA', float)
AccumB = NewType('AccumB', float)
Intermediate = NewType('Intermediate', float)
Target = NewType('Target', float)


def make_static_a() -> StaticA:
    return StaticA(1.0)


def make_static_b(a: StaticA) -> StaticB:
    return StaticB(a * 2)


def make_intermediate(dyn: DynamicA, static: StaticB) -> Intermediate:
    return Intermediate(dyn + static)


def make_accum_a(inter: Intermediate) -> AccumA:
    return AccumA(inter)


def make_accum_b(dyn: DynamicB) -> AccumB:
    return AccumB(dyn)


def make_target(a: AccumA, b: AccumB) -> Target:
    return Target(a / b)


def make_context_dependent(ctx: Context1) -> StaticB:
    return StaticB(ctx * 2)


def node_style(source: str, key: type) -> str:
    """The last statement for the node of ``key``, which sets its style."""
    return next(
        line
        for line in reversed(source.splitlines())
        if line.lstrip('\t').startswith(f'{key.__name__} [')
    )


STATIC = 'fillcolor="#e8e8e8"'
CACHED = 'penwidth=2.5'
DYNAMIC_KEY = 'fillcolor="#90ee90"'
DYNAMIC = 'fillcolor="#d4f4d4"'
CONTEXT_KEY = 'fillcolor="#87CEEB"'
CONTEXT_DEPENDENT = 'fillcolor="#d4e8f4"'
ACCUMULATOR = 'shape=cylinder'
FINALIZE = 'fillcolor="#DDA0DD"'
TARGET = 'peripheries=2'


class TestNodeStyles:
    """Tests that nodes are styled by when they are computed."""

    def test_static_nodes_and_cached_nodes(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )
        source = processor.visualize().source

        # StaticB is cached since the per-chunk part reads it, StaticA is not.
        assert STATIC in node_style(source, StaticA)
        assert CACHED not in node_style(source, StaticA)
        assert STATIC in node_style(source, StaticB)
        assert CACHED in node_style(source, StaticB)

    def test_dynamic_keys_and_nodes(self) -> None:
        workflow = sciline.Pipeline(
            (
                make_static_a,
                make_static_b,
                make_intermediate,
                make_accum_a,
                make_accum_b,
                make_target,
            )
        )
        workflow[DynamicA] = None
        workflow[DynamicB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA, DynamicB),
            target_keys=(Target,),
            accumulators=(AccumA, AccumB),
        )
        source = processor.visualize().source

        assert DYNAMIC_KEY in node_style(source, DynamicA)
        assert DYNAMIC_KEY in node_style(source, DynamicB)
        assert DYNAMIC in node_style(source, Intermediate)

    def test_accumulator_finalize_and_target_nodes(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )
        source = processor.visualize().source

        assert ACCUMULATOR in node_style(source, AccumA)
        assert DYNAMIC not in node_style(source, AccumA)
        assert FINALIZE in node_style(source, Target)
        assert TARGET in node_style(source, Target)

    def test_context_keys_and_context_dependent_nodes(self) -> None:
        def from_context(ctx: Context1) -> StaticB:
            return StaticB(ctx)

        def from_static_and_dynamic(s: StaticB, d: DynamicA) -> AccumA:
            return AccumA(s + d)

        workflow = sciline.Pipeline((from_context, from_static_and_dynamic))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            context_keys=(Context1,),
            target_keys=(AccumA,),
            accumulators=(AccumA,),
        )
        source = processor.visualize().source

        assert CONTEXT_KEY in node_style(source, Context1)
        assert CONTEXT_DEPENDENT in node_style(source, StaticB)
        # AccumA depends on both context and dynamic, so it is computed per chunk
        assert CONTEXT_DEPENDENT not in node_style(source, AccumA)
        assert ACCUMULATOR in node_style(source, AccumA)

    def test_unused_dynamic_key_is_not_drawn(self) -> None:
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(StaticA,),
            accumulators=(),
        )

        assert 'DynamicA' not in processor.visualize().source


class TestVisualize:
    """Integration tests for the visualize method."""

    def test_returns_graphviz_digraph(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        result = processor.visualize()
        assert isinstance(result, graphviz.Digraph)

    def test_includes_legend_by_default(self) -> None:
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(),
            target_keys=(StaticA,),
            accumulators=(),
        )

        result = processor.visualize()
        source = result.source
        assert 'cluster_legend' in source
        assert 'Legend' in source

    def test_legend_can_be_disabled(self) -> None:
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(),
            target_keys=(StaticA,),
            accumulators=(),
        )

        result = processor.visualize(show_legend=False)
        source = result.source
        assert 'cluster_legend' not in source

    def test_dynamic_keys_have_thick_border(self) -> None:
        workflow = sciline.Pipeline((make_intermediate,))
        workflow[DynamicA] = None
        workflow[StaticB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Intermediate,),
            accumulators=(Intermediate,),
        )

        result = processor.visualize()
        source = result.source
        # Check that DynamicA node has penwidth styling
        assert 'DynamicA' in source
        assert 'penwidth=' in source

    def test_accumulator_keys_have_cylinder_shape(self) -> None:
        workflow = sciline.Pipeline((make_accum_a,))
        workflow[Intermediate] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(Intermediate,),
            target_keys=(AccumA,),
            accumulators=(AccumA,),
        )

        result = processor.visualize()
        source = result.source
        assert 'shape=cylinder' in source

    def test_passes_kwargs_to_sciline_visualize(self) -> None:
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(),
            target_keys=(StaticA,),
            accumulators=(),
        )

        # Test that compact mode is passed through
        result = processor.visualize(compact=True)
        assert isinstance(result, graphviz.Digraph)

    def test_excludes_nodes_unreachable_from_targets(self) -> None:
        """Visualization should only show nodes that are ancestors of target_keys."""
        # Create a workflow with two independent branches
        # Branch 1: StaticA -> StaticB -> Intermediate -> AccumA -> Target
        # Branch 2: DynamicB -> AccumB (not used if we only target AccumA)
        workflow = sciline.Pipeline(
            (
                make_static_a,
                make_static_b,
                make_intermediate,
                make_accum_a,
                make_accum_b,  # This creates an independent branch via DynamicB
            )
        )
        workflow[DynamicA] = None
        workflow[DynamicB] = None

        # Only target AccumA - AccumB and DynamicB should NOT appear
        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA, DynamicB),
            target_keys=(AccumA,),
            accumulators=(AccumA,),
        )

        result = processor.visualize()
        source = result.source

        # Nodes in the path to AccumA should be present
        assert 'AccumA' in source
        assert 'Intermediate' in source
        assert 'DynamicA' in source
        assert 'StaticA' in source
        assert 'StaticB' in source

        # Nodes NOT in the path to AccumA should be excluded
        assert 'AccumB' not in source
        assert 'DynamicB' not in source

    def test_excludes_ancestors_of_input_keys(self) -> None:
        """Ancestors of dynamic/context keys should not appear in visualization."""
        # Create a workflow where dynamic key has an ancestor
        UpstreamOfDynamic = NewType('UpstreamOfDynamic', float)

        def make_dynamic_from_upstream(u: UpstreamOfDynamic) -> DynamicA:
            return DynamicA(u)

        workflow = sciline.Pipeline(
            (make_dynamic_from_upstream, make_intermediate, make_accum_a)
        )
        workflow[UpstreamOfDynamic] = 1.0
        workflow[StaticB] = StaticB(2.0)

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(AccumA,),
            accumulators=(AccumA,),
        )

        result = processor.visualize()
        source = result.source

        # DynamicA should appear (it's an input key)
        assert 'DynamicA' in source
        # But UpstreamOfDynamic should NOT appear (it's upstream of an input key)
        assert 'UpstreamOfDynamic' not in source

    def test_show_static_dependencies_false_hides_ancestors_of_cached(self) -> None:
        """Setting show_static_dependencies=False hides ancestors of cached nodes."""
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        # With show_static_dependencies=True (default), StaticA should appear
        result_with = processor.visualize(show_static_dependencies=True)
        assert 'StaticA' in result_with.source

        # With show_static_dependencies=False, StaticA should be hidden
        result_without = processor.visualize(show_static_dependencies=False)
        assert 'StaticA' not in result_without.source
        # But StaticB (the cached node) should still appear
        assert 'StaticB' in result_without.source

    def test_show_static_dependencies_false_updates_legend(self) -> None:
        """Legend should not show 'Static' entry when show_static_dependencies=False."""
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(),
            target_keys=(StaticA,),
            accumulators=(),
        )

        result = processor.visualize(show_static_dependencies=False)
        source = result.source

        # Should have 'Static (cached)' but not standalone 'Static'
        assert 'label="Static (cached)"' in source
        assert 'label=Static ' not in source
