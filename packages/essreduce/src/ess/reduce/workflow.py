# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, MutableSet, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import networkx as nx
from sciline import Pipeline
from sciline._utils import key_name
from sciline.typing import Key

from .parameter import ParameterSpec, keep_default, parameter_registry

T = TypeVar("T")
WorkflowFactory = Callable[..., Pipeline]


@dataclass(frozen=True)
class WorkflowSpec:
    """Metadata needed to generate interfaces for a workflow."""

    name: str
    title: str | None = None
    description: str | None = None
    version: str | None = None

    @classmethod
    def from_factory(
        cls,
        factory: WorkflowFactory,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        version: str | None = None,
    ) -> WorkflowSpec:
        return cls(
            name=name or factory.__name__,
            title=title,
            description=description,
            version=version,
        )


@dataclass(frozen=True)
class RegisteredWorkflow:
    """A workflow factory and the interface specification attached to it."""

    factory: WorkflowFactory
    spec: WorkflowSpec

    def __call__(self, *args: Any, **kwargs: Any) -> Pipeline:
        return self.factory(*args, **kwargs)

    @property
    def __name__(self) -> str:
        return self.factory.__name__


class WorkflowRegistry(MutableSet):
    def __init__(self):
        self._workflows: dict[str, RegisteredWorkflow] = {}

    def __contains__(self, item: object) -> bool:
        return item in self._workflows.values() or any(
            item is workflow.factory for workflow in self._workflows.values()
        )

    def __iter__(self):
        return iter(self._workflows.values())

    def __len__(self) -> int:
        return len(self._workflows)

    def add(
        self,
        value: WorkflowFactory | RegisteredWorkflow,
        spec: WorkflowSpec | None = None,
    ) -> None:
        workflow = (
            value
            if isinstance(value, RegisteredWorkflow)
            else _make_registered_workflow(value, spec)
        )
        key = workflow.factory.__qualname__
        self._workflows[key] = workflow

    def discard(self, value: WorkflowFactory | RegisteredWorkflow) -> None:
        self._workflows = {
            k: v
            for k, v in self._workflows.items()
            if v != value and v.factory is not value
        }

    def get(self, value: WorkflowFactory | RegisteredWorkflow) -> RegisteredWorkflow:
        if isinstance(value, RegisteredWorkflow):
            return value
        key = value.__qualname__
        try:
            return self._workflows[key]
        except KeyError as e:
            raise KeyError(f"Workflow {value.__qualname__!r} is not registered.") from e


workflow_registry = WorkflowRegistry()


def _make_registered_workflow(
    factory: WorkflowFactory, spec: WorkflowSpec | None = None
) -> RegisteredWorkflow:
    spec = spec or WorkflowSpec.from_factory(factory)
    return RegisteredWorkflow(factory=factory, spec=spec)


def register_workflow(
    cls: WorkflowFactory | WorkflowSpec | None = None,
    /,
    *,
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    version: str | None = None,
) -> WorkflowFactory | Callable[[WorkflowFactory], WorkflowFactory]:
    """Register a workflow factory for generated user interfaces."""

    def decorator(factory: WorkflowFactory) -> WorkflowFactory:
        spec = (
            cls
            if isinstance(cls, WorkflowSpec)
            else WorkflowSpec.from_factory(
                factory,
                name=name,
                title=title,
                description=description,
                version=version,
            )
        )
        workflow_registry.add(factory, spec=spec)
        return factory

    if isinstance(cls, WorkflowSpec) or cls is None:
        return decorator
    if any(value is not None for value in (name, title, description, version)):
        raise TypeError(
            "Use register_workflow(...) as a decorator when passing workflow metadata."
        )
    return decorator(cls)


def create_workflow(workflow: WorkflowFactory | RegisteredWorkflow) -> Pipeline:
    """Create a pipeline from a registered workflow."""
    registered = (
        workflow_registry.get(workflow)
        if not isinstance(workflow, RegisteredWorkflow)
        else workflow
    )
    return registered.factory()


def _get_defaults_from_workflow(workflow: Pipeline) -> dict[Key, Any]:
    nodes = workflow.underlying_graph.nodes
    return {key: values["value"] for key, values in nodes.items() if "value" in values}


def get_typical_outputs(pipeline: Pipeline) -> tuple[Key, ...]:
    if (typical_outputs := getattr(pipeline, "typical_outputs", None)) is None:
        graph = pipeline.underlying_graph
        sink_nodes = [node for node, degree in graph.out_degree if degree == 0]
        return sorted(_with_pretty_names(sink_nodes), key=lambda x: x[0])
    return _with_pretty_names(typical_outputs)


def get_possible_outputs(pipeline: Pipeline) -> tuple[Key, ...]:
    return sorted(
        _with_pretty_names(tuple(pipeline.underlying_graph.nodes)), key=lambda x: x[0]
    )


def _with_pretty_names(outputs: Sequence[Key]) -> tuple[tuple[str, Key], ...]:
    """Add a more readable string representation without full module path."""
    return tuple((key_name(output), output) for output in outputs)


def get_parameters(
    pipeline: Pipeline, outputs: tuple[Key, ...]
) -> dict[Key, ParameterSpec]:
    """Return a dictionary of parameters for the workflow."""
    subgraph = set(outputs)
    graph = pipeline.underlying_graph
    for key in outputs:
        subgraph.update(nx.ancestors(graph, key))
    defaults = _get_defaults_from_workflow(pipeline)
    parameters = _get_parameter_registry(pipeline)
    return {
        key: spec.with_default(defaults.get(key, keep_default))
        for key, spec in parameters.items()
        if any(filter_key in subgraph for filter_key in spec.filter_keys)
    }


def assign_parameter_values(
    pipeline: Pipeline,
    values: dict[Key, Any],
    parameters: dict[Key, ParameterSpec] | None = None,
) -> Pipeline:
    """Set a value for a parameter in the pipeline."""
    pipeline = pipeline.copy()
    parameters = parameters or _get_parameter_registry(pipeline)
    for key, value in values.items():
        spec = parameters[key]
        if spec.transform is not None:
            value = spec.transform(value)
        if spec.apply is not None:
            pipeline = spec.apply(pipeline, value)
        else:
            pipeline[key] = value
    return pipeline


def _get_parameter_registry(pipeline: Pipeline):
    return getattr(pipeline, 'parameter_registry', parameter_registry)
