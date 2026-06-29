# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSet, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from sciline import Pipeline
from sciline._utils import key_name
from sciline.handler import HandleAsComputeTimeException
from sciline.typing import Key

from .parameter import (
    ParameterRegistry,
    ParameterSpec,
    keep_default,
    parameter_registry,
)

T = TypeVar("T")
WorkflowFactory = Callable[..., Pipeline]


@dataclass(frozen=True)
class WorkflowSpec:
    """Workflow factory and metadata needed to generate user interfaces."""

    factory: WorkflowFactory
    parameters: Mapping[Key, ParameterSpec] = field(default_factory=ParameterRegistry)
    typical_outputs: tuple[Key, ...] | None = None
    name: str | None = None
    title: str | None = None
    description: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, 'name', self.factory.__name__)

    def __call__(self, *args: Any, **kwargs: Any) -> Pipeline:
        return self.factory(*args, **kwargs)

    @property
    def __name__(self) -> str:
        return self.factory.__name__

    def create_workflow(self) -> Pipeline:
        return self.factory()

    @classmethod
    def from_factory(
        cls,
        factory: WorkflowFactory,
        *,
        parameters: Mapping[Key, ParameterSpec] | None = None,
        typical_outputs: Sequence[Key] | None = None,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        version: str | None = None,
    ) -> WorkflowSpec:
        return cls(
            factory=factory,
            parameters=parameters if parameters is not None else ParameterRegistry(),
            typical_outputs=None if typical_outputs is None else tuple(typical_outputs),
            name=name,
            title=title,
            description=description,
            version=version,
        )


class WorkflowRegistry(MutableSet):
    def __init__(self):
        self._workflows: dict[str, WorkflowSpec] = {}

    def __contains__(self, item: object) -> bool:
        return item in self._workflows.values() or any(
            item is spec.factory for spec in self._workflows.values()
        )

    def __iter__(self):
        return iter(self._workflows.values())

    def __len__(self) -> int:
        return len(self._workflows)

    def add(self, value: WorkflowFactory | WorkflowSpec) -> None:
        if isinstance(value, WorkflowSpec):
            spec = value
        else:
            spec = WorkflowSpec.from_factory(value)
        key = spec.factory.__qualname__
        self._workflows[key] = spec

    def discard(self, value: WorkflowFactory | WorkflowSpec) -> None:
        self._workflows = {
            k: v
            for k, v in self._workflows.items()
            if v != value and v.factory is not value
        }

    def get(self, value: WorkflowFactory | WorkflowSpec) -> WorkflowSpec:
        if isinstance(value, WorkflowSpec):
            return value
        key = value.__qualname__
        try:
            return self._workflows[key]
        except KeyError as e:
            raise KeyError(f"Workflow {value.__qualname__!r} is not registered.") from e


workflow_registry = WorkflowRegistry()


def register_workflow(
    *,
    parameters: Mapping[Key, ParameterSpec] | None = None,
    typical_outputs: Sequence[Key] | None = None,
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    version: str | None = None,
) -> Callable[[WorkflowFactory], WorkflowFactory]:
    """Register a workflow factory for generated user interfaces."""

    def decorator(factory: WorkflowFactory) -> WorkflowFactory:
        workflow_registry.add(
            WorkflowSpec.from_factory(
                factory,
                parameters=parameters,
                typical_outputs=typical_outputs,
                name=name,
                title=title,
                description=description,
                version=version,
            )
        )
        return factory

    return decorator


def _get_defaults_from_workflow(workflow: Pipeline) -> dict[Key, Any]:
    nodes = workflow.underlying_graph.nodes
    return {key: values["value"] for key, values in nodes.items() if "value" in values}


def get_typical_outputs(
    pipeline: Pipeline, typical_outputs: Sequence[Key] | None = None
) -> tuple[Key, ...]:
    if typical_outputs is None:
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
    pipeline: Pipeline,
    outputs: tuple[Key, ...],
    parameters: Mapping[Key, ParameterSpec] = parameter_registry,
) -> dict[Key, ParameterSpec]:
    """Return a dictionary of parameters for the workflow."""
    required_keys = set(
        pipeline.get(outputs, handler=HandleAsComputeTimeException()).keys()
    )
    defaults = _get_defaults_from_workflow(pipeline)
    return {
        key: spec.with_default(defaults.get(key, keep_default))
        for key, spec in parameters.items()
        if key in required_keys
    }


def assign_parameter_values(
    pipeline: Pipeline,
    values: dict[Key, Any],
    parameters: Mapping[Key, ParameterSpec] = parameter_registry,
) -> Pipeline:
    """Set a value for a parameter in the pipeline."""
    pipeline = pipeline.copy()
    for key, value in values.items():
        spec = parameters[key]
        if spec.transform is not None:
            value = spec.transform(value)
        if spec.apply is not None:
            pipeline = spec.apply(pipeline, value)
        else:
            pipeline[key] = value
    return pipeline
