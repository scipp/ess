# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Literal, NewType

import pytest
import sciline as sl
from pydantic import BaseModel, Field

from ess.reduce.parameter import ParameterRegistry, ParameterSpec
from ess.reduce.ui import (
    WorkflowWidget,
    get_parameter_widget_values,
    set_parameter_widget_values,
    workflow_widget,
)
from ess.reduce.widgets import PydanticModelWidget, PydanticParameterWidget
from ess.reduce.workflow import (
    assign_parameter_values,
    get_parameters,
    register_workflow,
    workflow_registry,
)


class Mode(Enum):
    fast = 'fast'
    slow = 'slow'


class FlatParams(BaseModel):
    count: int = Field(default=1, title="Count")
    scale: float = Field(default=2.0, title="Scale")
    enabled: bool = Field(default=True, title="Enabled")
    path: Path | None = Field(default=None, title="Path")
    mode: Mode = Field(default=Mode.fast, title="Mode")
    choice: Literal['a', 'b'] = Field(default='a', title="Choice")
    names: tuple[str, ...] = Field(default=("sample",), title="Names")


class GroupParams(BaseModel):
    value: int = 2


class NestedParams(BaseModel):
    group: GroupParams = Field(default_factory=GroupParams, title="Group")


Count = NewType("Count", int)
Label = NewType("Label", str)
Unused = NewType("Unused", int)


def test_pydantic_parameter_widget_creates_model_from_widget_values() -> None:
    widget = PydanticParameterWidget(FlatParams)

    widget.widgets['count'].value = 3
    widget.widgets['scale'].value = 0.5
    widget.widgets['enabled'].value = False
    widget.widgets['path'].value = 'data.nxs'
    widget.widgets['mode'].value = Mode.slow
    widget.widgets['choice'].value = 'b'
    widget.widgets['names'].value = 'a, b'

    params = widget.create_model()

    assert params == FlatParams(
        count=3,
        scale=0.5,
        enabled=False,
        path=Path('data.nxs'),
        mode=Mode.slow,
        choice='b',
        names=('a', 'b'),
    )


def test_pydantic_parameter_widget_reports_validation_errors() -> None:
    class Params(BaseModel):
        value: int = Field(default=1, ge=0)

    widget = PydanticParameterWidget(Params)
    widget.widgets['value'].value = -1

    is_valid, errors = widget.validate()

    assert not is_valid
    assert 'value' in errors


def test_pydantic_parameter_widget_set_values_accepts_none_for_optional_text() -> None:
    class Params(BaseModel):
        value: str | None = None

    widget = PydanticParameterWidget(Params, initial_values=Params())

    assert widget.widgets['value'].value == ''
    assert widget.create_model() == Params(value=None)


def test_pydantic_parameter_widget_supports_nested_model_fields() -> None:
    class Params(BaseModel):
        group: GroupParams = Field(default_factory=GroupParams)

    widget = PydanticParameterWidget(Params, initial_values=Params())

    widget.widgets['group'].widget.widgets['value'].value = 5

    assert widget.create_model() == Params(group=GroupParams(value=5))


def test_pydantic_model_widget_supports_nested_parameter_groups() -> None:
    widget = PydanticModelWidget(NestedParams)
    group = widget.get_parameter_widget('group')
    assert group is not None
    group.widgets['value'].value = 4

    assert widget.parameter_values == NestedParams(group=GroupParams(value=4))


def provider(count: Count, label: Label) -> str:
    return f"{count}:{label}"


def make_workflow() -> sl.Pipeline:
    parameters = ParameterRegistry()
    parameters[Count] = ParameterSpec(
        model=int, category='General', title='Count', default=1, transform=Count
    )
    parameters[Label] = ParameterSpec(
        model=str,
        category='General',
        title='Label',
        default='fallback',
        transform=Label,
    )
    parameters[Unused] = ParameterSpec(
        model=int, category='General', title='Unused', default=2, transform=Unused
    )
    pipeline = sl.Pipeline([provider])
    pipeline[Label] = Label('a')
    pipeline.typical_outputs = (str,)
    pipeline.parameter_registry = parameters
    return pipeline


@contextmanager
def temporary_workflow_registry(
    *constructors: Callable[..., sl.Pipeline],
) -> Generator[None, None, None]:
    existence_flags = {
        constructor: constructor in workflow_registry for constructor in constructors
    }
    try:
        for constructor in constructors:
            register_workflow(title="Example")(constructor)
        yield
    finally:
        for constructor, existed in existence_flags.items():
            if not existed:
                workflow_registry.discard(constructor)


def test_register_workflow_with_metadata() -> None:
    with temporary_workflow_registry(make_workflow):
        record = workflow_registry.get(make_workflow)
        assert record.factory is make_workflow
        assert record.spec.title == "Example"

    with pytest.raises(KeyError):
        workflow_registry.get(make_workflow)


def _get_selection_widget(widget):
    return widget.children[0].children[0]


def test_workflow_registry_applied_to_selector() -> None:
    with temporary_workflow_registry(make_workflow):
        record = workflow_registry.get(make_workflow)
        selection_widget = _get_selection_widget(workflow_widget())

        assert ("Example", record) in selection_widget.options


def test_workflow_selection_builds_workflow_widget() -> None:
    with temporary_workflow_registry(make_workflow):
        record = workflow_registry.get(make_workflow)
        widget = workflow_widget()
        selection_widget = _get_selection_widget(widget)

        selection_widget.value = record

        assert len(widget.children[1].children) == 1
        selected_widget = widget.children[1].children[0]
        assert isinstance(selected_widget, WorkflowWidget)
        assert selected_widget.output_selection_box.typical_outputs_widget.options == (
            ('str', str),
        )
        assert selected_widget._pipeline.parameter_registry is not None


def test_get_parameters_filters_by_selected_output_and_uses_workflow_defaults() -> None:
    params = get_parameters(make_workflow(), (str,))

    assert set(params) == {Count, Label}
    assert params[Count].default == 1
    assert params[Label].default == 'a'


def test_assign_parameter_values_uses_key_specs() -> None:
    pipeline = make_workflow()
    params = get_parameters(pipeline, (str,))

    updated = assign_parameter_values(pipeline, {Count: 5, Label: 'x'}, params)

    assert updated.compute(str) == '5:x'


def test_workflow_widget_runs_with_keyed_parameter_values() -> None:
    registry = {}
    with temporary_workflow_registry(make_workflow):
        record = workflow_registry.get(make_workflow)
        widget = WorkflowWidget(record, result_registry=registry)
        widget.output_selection_box.typical_outputs_widget.value = (str,)
        widget.parameter_box.parameter_refresh_button.click()

        set_parameter_widget_values(widget, {Count: 5, Label: 'x'})
        widget.result_box.run_button.click()

    assert registry == {str: '5:x'}


def test_get_parameter_widget_values_returns_current_keyed_fields() -> None:
    with temporary_workflow_registry(make_workflow):
        record = workflow_registry.get(make_workflow)
        widget = WorkflowWidget(record)
        widget.output_selection_box.typical_outputs_widget.value = (str,)
        widget.parameter_box.parameter_refresh_button.click()

        set_parameter_widget_values(widget, {Count: 7})

        assert get_parameter_widget_values(widget)[Count] == {'value': 7}
