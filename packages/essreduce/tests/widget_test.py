# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Literal, NewType

import pytest
import sciline as sl
from ipywidgets import VBox
from pydantic import BaseModel, Field

from ess.reduce.parameter import ParameterRegistry, ParameterSpec
from ess.reduce.ui import (
    WorkflowWidget,
    get_parameter_widget_values,
    set_parameter_widget_values,
    workflow_widget,
)
from ess.reduce.widgets import (
    PydanticModelWidget,
    PydanticParameterValueWidget,
    PydanticParameterWidget,
)
from ess.reduce.workflow import (
    WorkflowSpec,
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


class LimitedParams(BaseModel):
    value: int = Field(default=1, ge=0)


Count = NewType("Count", int)
Label = NewType("Label", str)
Unused = NewType("Unused", int)
Limited = NewType("Limited", int)


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


def test_pydantic_parameter_widget_exposes_field_descriptions_as_help() -> None:
    class Params(BaseModel):
        count: int = Field(default=1, title='Count', description='Number of items.')

    widget = PydanticParameterWidget(Params)
    field = widget.widgets['count']

    assert 'Number of items.' in field.children[1].value
    field.value = 3
    assert widget.create_model() == Params(count=3)


def test_pydantic_parameter_value_widget_labels_composite_parameters() -> None:
    widget = PydanticParameterValueWidget(
        GroupParams,
        title='Grouped Parameter',
        description='Controls for the grouped parameter.',
        default=GroupParams(),
    )

    title = widget.children[0].children[0].value

    assert 'Grouped Parameter' in title
    assert 'Controls for the grouped parameter.' in title


def test_pydantic_parameter_value_widget_uses_title_as_placeholder_fallback() -> None:
    widget = PydanticParameterValueWidget(str | None, title='Sample Run', default=None)

    assert widget._widget.widgets['value'].placeholder == 'Sample Run'


def test_pydantic_model_widget_supports_nested_parameter_groups() -> None:
    widget = PydanticModelWidget(NestedParams)
    group = widget.get_parameter_widget('group')
    assert group is not None
    group.widgets['value'].value = 4

    assert widget.parameter_values == NestedParams(group=GroupParams(value=4))


def provider(count: Count, label: Label) -> str:
    return f"{count}:{label}"


test_parameters = ParameterRegistry()
test_parameters[Count] = ParameterSpec(
    model=int, category='General', title='Count', default=1, transform=Count
)
test_parameters[Label] = ParameterSpec(
    model=str,
    category='Text',
    title='Label',
    default='fallback',
    transform=Label,
)
test_parameters[Unused] = ParameterSpec(
    model=int, category='Other', title='Unused', default=2, transform=Unused
)


def make_workflow() -> sl.Pipeline:
    pipeline = sl.Pipeline([provider])
    pipeline[Label] = Label('a')
    return pipeline


def other_provider(unused: Unused) -> int:
    return unused


def limited_provider(limited: Limited) -> float:
    return float(limited)


def make_workflow_with_alternative_output() -> sl.Pipeline:
    pipeline = sl.Pipeline([provider, other_provider])
    pipeline[Label] = Label('a')
    return pipeline


def make_workflow_with_limited_parameter() -> sl.Pipeline:
    pipeline = sl.Pipeline([provider, limited_provider])
    pipeline[Label] = Label('a')
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
            register_workflow(
                parameters=test_parameters,
                typical_outputs=(str,),
                title="Example",
            )(constructor)
        yield
    finally:
        for constructor, existed in existence_flags.items():
            if not existed:
                workflow_registry.discard(constructor)


def test_register_workflow_with_metadata() -> None:
    with temporary_workflow_registry(make_workflow):
        spec = workflow_registry.get(make_workflow)
        assert spec.factory is make_workflow
        assert spec.parameters is test_parameters
        assert spec.typical_outputs == (str,)
        assert spec.title == "Example"

    with pytest.raises(KeyError):
        workflow_registry.get(make_workflow)


def _get_selection_widget(widget):
    return widget.children[0].children[0]


def _category_sections(widget: WorkflowWidget):
    return {
        section.get_title(0): section
        for section in widget.parameter_box._input_box.children
    }


def _category_expansion_state(widget: WorkflowWidget):
    return {
        title: section.selected_index
        for title, section in _category_sections(widget).items()
    }


def test_workflow_registry_applied_to_selector() -> None:
    with temporary_workflow_registry(make_workflow):
        spec = workflow_registry.get(make_workflow)
        selection_widget = _get_selection_widget(workflow_widget())

        assert ("Example", spec) in selection_widget.options


def test_workflow_selection_builds_workflow_widget() -> None:
    with temporary_workflow_registry(make_workflow):
        spec = workflow_registry.get(make_workflow)
        widget = workflow_widget()
        selection_widget = _get_selection_widget(widget)

        selection_widget.value = spec

        assert len(widget.children[1].children) == 1
        selected_widget = widget.children[1].children[0]
        assert isinstance(selected_widget, WorkflowWidget)
        assert selected_widget.output_selection_box.typical_outputs_widget.options == (
            ('str', str),
        )
        assert selected_widget.workflow is spec


def test_get_parameters_filters_by_selected_output_and_uses_workflow_defaults() -> None:
    params = get_parameters(make_workflow(), (str,), test_parameters)

    assert set(params) == {Count, Label}
    assert params[Count].default == 1
    assert params[Label].default == 'a'


def test_assign_parameter_values_uses_key_specs() -> None:
    pipeline = make_workflow()
    params = get_parameters(pipeline, (str,), test_parameters)

    updated = assign_parameter_values(pipeline, {Count: 5, Label: 'x'}, params)

    assert updated.compute(str) == '5:x'


def test_workflow_widget_runs_with_keyed_parameter_values() -> None:
    registry = {}
    with temporary_workflow_registry(make_workflow):
        spec = workflow_registry.get(make_workflow)
        widget = WorkflowWidget(spec, result_registry=registry)
        widget.output_selection_box.typical_outputs_widget.value = (str,)

        assert not hasattr(widget.parameter_box, 'parameter_refresh_button')
        assert set(widget.parameter_box.parameters) == {Count, Label}

        set_parameter_widget_values(widget, {Count: 5, Label: 'x'})
        widget.result_box.run_button.click()

    assert registry == {str: '5:x'}


def test_workflow_widget_clears_parameters_when_no_output_is_selected() -> None:
    with temporary_workflow_registry(make_workflow):
        spec = workflow_registry.get(make_workflow)
        widget = WorkflowWidget(spec)

        assert widget.result_box.run_button.disabled
        assert widget.parameter_box.parameters == {}

        widget.output_selection_box.typical_outputs_widget.value = (str,)
        assert not widget.result_box.run_button.disabled
        assert set(widget.parameter_box.parameters) == {Count, Label}

        widget.output_selection_box.typical_outputs_widget.value = ()
        assert widget.result_box.run_button.disabled
        assert widget.parameter_box.parameters == {}


def test_get_parameter_widget_values_returns_current_keyed_fields() -> None:
    with temporary_workflow_registry(make_workflow):
        spec = workflow_registry.get(make_workflow)
        widget = WorkflowWidget(spec)
        widget.output_selection_box.typical_outputs_widget.value = (str,)

        set_parameter_widget_values(widget, {Count: 7})

        assert get_parameter_widget_values(widget)[Count] == {'value': 7}


def test_parameter_categories_are_independently_open_and_full_width() -> None:
    with temporary_workflow_registry(make_workflow):
        spec = workflow_registry.get(make_workflow)
        widget = WorkflowWidget(spec)
        widget.output_selection_box.typical_outputs_widget.value = (str,)

        sections = widget.parameter_box._input_box.children

        assert [section.get_title(0) for section in sections] == ['General', 'Text']
        assert [section.selected_index for section in sections] == [0, 0]
        sections[0].selected_index = None
        assert [section.selected_index for section in sections] == [None, 0]
        assert isinstance(widget.parameter_box._input_box, VBox)
        assert widget.parameter_box.layout.width == '100%'
        assert widget.parameter_box._input_box.layout.width == '100%'
        assert all(section.layout.width == 'auto' for section in sections)
        assert all(section.children[0].layout.width == '100%' for section in sections)
        assert all(
            field.layout.width == 'auto'
            for parameter_widget in widget.parameter_box._input_widgets.values()
            for field in parameter_widget._widget.widgets.values()
        )


def test_parameter_category_expansion_state_is_preserved_by_category() -> None:
    widget = WorkflowWidget(
        WorkflowSpec.from_factory(
            make_workflow_with_alternative_output,
            parameters=test_parameters,
            typical_outputs=(str, int),
        )
    )
    output_widget = widget.output_selection_box.typical_outputs_widget

    output_widget.value = (str,)
    assert _category_expansion_state(widget) == {
        'General': 0,
        'Text': 0,
    }

    sections = _category_sections(widget)
    sections['Text'].selected_index = None
    output_widget.value = (str, int)
    assert _category_expansion_state(widget) == {
        'General': 0,
        'Text': None,
        'Other': 0,
    }

    sections = _category_sections(widget)
    sections['Other'].selected_index = None
    output_widget.value = (int,)
    assert _category_expansion_state(widget) == {
        'Other': None,
    }

    output_widget.value = (str, int)
    assert _category_expansion_state(widget) == {
        'General': 0,
        'Text': None,
        'Other': None,
    }


def test_parameter_refresh_preserves_invalid_fields_without_validating() -> None:
    parameters = ParameterRegistry()
    parameters[Count] = test_parameters[Count]
    parameters[Label] = test_parameters[Label]
    parameters[Limited] = ParameterSpec(
        model=LimitedParams,
        category='Limited',
        title='Limited',
        default=LimitedParams(),
        transform=lambda params: Limited(params.value),
    )
    widget = WorkflowWidget(
        WorkflowSpec.from_factory(
            make_workflow_with_limited_parameter,
            parameters=parameters,
            typical_outputs=(float, str),
        )
    )
    output_widget = widget.output_selection_box.typical_outputs_widget

    output_widget.value = (float,)
    widget.parameter_box._input_widgets[Limited]._widget.widgets['value'].value = -1
    output_widget.value = (float, str)

    limited_widget = widget.parameter_box._input_widgets[Limited]
    assert limited_widget._widget.widgets['value'].value == -1
    is_valid, errors = limited_widget.validate()
    assert not is_valid
    assert 'greater than or equal to 0' in errors
