# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from typing import Any

import ipywidgets as widgets
import sciline as sl
from IPython import display
from pydantic_core import PydanticUndefined
from sciline.typing import Key

from .parameter import ParameterSpec, keep_default
from .widgets import (
    PydanticParameterValueWidget,
    Spinner,
    full_width_layout,
)
from .workflow import (
    WorkflowSpec,
    assign_parameter_values,
    get_parameters,
    get_possible_outputs,
    get_typical_outputs,
    workflow_registry,
)


def _wrap_foldable(
    wrapped: widgets.Widget, title: str | None = None, *, expanded: bool = False
) -> widgets.Accordion:
    accordion = widgets.Accordion(
        [wrapped],
        layout=full_width_layout(width='auto', height='auto'),
        titles=(title,),
    )
    if expanded:
        accordion.selected_index = 0
    return accordion


class OutputSelectionWidget(widgets.VBox):
    def __init__(
        self,
        workflow: sl.Pipeline,
        typical_outputs: tuple[Key, ...] | None = None,
        **kwargs,
    ):
        self.typical_outputs_widget = widgets.SelectMultiple(
            options=get_typical_outputs(workflow, typical_outputs),
            layout=full_width_layout(width='auto', height='250px'),
        )
        self.possible_outputs_widget = widgets.SelectMultiple(
            options=get_possible_outputs(workflow),
            layout=full_width_layout(width='auto', height='auto'),
        )
        _typical_selection = _wrap_foldable(
            self.typical_outputs_widget, title='Typical Outputs', expanded=True
        )
        _possible_selection = _wrap_foldable(
            self.possible_outputs_widget, title='Extended Outputs'
        )
        super().__init__([_typical_selection, _possible_selection], **kwargs)

    @property
    def value(self) -> set[Key]:
        return set(
            self.typical_outputs_widget.value + self.possible_outputs_widget.value
        )


class ParameterBox(widgets.VBox):
    def __init__(
        self,
        registry_getter: Callable[[], dict[Key, ParameterSpec]],
        **kwargs,
    ):
        self._registry_getter = registry_getter
        self._input_registry: dict[Key, ParameterSpec] = {}
        self._input_widgets: dict[Key, PydanticParameterValueWidget] = {}
        self._expanded_categories: dict[str, bool] = {}
        self._input_box = widgets.VBox(layout=full_width_layout())
        kwargs.setdefault('layout', full_width_layout())
        super().__init__([self._input_box], **kwargs)

    def refresh(self) -> None:
        self._remember_expanded_categories()
        existing_fields = self.get_fields()
        new_input_parameters = dict(self._registry_getter())
        new_input_widgets: dict[Key, PydanticParameterValueWidget] = {}

        grouped: dict[str, list[widgets.Widget]] = {}
        for key, spec in new_input_parameters.items():
            widget = _create_parameter_widget(spec)
            if key in existing_fields:
                widget.set_fields(existing_fields[key])
            new_input_widgets[key] = widget
            grouped.setdefault(spec.category, []).append(widget)

        self._input_registry = new_input_parameters
        self._input_widgets = new_input_widgets

        if not grouped:
            self._input_box.children = [
                widgets.HTML("<em>No parameters</em>", layout=full_width_layout())
            ]
            return

        self._input_box.children = [
            _wrap_foldable(
                widgets.VBox(grouped[category], layout=full_width_layout()),
                title=category,
                expanded=self._expanded_categories.get(category, True),
            )
            for category in grouped
        ]

    def _remember_expanded_categories(self) -> None:
        for child in self._input_box.children:
            if not isinstance(child, widgets.Accordion):
                continue
            title = child.get_title(0)
            if title is not None:
                self._expanded_categories[title] = child.selected_index == 0

    @property
    def value(self) -> dict[Key, Any]:
        return {key: widget.value for key, widget in self._input_widgets.items()}

    @property
    def parameters(self) -> dict[Key, ParameterSpec]:
        return dict(self._input_registry)

    def validate(self) -> tuple[bool, list[str]]:
        errors = []
        for key, widget in self._input_widgets.items():
            is_valid, error_msg = widget.validate()
            if not is_valid:
                spec = self._input_registry[key]
                errors.append(f"{spec.name}: {error_msg}")
                widget.set_error_state(True, error_msg)
            else:
                widget.set_error_state(False, "")
        return len(errors) == 0, errors

    def set_values(self, values: dict[Key, Any]) -> None:
        for key, value in values.items():
            if key in self._input_widgets:
                self._input_widgets[key].set_value(value)

    def get_values(self) -> dict[Key, Any]:
        return {key: widget.value for key, widget in self._input_widgets.items()}

    def get_fields(self) -> dict[Key, dict[str, Any]]:
        return {key: widget.get_fields() for key, widget in self._input_widgets.items()}


def _create_parameter_widget(spec: ParameterSpec) -> PydanticParameterValueWidget:
    default = spec.default if spec.default is not keep_default else PydanticUndefined
    return PydanticParameterValueWidget(
        spec.model,
        title=spec.name,
        description=spec.description,
        default=default,
    )


class ResultBox(widgets.VBox):
    def __init__(
        self,
        workflow_runner: Callable[[], dict[type, Any]],
        result_registry: dict | None = None,
        **kwargs,
    ):
        self.output = widgets.Output()
        self.run_button = widgets.Button(
            description='Run',
            disabled=False,
            button_style='success',
            tooltip='Run',
        )
        output_clear_button = widgets.Button(
            description='Clear Output',
            button_style='warning',
            tooltip='Clear Output',
        )

        def run_workflow(_: widgets.Button) -> None:
            self.output.clear_output()
            with self.output:
                display.display(Spinner())
                try:
                    compute_result = workflow_runner()
                except Exception as e:
                    display.clear_output()
                    display.display(widgets.HTML(f"<pre>{type(e).__name__}: {e}</pre>"))
                    return
                display.clear_output()
                if result_registry is not None:
                    result_registry.clear()
                    result_registry.update(compute_result)
                for i in compute_result.values():
                    display.display(i)

        def clear_output(_: widgets.Button) -> None:
            self.output.clear_output()

        self.run_button.on_click(run_workflow)
        output_clear_button.on_click(clear_output)
        button_box = widgets.HBox([self.run_button, output_clear_button])
        super().__init__([button_box, self.output], **kwargs)


class WorkflowWidget(widgets.TwoByTwoLayout):
    def __init__(
        self,
        workflow: WorkflowSpec,
        result_registry: dict | None = None,
        **kwargs,
    ):
        self.workflow = workflow
        self._pipeline = self.workflow.create_workflow()
        self.output_selection_box = OutputSelectionWidget(
            self._pipeline, self.workflow.typical_outputs
        )

        def registry_getter() -> dict[Key, ParameterSpec]:
            return get_parameters(
                self._pipeline,
                tuple(self.output_selection_box.value),
                self.workflow.parameters,
            )

        self.parameter_box = ParameterBox(registry_getter)

        def workflow_runner() -> dict[type, Any]:
            is_valid, errors = self.parameter_box.validate()
            if not is_valid:
                raise ValueError('\n'.join(errors))
            pipeline = assign_parameter_values(
                self._pipeline,
                self.parameter_box.value,
                self.parameter_box.parameters,
            )
            return pipeline.compute(self.output_selection_box.value)

        self.result_box = ResultBox(workflow_runner, result_registry)

        def refresh_parameters(_) -> None:
            self.parameter_box.refresh()
            self.result_box.output.clear_output()
            has_outputs = bool(self.output_selection_box.value)
            self.result_box.run_button.disabled = not has_outputs
            self.result_box.run_button.tooltip = (
                'Run' if has_outputs else 'Select output quantities.'
            )

        for output_selection_widget in (
            self.output_selection_box.typical_outputs_widget,
            self.output_selection_box.possible_outputs_widget,
        ):
            output_selection_widget.observe(refresh_parameters, names='value')
        refresh_parameters(None)

        for box in (self.output_selection_box, self.parameter_box, self.result_box):
            box.layout.border = '1px solid black'

        super().__init__(
            top_left=self.output_selection_box,
            top_right=self.parameter_box,
            bottom_left=self.result_box,
            grid_gap="10px",
            layout=full_width_layout(),
            **kwargs,
        )


def workflow_widget(result_registry: dict | None = None) -> widgets.Widget:
    """Create a widget for a workflow selected from a dropdown."""
    workflow_select = widgets.Dropdown(
        options=[
            (workflow.title or workflow.name or workflow.factory.__name__, workflow)
            for workflow in workflow_registry
        ],
        description='Workflow:',
        value=None,
        layout=full_width_layout(),
        tooltip='Select a workflow.',
    )

    def refresh_workflow_box(change) -> None:
        workflow_box.children = [WorkflowWidget(change.new, result_registry)]

    workflow_select.observe(refresh_workflow_box, names='value')

    workflow_selection_box = widgets.HBox([workflow_select], layout=full_width_layout())
    workflow_box = widgets.Box(layout=full_width_layout())
    return widgets.VBox([workflow_selection_box, workflow_box])


def _get_parameter_box(widget: WorkflowWidget | ParameterBox) -> ParameterBox:
    if isinstance(widget, WorkflowWidget):
        return widget.parameter_box
    elif isinstance(widget, ParameterBox):
        return widget
    else:
        raise TypeError(
            f"Expected target_widget to be a WorkflowWidget or ParameterBox, "
            f"got {type(widget)}."
        )


def set_parameter_widget_values(
    widget: WorkflowWidget | ParameterBox, new_parameter_values: dict[Key, Any]
) -> None:
    """Set the values of the input widgets in the target widget."""
    _get_parameter_box(widget).set_values(new_parameter_values)


def get_parameter_widget_values(
    widget: WorkflowWidget | ParameterBox,
) -> dict[Key, dict[str, Any]]:
    """Return the current values of the input widgets in the target widget."""
    return _get_parameter_box(widget).get_fields()
