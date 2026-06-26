# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from typing import Any

import ipywidgets as widgets
import sciline as sl
from IPython import display
from ipywidgets import Layout
from pydantic_core import PydanticUndefined
from sciline.typing import Key

from .parameter import ParameterSpec, keep_default
from .widgets import (
    PydanticParameterValueWidget,
    Spinner,
    default_layout,
)
from .workflow import (
    RegisteredWorkflow,
    WorkflowFactory,
    assign_parameter_values,
    create_workflow,
    get_parameters,
    get_possible_outputs,
    get_typical_outputs,
    workflow_registry,
)


def _wrap_foldable(
    wrapped: widgets.Widget, title: str | None = None
) -> widgets.Accordion:
    return widgets.Accordion(
        [wrapped],
        layout=Layout(width='99%', height='auto'),
        titles=(title,),
    )


class OutputSelectionWidget(widgets.VBox):
    def __init__(self, workflow: sl.Pipeline, **kwargs):
        self.typical_outputs_widget = widgets.SelectMultiple(
            options=get_typical_outputs(workflow),
            layout=Layout(width='90%', height='250px'),
        )
        self.possible_outputs_widget = widgets.SelectMultiple(
            options=get_possible_outputs(workflow),
            layout=Layout(width='90%', height='auto'),
        )
        _typical_selection = _wrap_foldable(
            self.typical_outputs_widget, title='Typical Outputs'
        )
        _typical_selection.selected_index = 0
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
        self.parameter_refresh_button = widgets.Button(
            description='Refresh Parameters',
            disabled=False,
            button_style='success',
            tooltip='Generate Parameter Input Widgets',
        )
        self._registry_getter = registry_getter
        self._input_registry: dict[Key, ParameterSpec] = {}
        self._input_widgets: dict[Key, PydanticParameterValueWidget] = {}
        self._input_box = widgets.Box()
        self.parameter_refresh_button.on_click(self._refresh_input_box)
        super().__init__([self.parameter_refresh_button, self._input_box], **kwargs)

    def _refresh_input_box(self, _: widgets.Button | None = None) -> None:
        existing_values = self.get_values()
        new_input_parameters = self._registry_getter()
        self._input_registry.clear()
        self._input_registry.update(new_input_parameters)
        self._input_widgets.clear()

        grouped: dict[str, list[widgets.Widget]] = {}
        for key, spec in new_input_parameters.items():
            widget = _create_parameter_widget(spec)
            if key in existing_values:
                widget.set_value(existing_values[key])
            self._input_widgets[key] = widget
            grouped.setdefault(spec.category, []).append(widget)

        if not grouped:
            self._input_box.children = [widgets.HTML("<em>No parameters</em>")]
            return

        categories = list(grouped)
        accordion = widgets.Accordion(
            [widgets.VBox(grouped[category]) for category in categories]
        )
        for index, category in enumerate(categories):
            accordion.set_title(index, category)
        accordion.selected_index = 0
        self._input_box.children = [accordion]

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


def connect_refresh_button(
    refresh_button: widgets.Button, output_widget: widgets.Output
) -> None:
    def refresh_output(_: widgets.Button):
        output_widget.clear_output()

    refresh_button.on_click(refresh_output)


def connect_output_selection_and_parameter_run_button(
    *output_selection_widgets: widgets.Widget,
    parameter_refresh_button: widgets.Button,
    run_button: widgets.Button,
) -> None:
    def observe_selection_change(_) -> None:
        run_button.disabled = True
        run_button.tooltip = 'To run the workflow, refresh parameters.'

    for output_selection_widget in output_selection_widgets:
        output_selection_widget.observe(observe_selection_change)

    original_run_button_tooltip = run_button.tooltip

    def observe_parameter_refreshed(_) -> None:
        run_button.disabled = False
        run_button.tooltip = original_run_button_tooltip

    parameter_refresh_button.on_click(observe_parameter_refreshed)


class WorkflowWidget(widgets.TwoByTwoLayout):
    def __init__(
        self,
        workflow: sl.Pipeline | WorkflowFactory | RegisteredWorkflow,
        result_registry: dict | None = None,
        **kwargs,
    ):
        self.workflow = workflow
        self._pipeline = self._create_pipeline()
        self.output_selection_box = OutputSelectionWidget(self._pipeline)

        def registry_getter() -> dict[Key, ParameterSpec]:
            return get_parameters(
                self._pipeline, tuple(self.output_selection_box.value)
            )

        self.parameter_box = ParameterBox(registry_getter)

        def workflow_runner() -> dict[type, Any]:
            is_valid, errors = self.parameter_box.validate()
            if not is_valid:
                raise ValueError('\n'.join(errors))
            pipeline = assign_parameter_values(
                self._pipeline, self.parameter_box.value, self.parameter_box.parameters
            )
            return pipeline.compute(self.output_selection_box.value)

        self.result_box = ResultBox(workflow_runner, result_registry)
        connect_refresh_button(
            self.parameter_box.parameter_refresh_button, self.result_box.output
        )
        connect_output_selection_and_parameter_run_button(
            self.output_selection_box.typical_outputs_widget,
            self.output_selection_box.possible_outputs_widget,
            parameter_refresh_button=self.parameter_box.parameter_refresh_button,
            run_button=self.result_box.run_button,
        )
        for box in (self.output_selection_box, self.parameter_box, self.result_box):
            box.layout.border = '1px solid black'

        super().__init__(
            top_left=self.output_selection_box,
            top_right=self.parameter_box,
            bottom_left=self.result_box,
            grid_gap="10px",
            layout=default_layout,
            **kwargs,
        )

    def _create_pipeline(self) -> sl.Pipeline:
        if isinstance(self.workflow, sl.Pipeline):
            return self.workflow.copy()
        return create_workflow(self.workflow)


def workflow_widget(result_registry: dict | None = None) -> widgets.Widget:
    """Create a widget for a workflow selected from a dropdown."""
    workflow_select = widgets.Dropdown(
        options=[
            (workflow.spec.title or workflow.spec.name, workflow)
            for workflow in workflow_registry
        ],
        description='Workflow:',
        value=None,
        layout=default_layout,
        tooltip='Select a workflow.',
    )

    def refresh_workflow_box(change) -> None:
        workflow_box.children = [WorkflowWidget(change.new, result_registry)]

    workflow_select.observe(refresh_workflow_box, names='value')

    workflow_selection_box = widgets.HBox([workflow_select], layout=default_layout)
    workflow_box = widgets.Box(layout=default_layout)
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
