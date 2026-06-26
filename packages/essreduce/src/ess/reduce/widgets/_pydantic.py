# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Literal, get_args, get_origin

import ipywidgets as ipw
import pydantic
from pydantic import Field
from pydantic_core import PydanticUndefined

from ._base import WidgetWithFieldsMixin
from ._config import default_layout, default_style


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to CamelCase."""
    components = snake_str.split('_')
    return ''.join(word.capitalize() for word in components)


def get_defaults(model: type[pydantic.BaseModel]) -> dict[str, Any]:
    """Get default values for all fields in a Pydantic model."""
    defaults = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.default is not PydanticUndefined:
            defaults[field_name] = field_info.default
        elif callable(field_info.default_factory):
            defaults[field_name] = field_info.default_factory()
    return defaults


def _enum_options(enum_type: type[Enum]) -> dict[str, Enum]:
    options: dict[str, Enum] = {}
    for enum_val in enum_type:
        options[
            str(enum_val.value) if isinstance(enum_val.value, str) else enum_val.name
        ] = enum_val
    return options


def _strip_optional(field_type: Any) -> tuple[Any, bool]:
    origin = get_origin(field_type)
    if origin not in (UnionType, None) and not hasattr(field_type, '__args__'):
        return field_type, False
    args = get_args(field_type) or getattr(field_type, '__args__', ())
    if NoneType not in args:
        return field_type, False
    non_none = [arg for arg in args if arg is not NoneType]
    return (non_none[0] if len(non_none) == 1 else field_type), True


def _is_string_sequence(field_type: Any) -> bool:
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin in (list, set, frozenset) and len(args) == 1:
        return args[0] is str
    if origin is tuple and args:
        return args in ((str,), (str, ...))
    return False


def _is_set_of_enum(field_type: Any) -> bool:
    """True for ``set[E]`` / ``frozenset[E]`` where ``E`` is an ``Enum``."""
    if get_origin(field_type) not in (set, frozenset):
        return False
    args = get_args(field_type)
    if len(args) != 1:
        return False
    inner = args[0]
    return isinstance(inner, type) and issubclass(inner, Enum)


def _is_pydantic_model(field_type: Any) -> bool:
    return isinstance(field_type, type) and issubclass(field_type, pydantic.BaseModel)


def _sequence_to_text(value: Iterable[str] | str | None) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return ', '.join(value)


def _text_to_sequence(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(',') if item.strip())


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump()
    return dict(value or {})


class PydanticFieldWidget(ipw.VBox, ipw.ValueWidget):
    """Labelled wrapper for nested Pydantic model fields."""

    def __init__(self, title: str, widget: PydanticParameterWidget):
        self.widget = widget
        super().__init__([ipw.HTML(f"<strong>{title}</strong>"), widget])

    @property
    def value(self) -> pydantic.BaseModel:
        return self.widget.value

    def set_values(self, values: dict[str, Any] | pydantic.BaseModel) -> None:
        self.widget.set_values(values)


class PydanticParameterValueWidget(ipw.VBox, ipw.ValueWidget):
    """Widget for one workflow parameter value."""

    def __init__(
        self,
        model: Any,
        *,
        title: str,
        description: str | None = None,
        default: Any = PydanticUndefined,
    ):
        self._returns_model = _is_pydantic_model(model)
        if self._returns_model:
            initial_values = default if default is not PydanticUndefined else None
            self._widget = PydanticParameterWidget(model, initial_values=initial_values)
        else:
            if default is PydanticUndefined:
                field = Field(title=title, description=description)
            else:
                field = Field(default=default, title=title, description=description)
            model_class = pydantic.create_model('ParameterValue', value=(model, field))
            self._widget = PydanticParameterWidget(model_class)
        super().__init__([self._widget])

    @property
    def value(self) -> Any:
        value = self._widget.create_model()
        return value if self._returns_model else value.value

    def set_value(self, value: Any) -> None:
        if self._returns_model:
            self._widget.set_values(value)
        elif isinstance(value, dict):
            self._widget.set_values(value)
        else:
            self._widget.set_values({'value': value})

    def get_fields(self) -> dict[str, Any]:
        if self._returns_model:
            value = self.value
            return (
                value.model_dump() if isinstance(value, pydantic.BaseModel) else value
            )
        return {'value': self.value}

    def set_fields(self, values: dict[str, Any]) -> None:
        self.set_value(values)

    def validate(self) -> tuple[bool, str]:
        return self._widget.validate()

    def set_error_state(self, has_error: bool, error_message: str) -> None:
        self._widget.set_error_state(has_error, error_message)


class PydanticParameterWidget(ipw.VBox, ipw.ValueWidget, WidgetWithFieldsMixin):
    """Widget for creating and validating Pydantic model instances."""

    def __init__(
        self,
        model_class: type[pydantic.BaseModel],
        initial_values: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.widgets: dict[str, ipw.Widget] = {}
        self.fields = self.widgets
        self._error = ipw.HTML()
        self._create_widgets()
        if initial_values:
            self.set_values(initial_values)
        self.children = [*self.widgets.values(), self._error]

    def _create_widgets(self) -> None:
        for field_name, field_info in self.model_class.model_fields.items():
            self.widgets[field_name] = self._create_widget_for_field(
                field_name, field_info
            )

    def _create_widget_for_field(
        self, field_name: str, field_info: pydantic.fields.FieldInfo
    ) -> ipw.Widget:
        field_type, optional = _strip_optional(field_info.annotation)
        default_value = self._field_default(field_info)
        description = field_info.description or field_name
        display_name = field_info.title or snake_to_camel(field_name)
        disabled = bool(field_info.frozen)
        common = {
            'description': display_name,
            'disabled': disabled,
            'tooltip': description,
            'layout': default_layout,
            'style': default_style,
        }

        if field_type is float:
            return ipw.FloatText(value=default_value or 0.0, **common)
        if field_type is int:
            return ipw.IntText(value=default_value or 0, **common)
        if field_type is bool:
            return ipw.Checkbox(value=default_value or False, **common)
        if field_type in (Path, str):
            value = '' if default_value is None else str(default_value)
            return ipw.Text(value=value, placeholder=description, **common)
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            options = _enum_options(field_type)
            value = (
                default_value
                if default_value is not None
                else next(iter(options.values()))
            )
            return ipw.Dropdown(options=options, value=value, **common)
        if _is_set_of_enum(field_type):
            (enum_type,) = get_args(field_type)
            options = _enum_options(enum_type)
            value = tuple(default_value or ())
            return ipw.SelectMultiple(
                options=options,
                value=value,
                rows=min(len(options), 10),
                **common,
            )
        if get_origin(field_type) is Literal:
            literal_values = get_args(field_type)
            options = {str(val): val for val in literal_values}
            value = default_value if default_value is not None else literal_values[0]
            return ipw.Dropdown(options=options, value=value, **common)
        if _is_string_sequence(field_type):
            value = _sequence_to_text(default_value)
            return ipw.Text(value=value, placeholder=description, **common)
        if _is_pydantic_model(field_type):
            return PydanticFieldWidget(
                display_name,
                PydanticParameterWidget(
                    field_type, initial_values=_as_dict(default_value)
                ),
            )

        value = '' if default_value is None and optional else str(default_value or '')
        return ipw.Text(value=value, placeholder=description, **common)

    @staticmethod
    def _field_default(field_info: pydantic.fields.FieldInfo) -> Any:
        if field_info.default is not PydanticUndefined:
            return field_info.default
        if callable(field_info.default_factory):
            return field_info.default_factory()
        return None

    def get_values(self) -> dict[str, Any]:
        values = {}
        for field_name, widget in self.widgets.items():
            field_type, optional = _strip_optional(
                self.model_class.model_fields[field_name].annotation
            )
            value = widget.value
            if field_type is Path:
                value = Path(value) if value else None
            elif _is_string_sequence(field_type):
                value = _text_to_sequence(value)
            elif _is_set_of_enum(field_type):
                pass
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                pass
            elif optional and value == '':
                value = None
            values[field_name] = value
        return values

    def set_values(self, values: dict[str, Any] | pydantic.BaseModel) -> None:
        if isinstance(values, pydantic.BaseModel):
            values = values.model_dump()
        for field_name, value in values.items():
            if field_name not in self.widgets:
                continue
            field_type, _ = _strip_optional(
                self.model_class.model_fields[field_name].annotation
            )
            if value is None and field_type in (Path, str):
                value = ''
            elif isinstance(value, Path):
                value = str(value)
            elif _is_string_sequence(field_type):
                value = _sequence_to_text(value)
            elif _is_set_of_enum(field_type):
                (enum_type,) = get_args(field_type)
                value = tuple(enum_type(v) for v in value)
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                value = field_type(value)
            elif _is_pydantic_model(field_type):
                self.widgets[field_name].set_values(value)
                continue
            elif isinstance(value, pydantic.BaseModel):
                value = value.model_dump()
            self.widgets[field_name].value = value

    def create_model(self) -> pydantic.BaseModel:
        return self.model_class(**self.get_values())

    @property
    def value(self) -> pydantic.BaseModel:
        return self.create_model()

    def validate(self) -> tuple[bool, str]:
        try:
            self.create_model()
            return True, "Valid"
        except pydantic.ValidationError as e:
            error_details = e.errors()
            if error_details:
                field_name = error_details[0].get('loc', [''])[0]
                error_msg = error_details[0].get('msg', str(e))
                return False, f"{field_name}: {error_msg}" if field_name else error_msg
            return False, str(e)
        except ValueError as e:
            return False, str(e)

    def set_error_state(self, has_error: bool, error_message: str) -> None:
        for widget in self.widgets.values():
            widget.layout.border = ''
        if not has_error:
            self._error.value = ''
            return
        try:
            self.create_model()
        except pydantic.ValidationError as e:
            error_details = e.errors()
            if error_details:
                field_name = error_details[0].get('loc', [''])[0]
                if field_name in self.widgets:
                    self.widgets[field_name].layout.border = '2px solid #dc3545'
        self._error.value = (
            "<p style='color: #dc3545; margin: 5px 0; font-size: 0.9em;'>"
            f"{error_message}</p>"
        )


class PydanticModelWidget(ipw.VBox):
    """Build ipywidgets for flat or grouped Pydantic parameter models."""

    def __init__(
        self,
        model_class: type[pydantic.BaseModel],
        initial_values: dict[str, Any] | pydantic.BaseModel | None = None,
        hidden_fields: frozenset[str] = frozenset(),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_class = model_class
        self._hidden_fields = hidden_fields
        self._parameter_widgets: dict[str, PydanticParameterWidget] = {}
        self._failing_field_names: list[str] = []
        if isinstance(initial_values, pydantic.BaseModel):
            initial_values = initial_values.model_dump()
        self._initial_values = initial_values or {}
        self._build()

    def _build(self) -> None:
        if self._is_grouped_model():
            self._build_grouped()
        else:
            widget = PydanticParameterWidget(
                self._model_class, initial_values=self._initial_values
            )
            self._parameter_widgets[''] = widget
            self.children = [widget]

    def _is_grouped_model(self) -> bool:
        visible_fields = [
            field_info.annotation
            for name, field_info in self._model_class.model_fields.items()
            if not name.startswith('_') and name not in self._hidden_fields
        ]
        return bool(visible_fields) and all(
            _is_pydantic_model(field_type) for field_type in visible_fields
        )

    def _build_grouped(self) -> None:
        children = []
        titles = []
        root_defaults = get_defaults(self._model_class)
        for field_name, field_info in self._model_class.model_fields.items():
            if field_name.startswith('_') or field_name in self._hidden_fields:
                continue
            field_type = field_info.annotation
            values = get_defaults(field_type)
            values.update(self._as_dict(root_defaults.get(field_name, {})))
            values.update(self._as_dict(self._initial_values.get(field_name, {})))
            widget = PydanticParameterWidget(field_type, initial_values=values)
            self._parameter_widgets[field_name] = widget
            title = field_info.title or field_name.replace('_', ' ').title()
            titles.append(title)
            children.append(widget)
        accordion = ipw.Accordion(children=children)
        for index, title in enumerate(titles):
            accordion.set_title(index, title)
        if children:
            accordion.selected_index = 0
        self.children = [accordion]

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        return _as_dict(value)

    @property
    def parameter_values(self) -> pydantic.BaseModel:
        if '' in self._parameter_widgets:
            return self._parameter_widgets[''].create_model()
        widget_values = {
            name: widget.create_model()
            for name, widget in self._parameter_widgets.items()
        }
        return self._model_class(**widget_values)

    @property
    def value(self) -> pydantic.BaseModel:
        return self.parameter_values

    def validate_parameters(self) -> tuple[bool, list[str]]:
        errors = []
        self._failing_field_names = []
        for field_name, widget in self._parameter_widgets.items():
            is_valid, error_msg = widget.validate()
            if not is_valid:
                errors.append(f"{field_name}: {error_msg}" if field_name else error_msg)
                self._failing_field_names.append(field_name)
                widget.set_error_state(True, error_msg)
            else:
                widget.set_error_state(False, "")
        return len(errors) == 0, errors

    def get_failing_field_names(self) -> list[str]:
        return list(self._failing_field_names)

    def clear_validation_errors(self) -> None:
        self._failing_field_names = []
        for widget in self._parameter_widgets.values():
            widget.set_error_state(False, "")

    def set_values(self, values: dict[str, Any] | pydantic.BaseModel) -> None:
        if isinstance(values, pydantic.BaseModel):
            values = values.model_dump()
        if '' in self._parameter_widgets:
            self._parameter_widgets[''].set_values(values)
            return
        for field_name, field_values in values.items():
            if field_name in self._parameter_widgets:
                self._parameter_widgets[field_name].set_values(field_values)

    def get_values(self) -> dict[str, Any]:
        if '' in self._parameter_widgets:
            return self._parameter_widgets[''].get_values()
        return {
            field_name: widget.get_values()
            for field_name, widget in self._parameter_widgets.items()
        }

    def get_parameter_widget(self, field_name: str) -> PydanticParameterWidget | None:
        return self._parameter_widgets.get(field_name)
