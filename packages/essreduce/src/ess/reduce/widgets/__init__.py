# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Any, Protocol

from ._config import default_layout, default_style, full_width_layout
from ._pydantic import (
    PydanticModelWidget,
    PydanticParameterValueWidget,
    PydanticParameterWidget,
)
from ._spinner import Spinner


class EssWidget(Protocol):
    """Protocol for ESS widgets."""

    @property
    def value(self) -> Any: ...


__all__ = [
    'EssWidget',
    'PydanticModelWidget',
    'PydanticParameterValueWidget',
    'PydanticParameterWidget',
    'Spinner',
    'default_layout',
    'default_style',
    'full_width_layout',
]
