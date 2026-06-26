# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

from sciline import Pipeline
from sciline._utils import key_name
from sciline.typing import Key

T = TypeVar('T')


class KeepDefaultType:
    pass


keep_default = KeepDefaultType()


@dataclass(frozen=True)
class ParameterSpec(Generic[T]):
    """Specification for configuring a Sciline workflow key."""

    model: Any
    category: str
    title: str | None = None
    description: str | None = None
    default: T | KeepDefaultType = keep_default
    transform: Callable[[T], Any] | None = None
    apply: Callable[[Pipeline, T], Pipeline] | None = None
    filter_keys: tuple[Key, ...] = ()
    use_workflow_default: bool = True
    key: Key | None = None

    def bind(self, key: Key) -> ParameterSpec[T]:
        filter_keys = self.filter_keys or (key,)
        return replace(self, key=key, filter_keys=filter_keys)

    def with_default(
        self, default: T | KeepDefaultType = keep_default
    ) -> ParameterSpec[T]:
        if default is keep_default or not self.use_workflow_default:
            return self
        return replace(self, default=default)

    def with_apply(
        self, apply: Callable[[Pipeline, T], Pipeline] | None
    ) -> ParameterSpec[T]:
        return replace(self, apply=apply)

    @property
    def name(self) -> str:
        return self.title or (
            key_name(self.key) if self.key is not None else 'Parameter'
        )


class ParameterRegistry(MutableMapping):
    def __init__(self):
        self._parameters: dict[Key, ParameterSpec] = {}

    def __getitem__(self, key: Key) -> ParameterSpec:
        return self._parameters[key]

    def __setitem__(self, key: Key, value: ParameterSpec):
        self._parameters[key] = value.bind(key)

    def __delitem__(self, key: Key):
        del self._parameters[key]

    def __iter__(self):
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)


parameter_registry = ParameterRegistry()
