# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest

from ess.reduce.spec.parameters import (
    Quantity,
    Scale,
    WavelengthEdges,
    WavelengthRange,
    WavelengthUnit,
)


class TestQuantity:
    def test_scalar_and_vector(self) -> None:
        assert Quantity(value=1.0, unit='m').value == 1.0
        assert Quantity(value=(0.1, 0.2), unit='m').value == (0.1, 0.2)
        assert Quantity(value=2.0).unit is None

    def test_roundtrips_through_json(self) -> None:
        q = Quantity(value=(0.1, 0.2), unit='m')
        assert Quantity.model_validate_json(q.model_dump_json()) == q


class TestRangeModel:
    def test_valid_range(self) -> None:
        r = WavelengthRange(start=1.0, stop=2.0)
        assert r.unit == WavelengthUnit.ANGSTROM

    def test_stop_must_exceed_start(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WavelengthRange(start=2.0, stop=1.0)
        with pytest.raises(pydantic.ValidationError):
            WavelengthRange(start=1.0, stop=1.0)

    def test_bounds_are_required(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WavelengthRange(stop=2.0)

    def test_unit_is_constrained(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WavelengthRange(start=1.0, stop=2.0, unit='m')


class TestEdgesModel:
    def test_valid_edges(self) -> None:
        edges = WavelengthEdges(start=1.0, stop=10.0, num_bins=100)
        assert edges.scale == Scale.LINEAR

    def test_stop_must_exceed_start(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WavelengthEdges(start=10.0, stop=1.0, num_bins=100)

    def test_log_scale_requires_positive_start(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WavelengthEdges(start=0.0, stop=10.0, num_bins=100, scale=Scale.LOG)
        WavelengthEdges(start=0.1, stop=10.0, num_bins=100, scale=Scale.LOG)

    def test_num_bins_bounds(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WavelengthEdges(start=1.0, stop=10.0, num_bins=0)
        with pytest.raises(pydantic.ValidationError):
            WavelengthEdges(start=1.0, stop=10.0, num_bins=10001)


class TestJsonSchema:
    def test_unit_choices_appear_as_enum(self) -> None:
        schema = WavelengthEdges.model_json_schema()
        unit_ref = schema['properties']['unit']
        enum = schema['$defs']['WavelengthUnit']['enum']
        assert set(enum) == {'Å', 'nm'}
        assert unit_ref is not None

    def test_validated_model_roundtrips_through_json(self) -> None:
        edges = WavelengthEdges(start=1.0, stop=10.0, num_bins=100, scale=Scale.LOG)
        restored = WavelengthEdges.model_validate_json(edges.model_dump_json())
        assert restored == edges
