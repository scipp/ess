# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest

from ess.reduce.spec import (
    ArraySpec,
    NoParams,
    OutputSpec,
    SerializedWorkflowSpec,
    WorkflowSpec,
)


class Params(pydantic.BaseModel):
    lower: float
    upper: float

    @pydantic.model_validator(mode='after')
    def upper_greater_than_lower(self) -> 'Params':
        if self.upper <= self.lower:
            raise ValueError('upper must be greater than lower')
        return self


@pytest.fixture
def spec() -> WorkflowSpec:
    return WorkflowSpec(
        name='my-workflow',
        version=1,
        title='My workflow',
        description='Computes things.',
        params=Params,
        outputs={
            'iofq': OutputSpec(
                title='I(Q)',
                array=ArraySpec(dims=('Q',), unit='counts', coords={'Q': '1/Å'}),
            ),
            'transmission': OutputSpec(title='Transmission'),
        },
    )


class TestWorkflowSpec:
    def test_minimal_spec_defaults_to_no_params_and_result_output(self) -> None:
        spec = WorkflowSpec(
            name='wf', version=1, title='Workflow', description='Does things.'
        )
        assert spec.params is NoParams
        assert list(spec.outputs) == ['result']

    @pytest.mark.parametrize('field', ['name', 'title', 'description'])
    def test_empty_metadata_field_rejected(self, field: str) -> None:
        fields = {
            'name': 'wf',
            'version': 1,
            'title': 'Workflow',
            'description': 'Does things.',
        }
        with pytest.raises(pydantic.ValidationError):
            WorkflowSpec(**{**fields, field: ''})

    def test_version_must_be_positive(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WorkflowSpec(name='wf', version=0, title='W', description='D')

    def test_spec_is_frozen(self, spec: WorkflowSpec) -> None:
        with pytest.raises(pydantic.ValidationError):
            spec.title = 'Other'

    def test_no_params_rejects_any_input(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            NoParams(anything=1)

    def test_params_model_validates_in_process(self, spec: WorkflowSpec) -> None:
        with pytest.raises(pydantic.ValidationError):
            spec.params(lower=2.0, upper=1.0)


class TestSerialization:
    def test_serialize_projects_params_to_json_schema(self, spec: WorkflowSpec) -> None:
        serialized = spec.serialize()
        assert serialized.params_schema == Params.model_json_schema()
        assert set(serialized.params_schema['properties']) == {'lower', 'upper'}

    def test_serialize_preserves_metadata_and_outputs(self, spec: WorkflowSpec) -> None:
        serialized = spec.serialize()
        assert serialized.name == spec.name
        assert serialized.version == spec.version
        assert serialized.title == spec.title
        assert serialized.description == spec.description
        assert serialized.outputs == spec.outputs

    def test_output_order_preserved(self, spec: WorkflowSpec) -> None:
        assert list(spec.serialize().outputs) == ['iofq', 'transmission']

    def test_serialized_spec_roundtrips_through_json(self, spec: WorkflowSpec) -> None:
        serialized = spec.serialize()
        restored = SerializedWorkflowSpec.model_validate_json(
            serialized.model_dump_json()
        )
        assert restored == serialized

    def test_array_spec_survives_json_roundtrip(self, spec: WorkflowSpec) -> None:
        restored = SerializedWorkflowSpec.model_validate_json(
            spec.serialize().model_dump_json()
        )
        array = restored.outputs['iofq'].array
        assert array == ArraySpec(dims=('Q',), unit='counts', coords={'Q': '1/Å'})
