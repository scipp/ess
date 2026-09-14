# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest
from pydantic import Field

from ess.reduce.spec import (
    Array,
    ArraySpec,
    NexusFile,
    NoParams,
    Quantity,
    SerializedWorkflowSpec,
    WorkflowSpec,
)


class Params(pydantic.BaseModel):
    sample: NexusFile
    lower: float
    upper: float

    @pydantic.model_validator(mode='after')
    def upper_greater_than_lower(self) -> 'Params':
        if self.upper <= self.lower:
            raise ValueError('upper must be greater than lower')
        return self


IOFQ = ArraySpec(dims=('Q',), unit='counts', coords={'Q': '1/Å'})


class Outputs(pydantic.BaseModel):
    iofq: Array(IOFQ) = Field(title='I(Q)', description='Scattering intensity.')
    beam_centre: Quantity = Field(title='Beam centre')
    transmission: Array() | None = Field(default=None, title='Transmission')


class Result(pydantic.BaseModel):
    result: Array()


@pytest.fixture
def spec() -> WorkflowSpec:
    return WorkflowSpec(
        name='my-workflow',
        version=1,
        title='My workflow',
        description='Computes things.',
        params=Params,
        outputs=Outputs,
    )


class TestWorkflowSpec:
    def test_minimal_spec_defaults_to_no_params(self) -> None:
        spec = WorkflowSpec(
            name='wf', version=1, title='Workflow', description='D', outputs=Result
        )
        assert spec.params is NoParams
        assert spec.code_revision is None

    def test_outputs_are_required(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WorkflowSpec(name='wf', version=1, title='Workflow', description='D')

    @pytest.mark.parametrize('field', ['name', 'title', 'description'])
    def test_empty_metadata_field_rejected(self, field: str) -> None:
        fields = {
            'name': 'wf',
            'version': 1,
            'title': 'Workflow',
            'description': 'Does things.',
            'outputs': Result,
        }
        with pytest.raises(pydantic.ValidationError):
            WorkflowSpec(**{**fields, field: ''})

    def test_version_must_be_positive(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            WorkflowSpec(
                name='wf', version=0, title='W', description='D', outputs=Result
            )

    def test_spec_is_frozen(self, spec: WorkflowSpec) -> None:
        with pytest.raises(pydantic.ValidationError):
            spec.title = 'Other'

    def test_no_params_rejects_any_input(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            NoParams(anything=1)

    def test_params_model_validates_in_process(self, spec: WorkflowSpec) -> None:
        with pytest.raises(pydantic.ValidationError):
            spec.params(sample={'dataset': 'pid'}, lower=2.0, upper=1.0)

    def test_output_metadata_is_field_metadata(self, spec: WorkflowSpec) -> None:
        fields = spec.outputs.model_fields
        assert list(fields) == ['iofq', 'beam_centre', 'transmission']
        assert fields['iofq'].title == 'I(Q)'
        assert fields['iofq'].description == 'Scattering intensity.'
        assert fields['transmission'].is_required() is False


class TestSerialization:
    def test_serialize_projects_models_to_json_schema(self, spec: WorkflowSpec) -> None:
        serialized = spec.serialize()
        assert serialized.params_schema == Params.model_json_schema()
        assert serialized.outputs_schema == Outputs.model_json_schema()

    def test_serialize_preserves_metadata(self) -> None:
        spec = WorkflowSpec(
            name='wf',
            version=2,
            title='W',
            description='D',
            code_revision='abc123',
            outputs=Result,
        )
        serialized = spec.serialize()
        assert (serialized.name, serialized.version) == ('wf', 2)
        assert (serialized.title, serialized.description) == ('W', 'D')
        assert serialized.code_revision == 'abc123'

    def test_output_order_and_metadata_preserved(self, spec: WorkflowSpec) -> None:
        properties = spec.serialize().outputs_schema['properties']
        assert list(properties) == ['iofq', 'beam_centre', 'transmission']
        assert properties['iofq']['title'] == 'I(Q)'
        assert properties['iofq']['description'] == 'Scattering intensity.'

    def test_serialized_spec_roundtrips_through_json(self, spec: WorkflowSpec) -> None:
        serialized = spec.serialize()
        restored = SerializedWorkflowSpec.model_validate_json(
            serialized.model_dump_json()
        )
        assert restored == serialized

    def test_array_structure_survives_json_roundtrip(self, spec: WorkflowSpec) -> None:
        restored = SerializedWorkflowSpec.model_validate_json(
            spec.serialize().model_dump_json()
        )
        iofq = restored.outputs_schema['properties']['iofq']
        assert ArraySpec.model_validate(iofq['dataField']['array']) == IOFQ
