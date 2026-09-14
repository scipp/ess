# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import pytest
import scipp as sc
from pydantic import BaseModel, ValidationError

from ess.reduce.spec import (
    Array,
    ArraySpec,
    DatasetRef,
    Kind,
    NexusFile,
    OpaqueFile,
    OutputRef,
    Quantity,
    as_ref,
    data_fields,
    ref_fields,
    walk_refs,
)


class Params(BaseModel):
    data: Array()
    background: Array(ArraySpec(dims=('x',))) | None = None
    runs: list[NexusFile] = []
    banks: dict[str, Array(ArraySpec(dims=('tof',)))] = {}
    centre: Quantity | OutputRef | None = None
    label: str = ''


OUTPUT_REF = {'record': 'r1', 'output': 'data'}
DATASET_REF = {'dataset': 'pid-1'}


class TestFieldIntrospection:
    def test_data_fields_include_optionals_and_collections(self) -> None:
        fields = data_fields(Params)
        assert set(fields) == {'data', 'background', 'runs', 'banks'}
        assert fields['data'].kind is Kind.ARRAY
        assert fields['data'].array is None
        assert fields['background'].array == ArraySpec(dims=('x',))
        assert fields['banks'].array == ArraySpec(dims=('tof',))
        assert fields['runs'].kind is Kind.NEXUS

    def test_ref_fields_include_literal_or_reference_unions(self) -> None:
        assert ref_fields(Params) == {'data', 'background', 'runs', 'banks', 'centre'}


class TestValidation:
    def test_array_field_accepts_either_reference_form(self) -> None:
        assert Params(data=OUTPUT_REF).data == OutputRef(record='r1', output='data')
        assert Params(data=DATASET_REF).data == DatasetRef(dataset='pid-1')

    def test_array_field_accepts_a_scipp_object(self) -> None:
        assert Params(data=sc.scalar(1.0)).data.value == 1.0

    @pytest.mark.parametrize('bad', ['a path', 3, [1, 2], {'x': 1}, None])
    def test_array_field_rejects_plain_data(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            Params(data=bad)

    def test_file_field_accepts_a_reference_or_a_path(self) -> None:
        assert Params(data=OUTPUT_REF, runs=[DATASET_REF]).runs == [
            DatasetRef(dataset='pid-1')
        ]
        assert Params(data=OUTPUT_REF, runs=['/data/run.nxs']).runs == [
            Path('/data/run.nxs')
        ]

    def test_literal_or_reference_union_accepts_both(self) -> None:
        params = Params(data=OUTPUT_REF, centre={'value': (0.1, 0.2), 'unit': 'm'})
        assert params.centre == Quantity(value=(0.1, 0.2), unit='m')
        params = Params(data=OUTPUT_REF, centre={'record': 'r0', 'output': 'centre'})
        assert params.centre == OutputRef(record='r0', output='centre')


class TestJsonSchema:
    def test_marks_data_fields_with_kind_and_structure(self) -> None:
        schema = Params.model_json_schema()['properties']
        assert schema['data']['dataField'] == {'kind': 'array'}
        assert schema['runs']['items']['dataField'] == {'kind': 'nexus'}
        assert schema['background']['anyOf'][0]['dataField']['array'] == {
            'dims': ['x'],
            'unit': None,
            'coords': {},
            'binned': False,
        }
        assert 'dataField' not in schema['label']

    def test_array_field_schema_shows_reference_forms_only(self) -> None:
        schema = Params.model_json_schema()
        forms = {c['$ref'] for c in schema['properties']['data']['anyOf']}
        assert forms == {'#/$defs/OutputRef', '#/$defs/DatasetRef'}

    def test_file_field_schema_shows_path_form_too(self) -> None:
        class P(BaseModel):
            run: OpaqueFile

        forms = P.model_json_schema()['properties']['run']['anyOf']
        assert {'type': 'string', 'format': 'path'} in forms


class TestReferences:
    def test_walk_refs_finds_references_at_any_depth(self) -> None:
        params = {
            'data': OUTPUT_REF,
            'runs': [DATASET_REF, {'record': 'f2', 'output': 'file'}],
            'banks': {'a': {'record': 'r2', 'output': 'banks', 'key': 'a'}},
            'centre': {'value': 1.0, 'unit': 'm'},
        }
        assert [(p, str(r)) for p, r in walk_refs(params)] == [
            ('data', 'r1.data'),
            ('runs[0]', 'pid-1'),
            ('runs[1]', 'f2.file'),
            ('banks.a', 'r2.banks[a]'),
        ]

    def test_as_ref_decides_what_a_reference_is(self) -> None:
        assert as_ref(OutputRef(record='r', output='o')) == OutputRef(
            record='r', output='o'
        )
        assert as_ref(OUTPUT_REF) == OutputRef(record='r1', output='data')
        assert as_ref(DATASET_REF) == DatasetRef(dataset='pid-1')
        assert as_ref({'record': 'r1', 'output': 'o', 'extra': 1}) is None
        assert as_ref({'dataset': 'pid', 'extra': 1}) is None
        assert as_ref({'value': 1.0}) is None
        assert as_ref('r1.data') is None
