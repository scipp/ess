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
    Format,
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
        assert fields['data'].format is Format.SCIPP
        assert fields['data'].array is None
        assert fields['background'].array == ArraySpec(dims=('x',))
        assert fields['banks'].array == ArraySpec(dims=('tof',))
        assert fields['runs'].format is Format.NEXUS

    def test_ref_fields_include_literal_or_reference_unions(self) -> None:
        assert ref_fields(Params) == {'data', 'background', 'runs', 'banks', 'centre'}


class TestValidation:
    def test_data_field_accepts_either_reference_form(self) -> None:
        assert Params(data=OUTPUT_REF).data == OutputRef(record='r1', output='data')
        assert Params(data=DATASET_REF).data == DatasetRef(dataset='pid-1')

    @pytest.mark.parametrize(
        'bad',
        ['a path', Path('/data/run.nxs'), 3, [1, 2], {'x': 1}, None, sc.scalar(1.0)],
    )
    def test_data_field_rejects_anything_but_a_reference(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            Params(data=bad)

    def test_file_field_holds_references_like_any_data_field(self) -> None:
        params = Params(data=OUTPUT_REF, runs=[DATASET_REF, OUTPUT_REF])
        assert params.runs == [
            DatasetRef(dataset='pid-1'),
            OutputRef(record='r1', output='data'),
        ]
        with pytest.raises(ValidationError):
            Params(data=OUTPUT_REF, runs=['/data/run.nxs'])

    def test_dataset_identity_is_an_opaque_string(self) -> None:
        # The spec does not normalize: the framework that minted the identity is
        # the one that knows a run number and the PID minted from it are one
        # dataset.
        assert DatasetRef(dataset='pid:20.500.12269/abc') == DatasetRef(
            dataset='pid:20.500.12269/abc'
        )
        assert DatasetRef(dataset='run:dream/1') != DatasetRef(
            dataset='pid:20.500.12269/abc'
        )

    def test_literal_or_reference_union_accepts_both(self) -> None:
        params = Params(data=OUTPUT_REF, centre={'value': (0.1, 0.2), 'unit': 'm'})
        assert params.centre == Quantity(value=(0.1, 0.2), unit='m')
        params = Params(data=OUTPUT_REF, centre={'record': 'r0', 'output': 'centre'})
        assert params.centre == OutputRef(record='r0', output='centre')


class TestJsonSchema:
    def test_marks_data_fields_with_format_and_structure(self) -> None:
        schema = Params.model_json_schema()['properties']
        assert schema['data']['dataField'] == {'format': 'scipp'}
        assert schema['runs']['items']['dataField'] == {'format': 'nexus'}
        assert schema['background']['anyOf'][0]['dataField']['array'] == {
            'dims': ['x'],
            'unit': None,
            'coords': {},
            'binned': False,
        }
        assert 'dataField' not in schema['label']

    @pytest.mark.parametrize('annotation', [Array(), NexusFile, OpaqueFile])
    def test_data_field_schema_is_the_two_reference_forms(
        self, annotation: object
    ) -> None:
        class P(BaseModel):
            field: annotation

        forms = {
            c['$ref'] for c in P.model_json_schema()['properties']['field']['anyOf']
        }
        assert forms == {'#/$defs/OutputRef', '#/$defs/DatasetRef'}


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
