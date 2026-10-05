# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import pytest
import scipp as sc
from pydantic import BaseModel, ValidationError

from ess.reduce.spec import (
    AccumulatorRef,
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
    table_fields,
    walk_refs,
)


class Params(BaseModel):
    data: Array()
    background: Array(ArraySpec(dims=('x',))) | None = None
    runs: list[NexusFile] = []
    banks: dict[str, Array(ArraySpec(dims=('tof',)))] = {}
    centre: Quantity | OutputRef | None = None
    label: str = ''


class Parts(BaseModel):
    numerator: Array(ArraySpec(dims=('Q',), unit='counts'))
    denominator: Array(ArraySpec(dims=('Q',), unit='counts'))
    weight: float = 1.0


class SampleRun(BaseModel):
    sample: NexusFile
    transmission: list[NexusFile] = []
    label: str = ''


class TableParams(BaseModel):
    parts: list[Parts]
    runs: list[SampleRun] | None = None
    labels: list[str] = []


class Settings(BaseModel):
    weight: float = 1.0


OUTPUT_REF = {'record': 'r1', 'output': 'data'}
DATASET_REF = {'dataset': 'pid-1'}
ACCUMULATOR_REF = {'accumulator': 'a1', 'output': 'data'}


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


class TestTableFields:
    def test_finds_lists_of_a_model_with_their_row_model(self) -> None:
        assert table_fields(TableParams) == {'parts': Parts, 'runs': SampleRun}

    def test_lists_of_data_fields_or_literals_are_not_tables(self) -> None:
        assert table_fields(Params) == {}

    def test_a_table_is_not_a_data_field(self) -> None:
        assert data_fields(TableParams) == {}
        cells = data_fields(table_fields(TableParams)['runs'])
        assert set(cells) == {'sample', 'transmission'}
        assert cells['sample'].format is Format.NEXUS

    def test_a_table_with_reference_cells_is_a_ref_field(self) -> None:
        class Weights(BaseModel):
            rows: list[Settings]

        assert ref_fields(TableParams) == {'parts', 'runs'}
        assert ref_fields(Weights) == set()

    @pytest.mark.parametrize(
        'cell', [Settings, Settings | None, list[Settings], dict[str, Settings]]
    )
    def test_refuses_a_row_holding_a_model_or_a_table(self, cell: object) -> None:
        class Row(BaseModel):
            data: Array()
            nested: cell

        class Nested(BaseModel):
            rows: list[Row]

        with pytest.raises(ValueError, match=r'Nested\.rows: .* Row\.nested holds'):
            table_fields(Nested)

    def test_rows_validate_their_cells(self) -> None:
        params = TableParams(
            parts=[{'numerator': OUTPUT_REF, 'denominator': DATASET_REF}]
        )
        assert params.parts == [
            Parts(
                numerator=OutputRef(record='r1', output='data'),
                denominator=DatasetRef(dataset='pid-1'),
            )
        ]
        with pytest.raises(ValidationError):
            TableParams(parts=[{'numerator': OUTPUT_REF, 'denominator': 'a path'}])


class TestValidation:
    def test_data_field_accepts_every_reference_form(self) -> None:
        assert Params(data=OUTPUT_REF).data == OutputRef(record='r1', output='data')
        assert Params(data=DATASET_REF).data == DatasetRef(dataset='pid-1')
        assert Params(data=ACCUMULATOR_REF).data == AccumulatorRef(
            accumulator='a1', output='data'
        )

    def test_an_accumulator_reference_is_bound_to_a_count_of_pushes(self) -> None:
        bound = Params(data={**ACCUMULATOR_REF, 'upto': 3}).data
        assert bound == AccumulatorRef(accumulator='a1', output='data', upto=3)
        assert str(bound) == 'a1[:3].data'
        assert str(AccumulatorRef(accumulator='a1', output='data')) == 'a1.data'
        with pytest.raises(ValidationError):
            Params(data={**ACCUMULATOR_REF, 'upto': -1})

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
    def test_data_field_schema_is_the_three_reference_forms(
        self, annotation: object
    ) -> None:
        class P(BaseModel):
            field: annotation

        forms = {
            c['$ref'] for c in P.model_json_schema()['properties']['field']['anyOf']
        }
        assert forms == {
            '#/$defs/OutputRef',
            '#/$defs/DatasetRef',
            '#/$defs/AccumulatorRef',
        }


class TestReferences:
    def test_walk_refs_finds_references_at_any_depth(self) -> None:
        params = {
            'data': OUTPUT_REF,
            'runs': [DATASET_REF, {'record': 'f2', 'output': 'file'}],
            'banks': {'a': {'record': 'r2', 'output': 'banks', 'key': 'a'}},
            'centre': {'value': 1.0, 'unit': 'm'},
            'sum': {**ACCUMULATOR_REF, 'upto': 2},
        }
        assert [(p, str(r)) for p, r in walk_refs(params)] == [
            ('data', 'r1.data'),
            ('runs[0]', 'pid-1'),
            ('runs[1]', 'f2.file'),
            ('banks.a', 'r2.banks[a]'),
            ('sum', 'a1[:2].data'),
        ]

    def test_walk_refs_finds_references_in_table_rows(self) -> None:
        params = {
            'parts': [
                {'numerator': OUTPUT_REF, 'denominator': DATASET_REF, 'weight': 2.0},
                {
                    'numerator': {'record': 'r2', 'output': 'num'},
                    'denominator': DATASET_REF,
                },
            ]
        }
        assert [(p, str(r)) for p, r in walk_refs(params)] == [
            ('parts[0].numerator', 'r1.data'),
            ('parts[0].denominator', 'pid-1'),
            ('parts[1].numerator', 'r2.num'),
            ('parts[1].denominator', 'pid-1'),
        ]

    def test_as_ref_decides_what_a_reference_is(self) -> None:
        assert as_ref(OutputRef(record='r', output='o')) == OutputRef(
            record='r', output='o'
        )
        assert as_ref(OUTPUT_REF) == OutputRef(record='r1', output='data')
        assert as_ref(DATASET_REF) == DatasetRef(dataset='pid-1')
        assert as_ref(ACCUMULATOR_REF) == AccumulatorRef(
            accumulator='a1', output='data'
        )
        assert as_ref({'record': 'r1', 'output': 'o', 'extra': 1}) is None
        assert as_ref({'dataset': 'pid', 'extra': 1}) is None
        assert as_ref({**ACCUMULATOR_REF, 'record': 'r1'}) is None
        assert as_ref({'value': 1.0}) is None
        assert as_ref('r1.data') is None
