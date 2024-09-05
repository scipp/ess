# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib
import sys
from collections.abc import Generator

import pytest
import sciline as sl
import scipp as sc
import scippnexus as snx
from ess.nmx import default_parameters
from ess.nmx.data import small_mcstas_2_sample, small_mcstas_3_sample
from ess.nmx.mcstas.load import bank_names_to_detector_names, load_crystal_rotation
from ess.nmx.mcstas.load import providers as loader_providers
from ess.nmx.reduction import NMXData
from ess.nmx.types import (
    DetectorBankPrefix,
    DetectorIndex,
    FilePath,
    MaximumProbability,
)
from scipp.testing import assert_allclose, assert_identical

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mcstas_description_examples import (
    no_detectors,
    one_detector_no_filename,
    two_detectors_same_filename,
    two_detectors_two_filenames,
)


def check_scalar_properties_mcstas_2(dg: NMXData):
    """Test helper for NMXData loaded from McStas 2.

    Expected numbers are hard-coded based on the sample file.
    """
    assert_identical(
        dg['proton_charge'],
        sc.scalar(1e-4 * dg['weights'].bins.size().sum().data.values, unit=None),
    )
    assert_identical(dg['crystal_rotation'], sc.vector([20, 0, 90], unit='deg'))
    assert_identical(dg['sample_position'], sc.vector(value=[0, 0, 0], unit='m'))
    assert_identical(
        dg['source_position'], sc.vector(value=[-0.53123, 0.0, -157.405], unit='m')
    )
    assert dg['sample_name'] == sc.scalar("sampleMantid")


def check_nmxdata_properties(dg: NMXData, fast_axis, slow_axis) -> None:
    assert isinstance(dg, sc.DataGroup)
    assert dg.shape == ((1280, 1280)[0] * (1280, 1280)[1], 1)
    # Check maximum value of weights.
    assert_allclose(
        dg['weights'].max().data,
        sc.scalar(default_parameters[MaximumProbability], unit='counts', dtype=float),
        atol=sc.scalar(1e-10, unit='counts'),
        rtol=sc.scalar(1e-8),
    )
    assert_allclose(
        sc.squeeze(dg['fast_axis'], 'panel'), fast_axis, atol=sc.scalar(0.005)
    )
    assert_identical(sc.squeeze(dg['slow_axis'], 'panel'), slow_axis)


@pytest.mark.parametrize(
    ('detector_index', 'fast_axis', 'slow_axis'),
    [
        # Expected values are provided by the IDS
        # based on the simulation settings of the sample file.
        (0, (1.0, 0.0, -0.01), (0.0, 1.0, 0.0)),
        (1, (-0.01, 0.0, -1.0), (0.0, 1.0, 0.0)),
        (2, (0.01, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ],
)
def test_file_reader_mcstas2(
    detector_index, fast_axis, slow_axis, mcstas_2_deprecation_warning_context
) -> None:
    with mcstas_2_deprecation_warning_context():
        file_path = small_mcstas_2_sample()

    fast_axis = sc.vector(fast_axis)
    slow_axis = sc.vector(slow_axis)

    pl = sl.Pipeline(
        loader_providers,
        params={
            FilePath: file_path,
            DetectorIndex: detector_index,
            **default_parameters,
        },
    )
    dg = pl.compute(NMXData)

    check_scalar_properties_mcstas_2(dg)
    check_nmxdata_properties(dg, fast_axis, slow_axis)


def check_scalar_properties_mcstas_3(dg: NMXData):
    """Test helper for NMXData loaded from McStas 3.

    Expected numbers are hard-coded based on the sample file.
    """
    assert_identical(
        dg['proton_charge'],
        sc.scalar(1e-4 * dg['weights'].bins.size().sum().data.values, unit=None),
    )
    assert_identical(dg['crystal_rotation'], sc.vector([0, 0, 0], unit='deg'))
    assert_identical(dg['sample_position'], sc.vector(value=[0, 0, 0], unit='m'))
    assert_identical(
        dg['source_position'], sc.vector(value=[-0.53123, 0.0, -157.405], unit='m')
    )
    assert dg['sample_name'] == sc.scalar("sampleMantid")


@pytest.mark.parametrize(
    ('detector_index', 'fast_axis', 'slow_axis'),
    [
        # Expected values are provided by the IDS
        # based on the simulation settings of the sample file.
        (0, (1.0, 0.0, -0.01), (0.0, 1.0, 0.0)),
        (1, (-0.01, 0.0, -1.0), (0.0, 1.0, 0.0)),
        (2, (0.01, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ],
)
def test_file_reader_mcstas3(detector_index, fast_axis, slow_axis) -> None:
    file_path = small_mcstas_3_sample()

    pl = sl.Pipeline(
        loader_providers,
        params={
            FilePath: file_path,
            DetectorIndex: detector_index,
            **default_parameters,
        },
    )
    dg, bank = pl.compute((NMXData, DetectorBankPrefix)).values()

    entry_path = f"entry1/data/{bank}_dat_list_p_x_y_n_id_t"
    with snx.File(file_path) as file:
        raw_data = file[entry_path]["events"][()]
        data_length = raw_data.sizes['dim_0']

    check_scalar_properties_mcstas_3(dg)
    assert dg['weights'].bins.size().sum().value == data_length
    check_nmxdata_properties(dg, sc.vector(fast_axis), sc.vector(slow_axis))


@pytest.fixture(params=[small_mcstas_2_sample, small_mcstas_3_sample])
def tmp_mcstas_file(
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
    mcstas_2_deprecation_warning_context,
) -> Generator[pathlib.Path, None, None]:
    import os
    import shutil

    if request.param == small_mcstas_2_sample:
        with mcstas_2_deprecation_warning_context():
            original_file_path = request.param()
    else:
        original_file_path = request.param()

    tmp_file = tmp_path / pathlib.Path('file.h5')
    shutil.copy(original_file_path, tmp_file)
    yield tmp_file
    os.remove(tmp_file)


def test_file_reader_mcstas_additional_fields(tmp_mcstas_file: pathlib.Path) -> None:
    """Check if additional fields names do not break the loader."""
    import h5py

    entry_path = "entry1/data/bank01_events_dat_list_p_x_y_n_id_t"
    new_entry_path = entry_path + '_L'

    with h5py.File(tmp_mcstas_file, 'r+') as file:
        dataset = file[entry_path]
        del file[entry_path]
        file[new_entry_path] = dataset

    pl = sl.Pipeline(
        loader_providers,
        params={
            FilePath: str(tmp_mcstas_file),
            DetectorIndex: 0,
            **default_parameters,
        },
    )
    dg = pl.compute(NMXData)

    assert isinstance(dg, sc.DataGroup)


@pytest.fixture()
def rotation_mission_tmp_file(tmp_mcstas_file: pathlib.Path) -> pathlib.Path:
    import h5py

    param_keys = tuple(f"entry1/simulation/Param/XtalPhi{key}" for key in "XYZ")

    # Remove the rotation parameters from the file.
    with h5py.File(tmp_mcstas_file, 'a') as file:
        for key in param_keys:
            del file[key]

    return tmp_mcstas_file


def test_missing_rotation(rotation_mission_tmp_file: FilePath) -> None:
    with pytest.raises(KeyError, match="XtalPhiX"):
        load_crystal_rotation(rotation_mission_tmp_file, None)
        # McStasInstrument is not used due to error in the file.


def test_bank_names_to_detector_names_two_detectors():
    res = bank_names_to_detector_names(two_detectors_two_filenames)
    assert len(res) == 2
    assert all(len(v) == 1 for v in res.values())


def test_bank_names_to_detector_names_same_filename():
    res = bank_names_to_detector_names(two_detectors_same_filename)
    assert len(res) == 1
    assert all(len(v) == 2 for v in res.values())


def test_bank_names_to_detector_names_no_detectors():
    res = bank_names_to_detector_names(no_detectors)
    assert len(res) == 0


def test_bank_names_to_detector_names_no_filename():
    res = bank_names_to_detector_names(one_detector_no_filename)
    assert len(res) == 1
    ((bank, (detector,)),) = res.items()
    assert bank == detector
