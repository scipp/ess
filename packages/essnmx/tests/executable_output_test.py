# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Compare the workflow output to be consistent with the reference output file.

This test detects any changes in the output file.
If anything must change, consult the IDS member
and update the frozen file in the `ess.nmx.data` registry.
"""

import pathlib

# The bitshuffle plugin needs to be imported at least once in the session
# so that h5py can use the plugin.
import bitshuffle.h5  # noqa: F401
import h5py
import numpy as np
import pytest

from ess.nmx.configurations import (
    InputConfig,
    OutputConfig,
    ReductionConfig,
    WorkflowConfig,
)
from ess.nmx.data import get_small_nmx_nexus, get_small_nmx_reduced
from ess.nmx.executables import reduction
from ess.nmx.types import Compression


def assert_h5_attrs_equal(
    attrs_left: h5py.AttributeManager,
    attrs_right: h5py.AttributeManager,
    cur_path: pathlib.Path,
) -> None:

    assert attrs_left.keys() == attrs_right.keys()
    for attr_key, attr in attrs_left.items():
        if not isinstance(attr, str) and hasattr(attr, '__len__'):
            assert all(attr == attrs_right[attr_key]), cur_path
        else:
            assert attr == attrs_right[attr_key], cur_path


def assert_h5obj_equal(
    obj_left: h5py.Group | h5py.Dataset,
    obj_right: h5py.Group | h5py.Dataset,
    _cur_path: pathlib.Path = pathlib.Path('/'),
    excluded_paths: tuple[pathlib.Path, ...] = (),
) -> None:
    assert type(obj_left) is type(obj_right)
    assert_h5_attrs_equal(obj_left.attrs, obj_right.attrs, _cur_path)
    if isinstance(obj_left, h5py.Group) and isinstance(obj_right, h5py.Group):
        for key in obj_left.keys():
            cur_sub_path = _cur_path / key
            if cur_sub_path in excluded_paths:
                continue
            else:
                assert key in obj_right
                assert_h5obj_equal(
                    obj_left[key],
                    obj_right[key],
                    _cur_path=cur_sub_path,
                    excluded_paths=excluded_paths,
                )
    else:
        values_left = obj_left[()]
        values_right = obj_right[()]
        assert type(values_left) is type(values_right)
        if not isinstance(values_left, np.ndarray) or not isinstance(
            values_right, np.ndarray
        ):
            assert values_left == values_right, _cur_path
        else:
            assert values_left.shape == values_right.shape, _cur_path
            # We use np.allclose instead of np.all(left==right)
            # because we run ray-simulation on the fly
            # and the time-dependent values might slightly change.
            assert np.allclose(values_left, values_right), _cur_path


def test_compare_output_file_with_frozen(tmp_path: pathlib.Path):
    """Test that the executable runs and returns the expected output."""

    # Make a new output file from current implementation.
    input_file = get_small_nmx_nexus()
    output_file = tmp_path / "scipp_output_current.h5"
    assert not output_file.exists()
    config = ReductionConfig(
        inputs=InputConfig(input_file=[input_file.as_posix()]),
        workflow=WorkflowConfig(),
        output=OutputConfig(
            output_file=output_file.as_posix(),
            compression=Compression.NONE,
            skip_file_output=False,
        ),
    )
    with pytest.warns(RuntimeWarning, match="No crystal rotation*"):
        _ = reduction(config=config)

    entry_path = pathlib.Path('/entry')
    excluded_paths = (
        entry_path / 'reducer/program',  # version should be different
    )
    ref_file_path = get_small_nmx_reduced()
    with h5py.File(output_file) as cur_file:
        with h5py.File(ref_file_path) as reference_file:
            assert_h5obj_equal(reference_file, cur_file, excluded_paths=excluded_paths)
