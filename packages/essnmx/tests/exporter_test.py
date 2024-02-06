# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from unittest import mock

import pytest


class FakeGroup(dict):
    def create_dataset(self, name, data, **_):
        self[name] = data
        return self[name]


class FakeFile(dict):
    def create_group(self, name):
        self[name] = FakeGroup()
        return self[name]


@pytest.fixture
def mock_h5py():
    with mock.patch('ess.nmx.reduction.h5py') as mock_h5py:
        yield mock_h5py


def test_exports(mock_h5py):
    from ess.nmx.reduction import h5py

    fake_file = FakeFile()
    mock_h5py.File.return_value.__enter__.return_value = fake_file

    with h5py.File('fake_file', 'w') as f:
        f.create_group('entry1')
        assert f == {'entry1': {}}
