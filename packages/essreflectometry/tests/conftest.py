# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--file-output", help='Output folder for reduced data')


@pytest.fixture
def output_folder(request: pytest.FixtureRequest) -> Path:
    if (path := request.config.getoption("--file-output")) is not None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        return out
    return request.getfixturevalue("tmp_path")
