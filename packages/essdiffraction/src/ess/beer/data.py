# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with BEER."""

from pathlib import Path

import scipp as sc

from ess.reduce.data import make_registry

__all__ = ["mcstas_duplex", "mcstas_silicon_medium_resolution"]

_registry = make_registry(
    "ess/beer",
    version="1",
    files={
        "duplex-mode07.h5": "md5:e8d44197e4bc6a84ab9265bfabd96efe",
        "duplex-mode08.h5": "md5:7cd2cf86d5d98fe07097ff98b250ba9b",
        "duplex-mode09.h5": "md5:ebb3f9694ffdd0949f342bd0deaaf627",
        "duplex-mode10.h5": "md5:559e7fc0cce265f5102520e382ee5b26",
        "duplex-mode16.h5": "md5:2ccd05832bbc8a087a731b37364b995d",
        "silicon-mode09.h5": "md5:aa068d46dc7efc303b68a13e527e2607",
        "mccode_quartz_mode10.h5": "md5:fd6ea529cad7cefdcb57bf57bc27668c",
        "mccode_quartz_mode16.h5": "md5:b9bef04bd6e4c60a8a0d55d4d6ebb89f",
        "mccode_quartz_mode7.h5": "md5:24c7006336f58c26689abec1645cfba1",
        "mccode_quartz_mode8.h5": "md5:9a4606ce56a6397d05d70246d54407d6",
        "mccode_quartz_mode9.h5": "md5:1394df47287ecd62bfa62c200203f214",
        "silicon-dhkl.tab": "md5:59ee9ed57a7c039ce416c8df886da9cc",
        "duplex-dhkl.tab": "md5:b4c6c2fcd66466ad291f306b2d6b346e",
        "dhkl_quartz_nc.tab": "md5:40887d736e3acf859e44488bfd9a9213",
        # Simulations from new model with corrected(?) L0.
        # For correct reduction you need to use
        # beer.io.mcstas_chopper_delay_from_mode_new_simulations
        # to obtain the correct WavelengthDefinitionChopperDelay for these files.
        "silicon-mode10-new-model.h5": "md5:98500830f27700fc719634e1acd49944",
        "silicon-mode16-new-model.h5": "md5:393f9287e7d3f97ceedbe64343918413",
        "silicon-mode3-new-model.h5": "md5:260fc811352b96c1dcdf431d91c11787",
        "silicon-mode4-new-model.h5": "md5:625561f6a57c6e6892e1bbd4043d71ee",
        "silicon-mode5-new-model.h5": "md5:315d457d136bb597d7f7ab596497324b",
        "silicon-mode6-new-model.h5": "md5:be9b48ee9db9fcf1808ed2f2e35f594b",
        "silicon-mode7-new-model.h5": "md5:d2070d3132722bb551d99b243c62752f",
        "silicon-mode8-new-model.h5": "md5:d6dfdf7e87eccedf4f83c67ec552ca22",
        "silicon-mode9-new-model.h5": "md5:694a17fb616b7f1c20e94d9da113d201",
    },
)


def mcstas_duplex(mode: int) -> Path:
    """
    Simulated intensity from duplex sample with ``mode`` chopper configuration.
    """
    return _registry.get_path(f'duplex-mode{mode:02}.h5')


def mcstas_quartz(mode: int) -> Path:
    """
    Simulated intensity from quartz sample with ``mode`` chopper configuration.
    """
    return _registry.get_path(f'mccode_quartz_mode{mode}.h5')


def mcstas_silicon_medium_resolution() -> Path:
    """
    Simulated intensity from silicon sample with
    medium resolution chopper configuration.
    """
    return _registry.get_path('silicon-mode09.h5')


def mcstas_silicon_new_model(mode: int) -> Path:
    """
    Simulated intensity from duplex sample with ``mode`` chopper configuration.
    """
    return _registry.get_path(f'silicon-mode{mode}-new-model.h5')


def duplex_peaks() -> Path:
    return _registry.get_path('duplex-dhkl.tab')


def quartz_peaks() -> Path:
    return _registry.get_path('dhkl_quartz_nc.tab')


def silicon_peaks() -> Path:
    return _registry.get_path('silicon-dhkl.tab')


def _read_peak_file_to_dataarray(name: Path):
    with open(name) as f:
        return sc.array(
            dims='d',
            values=sorted([float(x) for x in f.read().split('\n') if x != '']),
            unit='angstrom',
        )


def duplex_peaks_array() -> sc.Variable:
    return _read_peak_file_to_dataarray(duplex_peaks())


def silicon_peaks_array() -> sc.Variable:
    return _read_peak_file_to_dataarray(silicon_peaks())


def quartz_peaks_array() -> sc.Variable:
    return _read_peak_file_to_dataarray(quartz_peaks())
