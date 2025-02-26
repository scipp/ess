# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import Any

import scipp as sc
import scippnexus as snx

from ess.reduce.streaming import Accumulator

from .mcstas.load import load_event_data_bank_name
from .types import DetectorBankPrefix, DetectorName, FilePath


class MinAccumulator(Accumulator):
    """Accumulator that keeps track of the maximum value seen so far."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cur_min: sc.Variable | None = None

    @property
    def value(self) -> sc.Variable | None:
        return self._cur_min

    def _do_push(self, value: sc.Variable) -> None:
        new_min = value.min()
        if self._cur_min is None:
            self._cur_min = new_min
        else:
            self._cur_min = min(self._cur_min, new_min)


class MaxAccumulator(Accumulator):
    """Accumulator that keeps track of the maximum value seen so far."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cur_max: sc.Variable | None = None

    @property
    def value(self) -> sc.Variable | None:
        return self._cur_max

    def _do_push(self, value: sc.Variable) -> None:
        new_max = value.max()
        if self._cur_max is None:
            self._cur_max = new_max
        else:
            self._cur_max = max(self._cur_max, new_max)


def calculate_number_of_chunks(
    file_path: FilePath,
    *,
    detector_name: DetectorName,
    bank_prefix: DetectorBankPrefix | None = None,
    chunk_size: int = 0,  # Number of rows to read at a time
) -> int:
    """Calculate number of chunks in the event data.

    Parameters
    ----------
    file_path:
        Path to the nexus file
    detector_name:
        Name of the detector to load
    pixel_ids:
        Pixel ids to generate the data array with the events
    chunk_size:
        Number of rows to read at a time.
        If 0, chunk slice is determined automatically by the ``iter_chunks``.
        Note that it only works if the dataset is already chunked.

    """
    # Find the data bank name associated with the detector
    bank_prefix = load_event_data_bank_name(
        detector_name=detector_name, file_path=file_path
    )
    bank_name = f'{bank_prefix}_dat_list_p_x_y_n_id_t'
    with snx.File(file_path, 'r') as f:
        root = f["entry1/data"]
        (bank_name,) = (name for name in root.keys() if bank_name in name)
    with snx.File(file_path, 'r') as f:
        root = f["entry1/data"]
        dset: snx.Field = root[bank_name]["events"]
        if chunk_size == 0:
            return len(list(dset.dataset.iter_chunks()))
        else:
            return dset.shape[0] // chunk_size + int(dset.shape[0] % chunk_size != 0)
