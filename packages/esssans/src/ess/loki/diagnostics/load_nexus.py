# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from functools import partial

from ess.reduce.diagnostics import load_nexus

from ..workflow import DETECTOR_BANK_SIZES

load_detectors = partial(load_nexus.load_detectors, bank_sizes=DETECTOR_BANK_SIZES)
load_monitors = load_nexus.load_monitors
