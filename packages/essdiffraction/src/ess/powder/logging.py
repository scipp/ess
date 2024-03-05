# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Logging tools for ess.power."""

import logging


def get_logger() -> logging.Logger:
    """Return the logger for ess.powder.

    Returns
    -------
    :
        The requested logger.
    """
    return logging.getLogger('scipp.ess.powder')
