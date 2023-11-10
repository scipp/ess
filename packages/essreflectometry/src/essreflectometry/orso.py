# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import warnings


def not_found_warning():
    """
    A function to raise a orso specific error if necessary.
    """
    warnings.warn(
        "For metadata to be logged in the data array, "
        "it is necessary to install the orsopy package.",
        UserWarning,
        stacklevel=2,
    )
