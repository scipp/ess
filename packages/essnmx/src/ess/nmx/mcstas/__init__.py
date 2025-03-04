# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


def McStasWorkflow():
    import sciline as sl

    from ess.nmx.reduction import (
        calculate_maximum_toa,
        calculate_minimum_toa,
        format_nmx_reduced_data,
        proton_charge_from_event_counts,
        raw_event_probability_to_counts,
        reduce_raw_event_probability,
    )

    from .load import providers as loader_providers
    from .xml import read_mcstas_geometry_xml

    return sl.Pipeline(
        (
            *loader_providers,
            calculate_maximum_toa,
            calculate_minimum_toa,
            read_mcstas_geometry_xml,
            proton_charge_from_event_counts,
            reduce_raw_event_probability,
            raw_event_probability_to_counts,
            format_nmx_reduced_data,
        )
    )
