# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import enum

import scipp as sc
from scippneutron.chopper import DiskChopper


class InstrumentConfiguration(enum.Enum):
    """Choose between high-flux and high-resolution configurations."""

    high_flux = enum.auto()
    high_resolution = enum.auto()


def choppers(configuration: InstrumentConfiguration) -> dict[str, DiskChopper]:
    """Return the chopper configuration for the given instrument configuration."""

    match configuration:
        case InstrumentConfiguration.high_flux:
            return {
                "psc1": DiskChopper(
                    frequency=sc.scalar(14.0, unit="Hz"),
                    beam_position=sc.scalar(0.0, unit="deg"),
                    phase=sc.scalar(286 - 180, unit="deg"),
                    axle_position=sc.vector(value=[0, 0, 6.145], unit="m"),
                    slit_begin=sc.array(
                        dims=["cutout"],
                        values=[
                            -1.23,
                            70.49,
                            84.765,
                            113.565,
                            170.29,
                            271.635,
                            286.035,
                            301.17,
                        ],
                        unit="deg",
                    ),
                    slit_end=sc.array(
                        dims=["cutout"],
                        values=[
                            1.23,
                            73.51,
                            88.035,
                            116.835,
                            175.31,
                            275.565,
                            289.965,
                            303.63,
                        ],
                        unit="deg",
                    ),
                    slit_height=sc.scalar(10.0, unit="cm"),
                    radius=sc.scalar(30.0, unit="cm"),
                ),
                "psc2": DiskChopper(
                    frequency=sc.scalar(-14.0, unit="Hz"),
                    beam_position=sc.scalar(0.0, unit="deg"),
                    phase=sc.scalar(-236, unit="deg"),
                    axle_position=sc.vector(value=[0, 0, 6.155], unit="m"),
                    slit_begin=sc.array(
                        dims=["cutout"],
                        values=[
                            -1.23,
                            27.0,
                            55.8,
                            142.385,
                            156.765,
                            214.115,
                            257.23,
                            315.49,
                        ],
                        unit="deg",
                    ),
                    slit_end=sc.array(
                        dims=["cutout"],
                        values=[
                            1.23,
                            30.6,
                            59.4,
                            145.615,
                            160.035,
                            217.885,
                            261.17,
                            318.11,
                        ],
                        unit="deg",
                    ),
                    slit_height=sc.scalar(10.0, unit="cm"),
                    radius=sc.scalar(30.0, unit="cm"),
                ),
                "oc": DiskChopper(
                    frequency=sc.scalar(14.0, unit="Hz"),
                    beam_position=sc.scalar(0.0, unit="deg"),
                    phase=sc.scalar(297 - 180 - 90, unit="deg"),
                    axle_position=sc.vector(value=[0, 0, 6.174], unit="m"),
                    slit_begin=sc.array(
                        dims=["cutout"], values=[-27.6 * 0.5], unit="deg"
                    ),
                    slit_end=sc.array(dims=["cutout"], values=[27.6 * 0.5], unit="deg"),
                    slit_height=sc.scalar(10.0, unit="cm"),
                    radius=sc.scalar(30.0, unit="cm"),
                ),
                "bcc": DiskChopper(
                    frequency=sc.scalar(112.0, unit="Hz"),
                    beam_position=sc.scalar(0.0, unit="deg"),
                    phase=sc.scalar(215 - 180, unit="deg"),
                    # Use 240 to reduce overlap between frames
                    # phase=sc.scalar(240 - 180, unit="deg"),
                    axle_position=sc.vector(value=[0, 0, 9.78], unit="m"),
                    slit_begin=sc.array(
                        dims=["cutout"], values=[-36.875, 143.125], unit="deg"
                    ),
                    slit_end=sc.array(
                        dims=["cutout"], values=[36.875, 216.875], unit="deg"
                    ),
                    slit_height=sc.scalar(10.0, unit="cm"),
                    radius=sc.scalar(30.0, unit="cm"),
                ),
                "t0": DiskChopper(
                    frequency=sc.scalar(28.0, unit="Hz"),
                    beam_position=sc.scalar(0.0, unit="deg"),
                    phase=sc.scalar(280 - 180, unit="deg"),
                    axle_position=sc.vector(value=[0, 0, 13.05], unit="m"),
                    slit_begin=sc.array(
                        dims=["cutout"], values=[-314.9 * 0.5], unit="deg"
                    ),
                    slit_end=sc.array(
                        dims=["cutout"], values=[314.9 * 0.5], unit="deg"
                    ),
                    slit_height=sc.scalar(10.0, unit="cm"),
                    radius=sc.scalar(30.0, unit="cm"),
                ),
            }
        case InstrumentConfiguration.high_resolution:
            return (
                {
                    "psc1": DiskChopper(
                        frequency=sc.scalar(15 * 14.0, unit="Hz"),
                        beam_position=sc.scalar(0.0, unit="deg"),
                        phase=sc.scalar(25 - 180, unit="deg"),
                        axle_position=sc.vector(value=[0, 0, 6.145], unit="m"),
                        slit_begin=sc.array(
                            dims=["cutout"],
                            values=[
                                -1.23,
                                70.49,
                                84.765,
                                113.565,
                                170.29,
                                271.635,
                                286.035,
                                301.17,
                            ],
                            unit="deg",
                        ),
                        slit_end=sc.array(
                            dims=["cutout"],
                            values=[
                                1.23,
                                73.51,
                                88.035,
                                116.835,
                                175.31,
                                275.565,
                                289.965,
                                303.63,
                            ],
                            unit="deg",
                        ),
                        slit_height=sc.scalar(10.0, unit="cm"),
                        radius=sc.scalar(30.0, unit="cm"),
                    ),
                    "psc2": DiskChopper(
                        frequency=sc.scalar(-14.0 * 14, unit="Hz"),
                        beam_position=sc.scalar(0.0, unit="deg"),
                        phase=sc.scalar(-100.5, unit="deg"),
                        axle_position=sc.vector(value=[0, 0, 6.155], unit="m"),
                        slit_begin=sc.array(
                            dims=["cutout"],
                            values=[
                                -1.23,
                                27.0,
                                55.8,
                                142.385,
                                156.765,
                                214.115,
                                257.23,
                                315.49,
                            ],
                            unit="deg",
                        ),
                        slit_end=sc.array(
                            dims=["cutout"],
                            values=[
                                1.23,
                                30.6,
                                59.4,
                                145.615,
                                160.035,
                                217.885,
                                261.17,
                                318.11,
                            ],
                            unit="deg",
                        ),
                        slit_height=sc.scalar(10.0, unit="cm"),
                        radius=sc.scalar(30.0, unit="cm"),
                    ),
                    "oc": DiskChopper(
                        frequency=sc.scalar(14.0, unit="Hz"),
                        beam_position=sc.scalar(0.0, unit="deg"),
                        phase=sc.scalar(297 - 180 - 90, unit="deg"),
                        axle_position=sc.vector(value=[0, 0, 6.174], unit="m"),
                        slit_begin=sc.array(
                            dims=["cutout"], values=[-27.6 * 0.5], unit="deg"
                        ),
                        slit_end=sc.array(
                            dims=["cutout"], values=[27.6 * 0.5], unit="deg"
                        ),
                        slit_height=sc.scalar(10.0, unit="cm"),
                        radius=sc.scalar(30.0, unit="cm"),
                    ),
                    "bcc": DiskChopper(
                        frequency=sc.scalar(112.0, unit="Hz"),
                        beam_position=sc.scalar(0.0, unit="deg"),
                        phase=sc.scalar(200 - 180, unit="deg"),
                        axle_position=sc.vector(value=[0, 0, 9.78], unit="m"),
                        slit_begin=sc.array(
                            dims=["cutout"], values=[-36.875, 143.125], unit="deg"
                        ),
                        slit_end=sc.array(
                            dims=["cutout"], values=[36.875, 216.875], unit="deg"
                        ),
                        slit_height=sc.scalar(10.0, unit="cm"),
                        radius=sc.scalar(30.0, unit="cm"),
                    ),
                    "t0": DiskChopper(
                        frequency=sc.scalar(28.0, unit="Hz"),
                        beam_position=sc.scalar(0.0, unit="deg"),
                        phase=sc.scalar(270 - 180, unit="deg"),
                        axle_position=sc.vector(value=[0, 0, 13.05], unit="m"),
                        slit_begin=sc.array(
                            dims=["cutout"], values=[-314.9 * 0.5], unit="deg"
                        ),
                        slit_end=sc.array(
                            dims=["cutout"], values=[314.9 * 0.5], unit="deg"
                        ),
                        slit_height=sc.scalar(10.0, unit="cm"),
                        radius=sc.scalar(30.0, unit="cm"),
                    ),
                },
            )
