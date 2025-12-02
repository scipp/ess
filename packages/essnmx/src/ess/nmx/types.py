import enum


class Compression(enum.StrEnum):
    """Compression type of the output file.

    These options are written as enum for future extensibility.
    """

    NONE = 'NONE'
    BITSHUFFLE_LZ4 = 'BITSHUFFLE_LZ4'
