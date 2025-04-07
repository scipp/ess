"""
DIALS .refl file loader

This loads msgpack-type DIALS reflection files, without having DIALS or
cctbx in the python environment.

Note: All modern .refl files are at time of writing msgpack-based. Some
much older files might be in pickle format, which this doesn't read.

Adapted from Nick Cavendish of the DIALS team.
"""

import functools
import logging
import operator
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, cast

import msgpack
import numpy as np


@dataclass
class Shoebox:
    panel: int
    bbox: tuple[int]
    data: np.array = None
    mask: np.array = None
    background: np.array = None


def _decode_raw_numpy(dtype, shape: int | Iterable = 1):
    """
    Decoding a column that maps straight to a numpy array.

    Args:
        dtype: The numpy dtype for the array
        shape:
            The shape of a single item. Either an int, or a collection
            of ints, in C-array order (row major)
    """
    # Convert to a shape tuple
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)

    def _decode_specific(data, copy):
        num_items, raw = data
        array = np.frombuffer(raw, dtype=dtype)

        if shape != (1,):
            item_width = functools.reduce(operator.mul, shape)
            if (len(raw) % item_width) != 0:
                raise AssertionError(
                    "Data length %s is not divisible by item width %s",
                    len(raw),
                    item_width,
                )
            elif (num_items * item_width) != len(array):
                raise AssertionError(
                    "Data length %s is not equal to "
                    "number of items %s times item width %s",
                    len(array),
                    num_items,
                    item_width,
                )
            array = array.reshape(num_items, *shape)
        if copy:
            return np.copy(array)
        return array

    return _decode_specific


def _decode_shoeboxes(data: list, copy) -> list[Shoebox | None]:
    # Shoebox is float
    num_items, raw = data
    shoeboxes: list[Shoebox | None] = []
    pos = 0
    while pos < len(raw):
        sbox_header_fmt = "<IiiiiiiB"
        sb_info = struct.unpack_from(sbox_header_fmt, raw, pos)
        pos += struct.calcsize(sbox_header_fmt)
        panel = sb_info[0]
        bbox = sb_info[1:7]
        data_present = sb_info[7]
        shoebox = {"panel": panel, "bbox": bbox}
        if data_present:
            bbox_size = (bbox[5] - bbox[4], bbox[3] - bbox[2], bbox[1] - bbox[0])
            data_size = (bbox_size[0] * bbox_size[1] * bbox_size[2]) * 4
            # Read three sets of data: data, mask and background
            shoebox["data"] = np.frombuffer(
                raw[pos : pos + data_size], dtype=np.float32
            ).reshape(bbox_size)
            pos += data_size
            shoebox["mask"] = np.frombuffer(
                raw[pos : pos + data_size], dtype=np.int32
            ).reshape(bbox_size)
            pos += data_size
            shoebox["background"] = np.frombuffer(
                raw[pos : pos + data_size], dtype=np.float32
            ).reshape(bbox_size)
            pos += data_size
            if copy:
                shoebox["data"] = np.copy(shoebox["data"])
                shoebox["mask"] = np.copy(shoebox["mask"])
                shoebox["background"] = np.copy(shoebox["background"])

        # Although this is technically a divergence,
        # return None instead of an empty shoebox
        if not data_present and all(x == 0 for x in bbox) and panel == 0:
            shoeboxes.append(None)
        else:
            shoeboxes.append(Shoebox(**shoebox))
    if len(shoeboxes) != num_items:
        raise AssertionError(
            "Warning: Mismatch of shoebox length: %s "
            "is not same as the number of items: %s",
            len(shoeboxes),
            num_items,
        )

    return np.array(shoeboxes, dtype=np.object_)


_reftable_decoders = {
    "bool": _decode_raw_numpy(bool),
    "int": _decode_raw_numpy(np.int32),
    "double": _decode_raw_numpy(np.double),
    "int6": _decode_raw_numpy(np.int32, shape=6),
    "std::size_t": _decode_raw_numpy(np.uint64),
    "vec3<double>": _decode_raw_numpy(np.double, shape=3),
    "cctbx::miller::index<>": _decode_raw_numpy(np.int32, shape=3),
    "Shoebox<>": _decode_shoeboxes,
    "vec2<double>": _decode_raw_numpy(np.double, shape=2),
    "mat3<double>": _decode_raw_numpy(np.double, shape=(3, 3)),
    # "std::string": _decode_wip, # - string writing broken; dials/dials#1858
}


def decode_column(column_entry, copy):
    """Decode a single column value"""
    datatype, data = column_entry

    converter = _reftable_decoders.get(datatype)
    if not converter:
        logging.warning(
            "Data type '%s' does not have a converter; cannot read", datatype
        )
        return None
    return converter(data, copy=copy)


def _get_unpacked(stream_or_path: str | IO | bytes | os.PathLike):
    """Works out the logic to pass a stream/pathlike to msgpack"""
    try:
        logging.INFO(type(stream_or_path))
        path = os.fspath(cast(str, stream_or_path))
        is_fspathlike = True
    except (TypeError, ValueError):
        path = stream_or_path
        is_fspathlike = isinstance(stream_or_path, str)

    if is_fspathlike:
        with open(path, "rb") as f:
            un = msgpack.Unpacker(f, strict_map_key=False)
            return un.unpack()
    else:
        un = msgpack.Unpacker(stream_or_path, strict_map_key=False)
        return un.unpack()


def loads(data: bytes, copy=False):
    """
    Load a DIALS msgpack-encoded .refl file.

    Args:
        data: bytes data, already read from the file.
        copy: Should the data be copied into writable numpy arrays.

    Returns: See .load(stream_or_path)
    """
    return load_reflection_file(BytesIO(data), copy)


def load_reflection_file(stream_or_path: IO | Path | os.PathLike, copy=False) -> dict:
    """
    Load a DIALS msgpack-encoded .refl file

    Args:
        stream_or_path: The filename or data to load
        copy:
            Should the data be copied. This will cause more memory usage
            whilst loading the raw data.

    Returns:

        A dictionary with each column in the reflection table. If there
        is an identifier mapping as part of the reflection table, then
        this is returned as an extra 'experiment_identifier' column.
        All columns except Shoeboxes are returned as numpy arrays,
        except Shoebox columns, which are returned as Dataclass objects
        which contain the portions of data from the file.

        With copy=False, all numpy arrays are pointing against the raw
        memory returned by msgpack, which means they are read-only.
        With copy=True, an immediate copy is done. This causes memory
        usage to double while loading, but the created numpy arrays own
        their own memory.
    """
    root_data = _get_unpacked(stream_or_path)

    if not root_data[0] == "dials::af::reflection_table":
        raise ValueError("Does not appear to be a dials reflection table file")
    if not root_data[1] == 1:
        raise ValueError(
            f"reflection_table data is version {root_data[1]}. "
            "Only Version 1 is understood"
        )
    refdata = root_data[2]

    rows = refdata["nrows"]
    identifiers = refdata["identifiers"]
    data = refdata["data"]

    decoded_data = {
        name: decode_column(value, copy=copy) for name, value in data.items()
    }

    # Filter out empty (unknown) columns
    decoded_data = {k: v for k, v in decoded_data.items() if v is not None}

    # Cross-check the columns are the expected lengths
    for name, column in decoded_data.items():
        if len(column) != rows:
            logging.warning(
                "Warning: Mismatch of column lengths: %s is %s instead of expected %s",
                name,
                len(column),
                rows,
            )

    # Make an "identifiers" column
    if "id" in decoded_data and identifiers:
        decoded_data["experiment_identifier"] = [
            identifiers[x] for x in decoded_data["id"] if x > 0
        ]

    return decoded_data
