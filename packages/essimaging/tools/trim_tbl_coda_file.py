"""
Trim a TBL NeXus file to keep only certain entries.
"""

import h5py
import posixpath

input_file = "coda_tbl_999999_00026230.hdf"
output_file = "filtered.hdf"

# Absolute HDF5 paths to keep
keep_paths = [
    "/entry/sample",
    "/entry/start_time",
    "/entry/end_time",
    "/entry/instrument/beam_monitor_1",
    "/entry/instrument/bwc_1",
    "/entry/instrument/bwc_2",
    "/entry/instrument/he3_bank_0",
    "/entry/instrument/he3_bank_1",
    "/entry/instrument/multiblade",
    "/entry/instrument/ngem",
    "/entry/instrument/timepix3",
    "/entry/instrument/source",
    "/entry/instrument/moderator",
]


def filter_hdf5(input_file, output_file, keep_paths):
    # Normalize paths and remove duplicates
    keep_paths = sorted(set(posixpath.normpath(p) for p in keep_paths))

    # If a parent is already selected, its entire subtree is retained.
    selected = []
    for path in keep_paths:
        if not path.startswith("/") or path == "/":
            raise ValueError(f"Expected a non-root absolute path: {path}")
        if not any(
            path.startswith(parent.rstrip("/") + "/")
            for parent in selected
        ):
            selected.append(path)

    with h5py.File(input_file, "r") as src, \
         h5py.File(output_file, "w") as dst:

        def ensure_parent_groups(path):
            """Create required parent groups and preserve their attributes."""
            parts = path.strip("/").split("/")[:-1]
            current = ""

            for part in parts:
                current += "/" + part
                if current not in dst:
                    group = dst.create_group(current)
                    for key, value in src[current].attrs.items():
                        group.attrs[key] = value

        for path in selected:
            if path not in src:
                raise KeyError(f"Path not found in input file: {path}")

            ensure_parent_groups(path)

            parent = posixpath.dirname(path)
            name = posixpath.basename(path)
            dst_parent = dst[parent or "/"]

            # Copies datasets or groups, including dataset contents,
            # group descendants, and object attributes.
            src.copy(path, dst_parent, name=name)

    print(f"Filtered file written to: {output_file}")


filter_hdf5(input_file, output_file, keep_paths)