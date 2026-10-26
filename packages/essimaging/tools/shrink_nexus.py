#!/usr/bin/env python3
"""
Trim a NeXus neutron experiment file to keep only the first N pulses
in all NXevent_data groups, writing the result to a new file.

Usage:
    python trim_nexus_pulses.py <input_file> <output_file> <n_pulses>
"""

import argparse
import os
import shutil
import subprocess

import h5py


def repack(filepath):
    """Run h5repack on filepath in-place (repacks to a temp file then replaces)."""
    if shutil.which("h5repack") is None:
        print("Warning: 'h5repack' not found on PATH, skipping repacking.")  # noqa: T201
        return

    tmp = filepath + ".repacked"
    try:
        subprocess.run(["h5repack", filepath, tmp], check=True)  # noqa: S603, S607
        shutil.move(tmp, filepath)
        print("h5repack completed successfully.")  # noqa: T201
    except subprocess.CalledProcessError as e:
        print(  # noqa: T201
            f"Warning: h5repack failed (exit code {e.returncode}), "
            "output file is kept as-is."
        )
        if os.path.exists(tmp):
            os.remove(tmp)


def find_nxevent_data_groups(h5file):
    """Recursively find all groups with NX_class == 'NXevent_data'."""
    groups = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            nx_class = obj.attrs.get("NX_class")
            # The attribute may be bytes or str depending on how it was written
            if isinstance(nx_class, bytes):
                nx_class = nx_class.decode()
            if nx_class == "NXevent_data":
                groups.append(name)

    h5file.visititems(visitor)
    return groups


def trim_nxevent_data(group, n_pulses):
    """
    Trim an NXevent_data group in-place (on a copy) to the first n_pulses.

    Parameters
    ----------
    group : h5py.Group
        The NXevent_data group to modify.
    n_pulses : int
        Number of pulses to keep.
    """
    # -- Pulse-level arrays (shape N) -----------------------------------------
    event_time_zero = group["event_time_zero"][:]
    event_index = group["event_index"][:]

    actual_pulses = len(event_time_zero)
    if n_pulses >= actual_pulses:
        print(  # noqa: T201
            f"  Group '{group.name}': only {actual_pulses} pulse(s) available, "
            "nothing to trim."
        )
        return

    # -- Determine how many events belong to the first n_pulses ---------------
    # event_index[i] is the start index of pulse i in the flat event arrays.
    # The end of the last kept pulse is event_index[n_pulses] if it exists,
    # otherwise it's the full length of the event arrays.
    if n_pulses < len(event_index):
        n_events = int(event_index[n_pulses])
    else:
        # n_pulses == actual_pulses (already handled above, but be safe)
        n_events = len(group["event_id"])

    print(  # noqa: T201
        f"  Group '{group.name}': keeping {n_pulses}/{actual_pulses} pulses "
        f"and {n_events}/{len(group['event_id'])} events."
    )

    # -- Trim each dataset and replace it -------------------------------------
    trimmed = {
        "event_time_zero": event_time_zero[:n_pulses],
        "event_index": event_index[:n_pulses],
        "event_id": group["event_id"][:n_events],
        "event_time_offset": group["event_time_offset"][:n_events],
    }

    for name, data in trimmed.items():
        # Preserve attributes (units, NX_class, etc.) before deleting
        attrs = dict(group[name].attrs)

        del group[name]

        ds = group.create_dataset(name, data=data)
        for attr_key, attr_val in attrs.items():
            ds.attrs[attr_key] = attr_val


def main():
    parser = argparse.ArgumentParser(
        description="Trim a NeXus file to the first N pulses in every "
        "NXevent_data group."
    )
    parser.add_argument(
        "input_file", help="Path to the input NeXus (.nxs / .hdf5) file"
    )
    parser.add_argument("output_file", help="Path for the trimmed output file")
    parser.add_argument(
        "-n",
        "--n-pulses",
        type=int,
        required=True,
        dest="n_pulses",
        help="Number of pulses to keep in each NXevent_data group",
    )
    args = parser.parse_args()

    if args.n_pulses < 1:
        parser.error("n_pulses must be a positive integer.")

    # Copy the entire file so we never touch the original
    print(f"Copying '{args.input_file}' -> '{args.output_file}' ...")  # noqa: T201
    shutil.copyfile(args.input_file, args.output_file)

    with h5py.File(args.output_file, "r+") as f:
        groups = find_nxevent_data_groups(f)

        if not groups:
            print("No NXevent_data groups found. Output file is an unmodified copy.")  # noqa: T201
            return

        print(f"Found {len(groups)} NXevent_data group(s): {groups}\n")  # noqa: T201

        for group_path in groups:
            trim_nxevent_data(f[group_path], args.n_pulses)

    print("\nRepacking output file...")  # noqa: T201
    repack(args.output_file)

    print("\nDone.")  # noqa: T201


if __name__ == "__main__":
    main()
