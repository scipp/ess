# ADR 0001: File format for pixel masks

- Status: proposed
- Date: 2026-11

## Context

Instrument scientists will create pixel masks, which are stored as files and used by reduction workflows for years.
Some are static, such as broken tubes, and are replaced when the hardware changes (see [#767](https://github.com/scipp/ess/issues/767)).
A file that exists in a data catalog cannot be changed by a software release, so this format is harder to change than the code that reads it.
The files must remain readable without ESS software or a particular version of it.

The existing formats are ISIS XML and scipp HDF5 files holding a list of masked detector IDs.
Neither has a place for a bank, a name, or a validity date.
The planned masking tool for LoKI works on tubes and straws of a bank, after the detector data is folded into its logical dimensions.

## Decision

A mask file is a NeXus file, which is an HDF5 file with a defined structure:

```text
/entry                             NXentry
    title                          "broken-tubes", optional
    valid_from                     "2026-11-01T00:00:00Z", not read by workflows
    instrument                     NXinstrument
        loki_detector_0            NXdetector
            pixel_mask             int32, HDF5 dimension labels ("layer", "tube")
        loki_detector_1            NXdetector
            pixel_mask             ...
```

The content can be read with any HDF5 library:

```python
with h5py.File(path) as f:
    dataset = f['entry/instrument/loki_detector_0/pixel_mask']
    masked = (dataset[()] & 0xFFFF) != 0
    dims = [dim.label for dim in dataset.dims]
```

- Names of `NXdetector` groups are the names of the groups in the NeXus files of the runs.
  A bank without `pixel_mask` has no mask from this file.
  A name that is not a bank of the instrument is an error, since a misspelled name would otherwise mask nothing.
- `pixel_mask` is the integer bit field defined by NeXus for [`NXdetector`](https://manual.nexusformat.org/classes/base_classes/NXdetector.html).
  A pixel is masked if any bit in `pixel_mask & 0xFFFF` is set, as NeXus specifies.
  The other bits are tags, which do not mask a pixel.
  `ess.reduce.masking.write_pixel_masks` sets bit 8, "user defined mask".
- Dimensions of `pixel_mask` are named with HDF5 dimension labels.
  They are a subset of the logical dimensions of the bank, with matching sizes.
  This is checked when masks are applied, because scipp ignores a mask with a dimension that the data lacks, without an error.
- Each file gives one mask per bank.
  Tube and straw masks combine into one `(layer, tube, straw)` array.
  The `pixel_mask_N` fields of NeXus are an error, so that they are not ignored.
- `title` becomes the name of the mask in the reduced data.
  Without it, the name of the file is used.
  Two masks with the same name are an error.

Workflows take a list of filenames and read all `pixel_mask` fields.
They do not check validity dates.
Selecting the files that are valid for a run is left to the caller of the workflow, which sets the list of filenames.

The readers for the existing formats stay, since they carry a list of detector IDs that works without a bank name.
New files should not use scipp HDF5.

## Alternatives considered

- **A scipp HDF5 file holding a `DataGroup`.**
  This needs no writer, but the layout is defined by the code of scipp and not by a specification, so an archived file would depend on that code.
- **HDF5 with our own group names and a boolean dataset.**
  It would need its own specification of what NeXus already defines.
  HDF5 also has no boolean type: h5py stores booleans as an enumeration, which is a convention of h5py.
- **Lists of detector IDs, for new files too.**
  `detector_number` is unique across the banks of LoKI, so such a file needs no bank level, and it stays correct if the layout of a bank changes.
  It cannot express a mask of a tube or a straw except by listing every pixel.
- **Workflows check the validity date.**
  This would require each workflow to know the time of a run and how to handle expired masks, while a list of filenames is enough for it to do its job.

## Consequences

- The format goes beyond the NeXus definition in three places.
  NeXus describes `pixel_mask` with rank 2, while masks here have the rank of the logical dimensions they use.
  Dimension labels are an HDF5 feature, not defined by NeXus.
  `valid_from` is not a field that NeXus defines for `NXentry`.
- Boolean masks are positional.
  A change of a bank's layout that keeps its shape would apply old masks to the wrong pixels without an error.
  Mask files must be recreated when the layout of a bank changes.
- Mask files have no format version.
  A later incompatible change needs a different location than `/entry/instrument/<bank>/pixel_mask`.
