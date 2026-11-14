# ADR 0001: File format for pixel masks

- Status: proposed
- Date: 2026-11

## Context

Instrument scientists will create pixel masks, which are stored as files and used by reduction workflows for years.
Some are static, such as broken tubes, and are replaced when the hardware changes (see [#767](https://github.com/scipp/ess/issues/767)).
Others depend on the experiment, such as the shadow of a beamstop.
A file that exists in a data catalog cannot be changed by a software release, so this format is harder to change than the code that reads it.
The files must remain readable without ESS software or a particular version of it.

The existing formats are ISIS XML and scipp HDF5 files holding a list of masked detector IDs.
Neither has a place for a bank, a name, or a validity date.
The planned masking tool for LoKI works on tubes and straws of a bank, after the detector data is folded into its logical dimensions.

## Decision

A mask file is an HDF5 file with the group structure of NeXus:

```text
/                                  attributes file_time, creator and creator_version
/entry                             NXentry
    valid_from                     "2026-11-01T00:00:00Z", not read by workflows
    instrument                     NXinstrument
        name                       "LOKI"
        loki_detector_0            NXdetector
            masks                  NXcollection
                broken_tubes       uint8, HDF5 dimension labels ("tube",)
                beamstop           uint8, HDF5 dimension labels ("layer", "tube", "straw")
        loki_detector_1            NXdetector
            masks                  NXcollection
                broken_tubes       ...
```

The content can be read with any HDF5 library:

```python
with h5py.File(path) as f:
    dataset = f['entry/instrument/loki_detector_0/masks/broken_tubes']
    masked = dataset[()] != 0
    dims = [dim.label for dim in dataset.dims]
```

- Names of `NXdetector` groups are the names of the groups in the NeXus files of the runs.
  A name that is not a bank of the instrument is an error, since a misspelled name would otherwise mask nothing.
- The name of a dataset in `masks` is the name of the mask in the reduced data.
  Masks are associated with their detector group; a detector without a dataset does
  not get that mask.
- A pixel is masked if its value is not zero.
  HDF5 has no boolean type.
  h5py stores numpy booleans as an enumeration based on `int8`, which other HDF5 libraries may not read as booleans.
  `ess.reduce.masking.write_pixel_masks` therefore writes `uint8`, the smallest plain integer type.
  Readers accept any integer type, and the boolean enumeration that h5py writes for numpy booleans.
- Dimensions of a mask are named with HDF5 dimension labels.
  They are a subset of the logical dimensions of the bank, in any order, with matching sizes.
  Masks with the same name in different banks use the same dimension labels, though their sizes may differ.
  Each mask has its own dimensions, so a tube mask and a straw mask need not be combined into one array.
  The sizes are checked when masks are applied, because scipp ignores a mask with a dimension that the data lacks, without an error.
- A file with several `NXentry` groups is an error, since the reader would use only one of them and leave pixels unmasked.
- As in raw data files, `/entry/instrument/name` holds the name of the instrument, and the NeXus root attributes `file_time`, `creator` and `creator_version` hold the time, the software and the software version that wrote the file.
  Bank names alone do not always make the instrument obvious.
  `file_time` gives a rough idea of whether a file may still be valid, also when it has no `valid_from`.
  `write_pixel_masks` writes all three, and workflows do not read them.

Workflows take a list of filenames and use the masks of all files.
Two files with a mask of the same name for the same detector are an error.
A file can hold all masks of an instrument, or masks can be split into files, for example by validity period: one file for broken tubes and one for the beamstop of an experiment.
Workflows do not check validity dates.
Selecting the files that are valid for a run is left to the caller of the workflow, which sets the list of filenames.

The readers for the existing formats stay, since they carry a list of detector IDs that works without a bank name.
Such a mask is named after its file.
New files should not use scipp HDF5.

## Alternatives considered

- **The NeXus field `pixel_mask` of `NXdetector`.**
  This is the field NeXus defines for masks: a bit field where a pixel is masked if any bit in `pixel_mask & 0xFFFF` is set.
  It allows one mask per bank, with no name.
  The additional fields `pixel_mask_N` have no names either, and NeXus defines them as parts of one mask, combined with a bitwise OR.
  A file could therefore hold only one named mask, and a tube mask and a straw mask of the same bank would have to be combined into one array.
  Naming masks by file or by `NXentry` title would work around this, at the cost of many files per instrument.
  NeXus also describes `pixel_mask` with rank 2, while our masks have the rank of the logical dimensions they use, so we would deviate from the definition anyway.
  No tool outside ESS is expected to read these files, so following the definition gains little.
  The name `pixel_mask` is also confusing for LoKI, which has a dimension `pixel`.
- **Masks as datasets directly in `NXdetector`, without the `masks` group.**
  NeXus defines many fields of `NXdetector`, such as `data`, `detector_number`, `depends_on` and `pixel_mask`.
  Mask names would have to avoid all of them, and a reader could not tell a mask from another field.
  NeXus describes `NXcollection` as the place for terms that it does not define, and the group keeps masks apart if a file ever holds other fields of a detector, such as geometry.
- **A scipp HDF5 file holding a `DataGroup`.**
  This needs no writer, but the layout is defined by the code of scipp and not by a specification, so an archived file would depend on that code.
- **Boolean datasets.**
  HDF5 has no boolean type.
  h5py stores numpy booleans as an enumeration, which is a convention of h5py.
  `uint8` is a standard HDF5 type that every HDF5 library reads, and it has the same size.
- **The NeXus attribute `axes` for the names of dimensions, instead of or in addition to HDF5 dimension labels.**
  NeXus defines `axes` on `NXdata`, as the names of the coordinate fields along each dimension of the signal.
  `NXdetector` does not define it, and a mask has no coordinates, so a correct value would be `"."` for each dimension, which holds no names.
  The names decide how a mask of lower rank applies to the pixels, so they should be stored in one place only.
- **Lists of detector IDs, for new files too.**
  `detector_number` is unique across the banks of LoKI, so such a file needs no bank level, and it stays correct if the layout of a bank changes.
  It cannot express a mask of a tube or a straw except by listing every pixel.
- **Workflows check the validity date.**
  This would require each workflow to know the time of a run and how to handle expired masks, while a list of filenames is enough for it to do its job.
- **The NeXus fields `start_time` and `end_time` for the validity period.**
  NeXus defines them as the start and end of a measurement, which can differ from the validity of a mask.
  For example, a mask derived from a run in November can be valid from the day in October when a tube broke.
  A reader of an archived file would also take them for measurement times.
  `end_time` is usually unknown when a file is archived, since a static mask stops being valid when the hardware changes, and the file cannot be edited then.

## Consequences

- The file uses the group classes of NeXus, but the masks are not fields that NeXus defines.
  A generic NeXus tool shows them as datasets of an `NXcollection`, without knowing that they are masks.
- `valid_from` applies to all masks in a file.
  Masks with different validity periods go into different files.
  `valid_from` is provisional: nothing reads it yet, and the software that selects mask files for a run may need a different field, or may keep the date in catalog metadata.
- Boolean masks are positional.
  A change of a bank's layout that keeps its shape would apply old masks to the wrong pixels without an error.
  Mask files must be recreated when the layout of a bank changes.
- Mask files have no format version.
  A later incompatible change needs a different location than `/entry/instrument/<bank>/masks`.
