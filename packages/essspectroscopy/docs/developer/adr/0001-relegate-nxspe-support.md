# ADR 0001: Relagate NXspe support

- Status: accepted
- Deciders: Greg, Jan-Lukas
- Date: 2024-10

## Context
Targeting community supported data formats will enable efficient use of existing analysis and visualization tools.
A number of histogram formats have been [listed for possible support](https://jira.ess.eu/browse/DMSCSPEC-46):
- [`NXspe`](https://jira.ess.eu/browse/DMSCSPEC-50)
- [`SQW`](https://jira.ess.eu/browse/DMSCSPEC-51)
- [`MJOLNIR`](https://jira.ess.eu/browse/DMSCSPEC-52)

### `SQW` Prioritization and Challenges

One of which, [`SQW`](https://github.com/scipp/essspectroscopy/issues/23),
has been identified as higher importance because it will enable interfacing with
the PACE tools.

The major drawback to the `SQW` format is that it is documented most thoroughly by
its `MATLAB` serialization/deserialization routines, which are less than straightforward
to replicate in part due to an extensive class hierarchy, single-source backwards
compatibility over multiple file format versions, the ability to mix definition and
implementation of functions and place either in at least three different locations
(the `rundata` helper class seems to be spread over six),
and the lack of a free IDE that can help sort out the runtime behavior.

Recently it was posited that, since `Horace` can read `NXspe` files and produce a valid
`SQW` file from them, perhaps focusing on their output from `essspectroscopy` could
be prudent.

### Bending `NXspe` to Indirect-Geometry Time-of-Flight
`NXspe` is the [`NeXus` application definition](https://manual.nexusformat.org/classes/applications/NXspe.html#nxspe)
implementation of the columns-of-text `SPE` data format.

`SPE` format files contain energy-transfer (or time) histograms of intensity,
one per detector element in the instrument.
Information about the positions and sizes of the detector elements was contained in
a separate file, either `PAR` or `PHX`.
The format was defined with Direct-Geometry (DG) Time-of-Flight (TOF) spectrometers in
mind where a single incident energy applied to all data, and all detector elements
shared the same energy-transfer histogram bin edges. This information was kept
separately from the two text files, and put into analysis scripts by the user.

`NXspe` incorporates all information contained previously in `SPE` and `PAR` files
plus minimal information that otherwise may have gone into a log book.
It represents an appreciable improvement over the previous status-quo for DG-TOF.

For Indirect-Geometry (IG) TOF instruments with a single final neutron energy, like
back-scattering instruments, it is also possible to histogram all detector energy
spectra with a single set of bin edges. By means of a flag, `Horace` can exchange
_E_<sub>i</sub> and _E_<sub>f</sub> in early data loading calculations, and then
otherwise treat conversion more-or-less the same to produce intensity of (**_Q_**,_E_)
observations binned, not unlike `scipp`, in a coarse 4-D grid.

The problems start, however, when not all detectors share a single final energy,
as with BIFROST.
A single incident wavelength range applies for all BIFROST detector elements,
so one _E_<sub>i</sub> bin-boundary set would naturally apply for all detectors.
But this is _far_ from fixed _E_<sub>i</sub> or fixed _E_<sub>f</sub> with a single
energy transfer, _E_, bin-boundary set.

#### Imaginary observations
The closest BIFROST data can get to 'standard' `NXspe` _E_ binning is to stitch
together the single _E_<sub>i</sub> ranges offset by each _E_<sub>f</sub>.
This has three drawbacks
1. The number of bins per detector element will consequently be larger
2. The sizes of the bins may not match well to the 'natural' incident energy bin sizes for all detector elements
3. Imaginary observations will be created for all detector elements.

> **Note:**
> The term imaginary is used here to indicate that no measurement was performed
> for the incident neutron energies needed to produce the _E_ in one or more bin.

#### Natural _E_ binning
Instead, by keeping per-detector-element _E_ binning the adherence to
the `NXspe` standard can be maintained but software like `Horace` struggle to handle
such data &mdash;
the required `/entry/data/energy` becomes 2-D but `Horace` expects (and reshapes) this
data to be 1-D.

One symptom of this incompatibility is excessive memory usage.
For the 13500 BIFROST detector elements, with energy transfer binned into 103 bins,
in an early conversion step `Horace` requires a 13500 &times; 104 &times; 13500 array
of double values, and while &sim; 141 GB is not an insurmountable value it's also
not necessary.

In testing, moving to a machine with 256 GB of memory was not sufficient for the
conversion to proceed. And it seems likely there are more incompatible assumptions
made about the content of the data for `Horace` to successfully produce an `SQW` object
from the '`BIFROST`-`NXspe`' files.


## Decision
It is now possible to make `BIFROST` data conform to the `NXspe` standard published by
the NeXus Foundation, but community software that _reads_ `NXspe` is likely to have
made similar assumptions about the shape of required data as `Horace`.
So the utility of `NXspe` as a common data format is likely limited.

_Producing_ `NXspe` may still be useful, but it can not replace direct output of
`SQW` data files.

&therefore; Relegate `NXspe` to lesser-supported output formats.

## Consequences
A fully-fledged `SQW` output method is still required, which will still pose the same
problems of construction and maintainability.
