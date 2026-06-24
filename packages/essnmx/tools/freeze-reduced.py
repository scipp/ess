"""Reduced file freezing script.

Example:
```
pixi run --frozen python packages/essnmx/tools/freeze-reduced.py \
        --freeze-version 26.6.0 \
        --overwrite

# This command will generate essnmx-reduced-26.6.0.hdf
# and auxiliary outputs in the essnmx-reduced-26.6.0 directory.
```

Please note that the `freeze-version` is a hard-coded version
and it does not automatically use specific version of essnmx.
It is because it is meant to freeze a reduced file for a future release.
"""

import argparse

import ess.nmx.data as nmxdat
from ess.nmx.configurations import (
    InputConfig,
    OutputConfig,
    ReductionConfig,
)
from ess.nmx.executables import reduction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a reduced file from "
        "a small mcstas simulation result, injected into a nexus template. "
        "The output file, essnmx-reduced-{version}.hdf, "
        "should be published and be registered "
        "in the `ess.nmx.data` module. "
        "The registered file will be used for the executable-output-tests "
        "that compare new reduction result to the frozen file. "
        "When there is a breaking change, i.e. new version produces different result, "
        "it should be documented in the developers guide, "
        "under regression test dataset section. "
        "The notebook file path: packages/essnmx/docs/developer/test-dataset.ipynb."
    )
    parser.add_argument(
        "--freeze-version",
        help="Future version of essnmx that can reproduce the frozen file.",
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        help="Overwrite the existing output file.",
        action='store_true',
    )
    args = parser.parse_args()
    freezing_version = args.freeze_version
    raw_nexus_file = nmxdat.get_small_nmx_nexus().as_posix()

    input_config = InputConfig(input_file=[raw_nexus_file])
    output_config = OutputConfig(
        output_file=f'essnmx-reduced-{freezing_version}.hdf', overwrite=args.overwrite
    )
    reduction_config = ReductionConfig(inputs=input_config, output=output_config)
    reduction(config=reduction_config)
