import argparse

import ess.nmx.data as nmxdat
from ess.nmx.configurations import (
    InputConfig,
    OutputConfig,
    ReductionConfig,
)
from ess.nmx.executables import reduction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
