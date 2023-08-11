# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import argparse
import os
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser(description='Build doc pages with sphinx')
parser.add_argument('--build-dir', default=None)
parser.add_argument('--work-dir', default=None)
parser.add_argument('--builder', default='html')

if __name__ == '__main__':
    args = parser.parse_args()

    docs_dir = pathlib.Path(__file__).parent.absolute()
    work_dir = (
        args.work_dir
        if args.work_dir is not None
        else os.path.join(docs_dir, '.doctrees')
    )
    build_dir = (
        args.build_dir
        if args.build_dir is not None
        else os.path.join(docs_dir, 'build')
    )

    # Build the docs with sphinx-build
    subprocess.check_call(
        ['sphinx-build', '-v', '-b', args.builder, '-d', work_dir, docs_dir, build_dir],
        stderr=subprocess.STDOUT,
        shell=sys.platform == "win32",
    )

    # Remove Jupyter notebooks used for documentation build,
    # they are not accessible and create size bloat.
    # However, keep the ones in the `_sources` folder,
    # as the download buttons links to them.
    sources_dir = os.path.join(build_dir, '_sources')
    for path in pathlib.Path(build_dir).rglob('*.ipynb'):
        if not str(path).startswith(sources_dir):
            os.remove(path)
