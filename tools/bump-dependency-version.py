import argparse
import glob
import logging
import pathlib
import re

import tomllib

"""Minimum dependency updating script.

This script updates minimum dependency to the given version
if any matching package depends on it(dependency-name).

Example 1 - update minimum version of plopp to 26.5.0 for all packages:
```
pixi run --frozen python tools/bump-dependency-version.py \
  --package ess* \
  --dependency-name plopp \
  --version 26.5.0
```

Example 2 - update minimum version of pytest to 8.0 for all packages:
```
pixi run --frozen python tools/bump-dependency-version.py \
  --package ess* \
  --dependency-name pytest \
  --extra test \
  --version 8.0
```

Example 3 - update minimum version of ipywidgets to 30.0 for essnmx and essdiffraction:
```
pixi run --frozen python tools/bump-dependency-version.py \
  --package essnmx \
  --package essdiffraction \
  --dependency-name ipywidgets \
  --extra docs \
  --version 30.0
```

"""


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger(__file__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        help="Name of the package to bump the dependency version. "
        "Glob expression such as ess* or "
        "multiple packages can be specified.",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--dependency-name",
        help="Name of the dependency to bump the minimum version.",
        required=True,
    )
    parser.add_argument(
        "--version", help="New minimum version of the dependency.", required=True
    )
    parser.add_argument(
        "--extra",
        help="Extra dependency group that the dependency belongs to.",
        default=None,
        required=False,
    )
    return parser.parse_args()


def _collect_packages(package_patterns: list[str]) -> list[pathlib.Path]:
    packages = []
    for pckg_name in package_patterns:
        packages.extend(
            glob.glob(pathname=pckg_name, root_dir="packages", recursive=False)
        )
    return packages


def _parse_and_filter_package_deps(
    *,
    packages: list[pathlib.Path],
    package_root: pathlib.Path,
    dependency_regex: re.Pattern,
    final_dependency_version: str,
    extra: str | None = None,
    logger: logging.Logger,
) -> dict[pathlib.Path, str]:
    def _cur_dep(package: pathlib.Path) -> str | None:
        toml_file_path = package_root / package / "pyproject.toml"
        package_toml_file: dict[str, dict] = tomllib.loads(toml_file_path.read_text())
        dependencies: list[str] = (
            package_toml_file["project"]["dependencies"]
            if extra is None
            else package_toml_file["project"]["optional-dependencies"][extra]
        )
        matching_dependencies = [
            dep for dep in dependencies if dependency_regex.match(dep)
        ]
        cur_dep = None
        if (num_matched := len(matching_dependencies)) == 0:
            logger.info("No matching dependency to update for package %s...", package)
        elif num_matched == 1:
            cur_dep = matching_dependencies[0]
            logger.info(
                "Found the dependency to update for package %s. Current version: %s",
                package,
                cur_dep,
            )
        else:
            logger.info(
                "Found more than one matching pattern. "
                "Cannot update the dependency automatically for package %s...",
                package,
            )
        if cur_dep == final_dependency_version:
            logger.info("Dependency already up to date for package %s...", package)
        return cur_dep

    return {
        pckg: cur_dep for pckg in packages if (cur_dep := _cur_dep(pckg)) is not None
    }


def _bump_up_dependency(
    package_deps_map: dict[pathlib.Path, str],
    package_root: pathlib.Path,
    logger: logging.Logger,
) -> None:
    for pckg_name, cur_dep in package_deps_map.items():
        toml_file_path = package_root / pckg_name / "pyproject.toml"
        updated_toml = toml_file_path.read_text().replace(
            cur_dep, final_dependency_version
        )
        toml_file_path.write_text(updated_toml)
        logger.info(
            "Minimum dependency version updated to: %s for package %s",
            final_dependency_version,
            pckg_name,
        )


if __name__ == "__main__":
    logger = _setup_logger()
    args = _parse_arguments()
    package_root = pathlib.Path("packages")

    logger.info("Bump dependency version using these arguments: %s", args)

    packages = _collect_packages(args.package)
    logger.info("Matching package names: %s", packages)

    dependency_regex = re.compile(pattern=f"{args.dependency_name}>=[0-9]+")
    final_dependency_version = f"{args.dependency_name}>={args.version}"

    filtered_packages = _parse_and_filter_package_deps(
        packages=packages,
        package_root=package_root,
        dependency_regex=dependency_regex,
        final_dependency_version=final_dependency_version,
        extra=args.extra,
        logger=logger,
    )
    _bump_up_dependency(
        package_deps_map=filtered_packages,
        package_root=package_root,
        logger=logger,
    )
