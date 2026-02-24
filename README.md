# ESS Data Reduction Monorepo

Monorepo for ESS neutron scattering data reduction packages, managed with [pixi](https://pixi.sh/).

| Package | Description |
|---------|-------------|
| [essreduce](packages/essreduce/) | Common data reduction tools (core) |
| [essimaging](packages/essimaging/) | Neutron imaging (ODIN, TBL, YMIR) |

## Dependency graph

```
essreduce
└── essimaging
```

---

## Developer Guide

### Prerequisites

Install [pixi](https://pixi.sh/):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Setup

```bash
git clone git@github.com:scipp/ess.git
cd ess

# Install all packages (editable, with test deps):
pixi install

# Or just one package:
pixi install -e essreduce
```

The `pixi.lock` file pins all dependencies reproducibly. No tox, no pip-compile, no manual virtualenv.

### Running tests

```bash
# Test a package:
pixi run test essreduce
pixi run test essimaging

# Test a single file:
pixi run -e essreduce pytest packages/essreduce/tests/normalization_test.py
```

### Linting and formatting

```bash
pixi run -e lint lint       # all pre-commit hooks
pixi run -e lint check      # ruff check
pixi run -e lint format     # ruff format
```

### Building docs

```bash
pixi run docs essreduce
pixi run docs essimaging
```

### Adding or changing dependencies

Edit the package's `pyproject.toml`, then re-lock:

```bash
pixi install
```

Commit the updated `pixi.lock`.

### Releasing a package

Push a tag with the package prefix:

```bash
git tag essreduce/26.3.0
git push origin main --tags
```

The `release.yml` workflow builds, publishes to PyPI, and deploys docs.

### How CI works

- **On PRs:** Only changed packages are tested (changing `essreduce` also tests `essimaging`)
- **Nightly:** All packages tested against latest deps + lower-bound deps
- **Weekly:** All packages tested on macOS and Windows

### Repo structure

```
pixi.toml                  ← workspace root (features, tasks, environments)
pixi.lock                  ← single lockfile for all packages
.pre-commit-config.yaml    ← shared linting hooks
packages/
  essreduce/
    pyproject.toml          ← package deps, version, pytest config
    src/ess/reduce/         ← source code (ess.reduce namespace)
    tests/
    docs/
  essimaging/
    pyproject.toml
    src/ess/imaging/        ← source code (ess.imaging namespace)
    ...
```
