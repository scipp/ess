# Adding a new package

To add a package (e.g. `essdiffraction` from `scipp/essdiffraction`):

## 1. Clone, rewrite history, and merge

```bash
git clone git@github.com:scipp/essdiffraction.git /tmp/essdiffraction
cd /tmp/essdiffraction
git filter-repo --to-subdirectory-filter packages/essdiffraction --tag-rename ':essdiffraction/'
cd /path/to/ess
git remote add essdiffraction /tmp/essdiffraction
git fetch essdiffraction --tags
git merge essdiffraction/main --allow-unrelated-histories
git remote remove essdiffraction
```

## 2. Update `packages/essdiffraction/pyproject.toml`

- Set `license-files = ["../../LICENSE"]`
- Add setuptools-scm monorepo config:
  ```toml
  [tool.setuptools_scm]
  root = "../.."
  tag_regex = "^essdiffraction/(?P<version>[vV]?\\d+(?:\\.\\d+)*(?:[._-]?\\w+)*)$"
  git_describe_command = ["git", "describe", "--dirty", "--tags", "--long", "--match", "essdiffraction/*[0-9]*"]
  ```
- Add `docs` extra with deps from `requirements/docs.in` (minus base deps)
- Delete `tox.ini`, `requirements/`, `.github/`, `.pre-commit-config.yaml`

## 3. Add to `pixi.toml`

Package feature:

```toml
[feature.essdiffraction.pypi-dependencies]
essdiffraction = { path = "packages/essdiffraction", editable = true, extras = ["test"] }
```

Docs feature:

```toml
[feature.docs-essdiffraction.pypi-dependencies]
essdiffraction = { path = "packages/essdiffraction", editable = true, extras = ["test", "docs"] }
```

Environments (include features for workspace dependencies, e.g. `essreduce`):

```toml
essdiffraction = { features = ["essdiffraction", "essreduce"], solve-group = "default" }
lb-essdiffraction = { features = ["essdiffraction", "essreduce"], solve-group = "lower-bound" }
docs-essdiffraction = { features = ["docs-essdiffraction", "docs-essreduce", "docs"], solve-group = "default" }
```

Add the new feature to the `default` environment.

## 4. Add to `.github/workflows/ci.yml`

Add a change filter and matrix entry. If the package depends on `essreduce`, include essreduce paths in the filter.

## 5. Run `pixi install` and commit
