# Develoment

Build the local documentation prototype from the repository root with:

```sh
pixi run -e docs-essdiffraction python -m sphinx -W --keep-going -b html \
-d ~/mlzdiffraction-doctrees packages/mlzdiffraction/docs ~/mlzdiffraction-html
```

Open the built homepage at:

```text
~/mlzdiffraction-html/index.html
```

This uses `docs-essdiffraction` temporarily because `mlzdiffraction` is not yet
registered as its own pixi package or docs environment.

The `packages/mlzdiffraction/docs/generated/` directory is produced by `autosummary` during builds and is ignored by git.
