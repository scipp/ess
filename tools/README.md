# ESS Project Tools

This folder contains various scripts for managing the ess project.

## Python Scripts

Python scripts are written as standalone scripts with PEP 723 metadata.
See, e.g., https://pixi.prefix.dev/latest/python/scripts

The scripts can be run using
```sh
pixi run --script tools/<scriptname.py>
```

Each script has a pixi lockfile to pin dependencies.
We should rarely need to update dependencies and by pinning them, we are less susceptible to breaking changes and security problems from compomized packages.

## Shell Scripts

Shell scripts in this folder generally assume a POSIX shell and often specifically require bash.
