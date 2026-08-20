# ADR 0001: Minimal implementation-independent workflow specifications

- Status: proposed
- Deciders: Simon
- Date: 2026-08-05

## Context

Three mechanisms currently describe workflow interfaces to users, with
overlapping purpose and no shared shape:

- `ess.reduce.parameter` / `ess.reduce.workflow`: per-sciline-key `Parameter`
  dataclasses in a global registry, with parameters discovered by walking the
  pipeline graph from selected outputs. Drives the ipywidgets GUI.
- `ess.livedata.config.workflow_spec.WorkflowSpec`: one pydantic params model
  per workflow, plus output descriptions, driving the live-data dashboard.
- `ess.nmx.configurations`: standalone pydantic models for batch reduction.

Consolidation was analyzed at length in
[scipp/ess#653](https://github.com/scipp/ess/issues/653) and
[scipp/esslivedata#889](https://github.com/scipp/esslivedata/issues/889). Two
earlier attempts stalled, both for the same reason: scope. A universal
`ess.schemas` catalog ("all workflows for all instruments, imported by
everyone") turned shared conventions into a cross-team release-coordination
problem; a rewrite of the essreduce widget layer
([scipp/ess#689](https://github.com/scipp/ess/pull/689)) kept sciline keys,
workflow factories, and widget concerns inside the spec, so the spec could not
outlive or precede any particular implementation.

The goal is the minimal layer that lets a *generic* user interface — ipywidgets,
a web dashboard, or a command-line tool — be generated from a workflow
description alone. Compute is deliberately abstracted away: the same spec must
make sense whether the workflow runs as a local sciline pipeline, behind a web
service, or as a cluster job. Compute is not part of this work, but it shapes
the design: nothing implementation-bound may appear in the spec.

## Decision

A new module `ess.reduce.spec` defines the spec layer. Its only dependency
beyond the standard library is pydantic (a new essreduce dependency); the one
scipp-facing piece is quarantined in a submodule.

### The spec is pure interface: no factory, no keys, no registry

`WorkflowSpec` holds identity (`name`, `version`), display metadata (`title`,
`description`, both mandatory), a params model, and output descriptions.
Nothing else. In particular it holds *no* workflow factory and *no* sciline
keys: a spec describes *what a user can configure and what they get back*, not
how it is computed. Binding a spec to an executor — conceptually a mapping from
spec identity to `Callable[[BaseModel], Mapping[str, Any]]`, or a remote
service holding the same spec — is a parallel mechanism, intentionally
undefined here. This is what keeps the spec valid across local, service, and
cluster execution.

How specs are enumerated (module-level tuples, entry points, esslivedata's
per-instrument registration) is likewise out of scope. Any mechanism works
against the same spec type; prescribing one here would recreate the catalog
problem that sank the `ess.schemas` plan.

### One params model per workflow

Parameters are a single pydantic model class per workflow
(`params: type[BaseModel]`), not per-key entries in a registry. This enables
cross-parameter validation, gives JSON Schema for free, and removes the
implementation coupling of key-addressed parameters. The graph-derived
"select outputs, then see only relevant parameters" feature of
`ess.reduce.workflow.get_parameters` does not survive: it treats output
selection as workflow slicing, which only the sciline implementation can
express. If output-dependent parameter sets are needed, they are distinct
workflows (distinct specs).

The field defaults to `NoParams` (a closed model with no fields), so consumers
never branch on params being absent, and sending parameters to a workflow that
takes none is a validation error rather than silently ignored.

### Two forms, one-way projection

`WorkflowSpec` is the in-process form: it holds the params model *class*, so
same-process consumers (ipywidgets, a CLI wrapping a local pipeline) get full
pydantic validation including custom validators. `spec.serialize()` projects
onto `SerializedWorkflowSpec`, a plain-data pydantic model with params as JSON
Schema (`model_json_schema()`), which round-trips through JSON and is what a
service announces to remote consumers.

There is deliberately no inverse. Validators do not survive JSON Schema, so a
deserialized spec would be a lie about its own validation. Instead, validation
authority sits with the process owning the model class: in-process UIs validate
directly; remote UIs validate optimistically against the schema and the owning
service accepts or rejects authoritatively. This matches the
announcement-as-contract design adopted for esslivedata in
[scipp/esslivedata#889](https://github.com/scipp/esslivedata/issues/889): the
serialized spec is the entire cross-process surface, and where a model class is
*defined* is invisible to consumers.

### Identity is `name` + `version`; scoping is the enumerator's problem

No `instrument` field and no `WorkflowId` class at this level. Instrument is
meaningless for technique-level batch workflows, and a spec cannot guarantee
global uniqueness of anything — only the context that enumerates or deploys
specs can. esslivedata keeps keying workflows by `(instrument, name, version)`,
supplying the instrument from its registration context. Data-provenance
identity (which spec, params, and input datasets produced a dataset) similarly
composes spec identity with deployment context; the spec's contribution is
being serializable and versioned.

### Outputs are declared, structurally, without scipp

`outputs` maps output names to `OutputSpec` (mandatory title, description,
optional `ArraySpec`). `ArraySpec` describes dims, unit, and coordinate units —
plain data, so it serializes, replacing the `sc.DataArray` default-factory
templates esslivedata currently uses for plotter selection. Output *selection*
(choosing which sciline targets to compute) is not modeled: like parameter
slicing, it is an implementation notion. Declaration order is meaningful
(consumers show outputs in order, primary output first). Livedata-specific
output machinery (`OutputView`, `Temporality`, windowing) stays in esslivedata.

### Shared parameter vocabulary, scipp-free

`ess.reduce.spec.parameters` provides constrained unit enums and range/edges
models with cross-field validation (`stop > start`, log-scale positivity) —
the models previously duplicated between esslivedata and package-specific
code. They contain no scipp: conversion of validated values into scipp objects
(`edges_to_variable`, `range_to_variables`) lives in
`ess.reduce.spec.conversions`, imported by workflow implementations only. This
keeps the vocabulary JSON-Schema-clean and the spec layer importable without
touching scipp. Value defaults (start/stop/bin counts) are set by workflow
authors at the use site, not by the vocabulary — sensible values are a
workflow/instrument decision, and a generic default is a wrong default.

### Convergence with esslivedata

Explicit goal: `ess.livedata.config.workflow_spec.WorkflowSpec` eventually
inherits from this spec, adding its live-data fields (`instrument`, `group`,
`source_names`, `aux_sources`, `device_outputs`, reset flags). The base spec's
field names and semantics (`name`, `version`, `title`, `description`,
`params`) are a strict subset of esslivedata's today for exactly this reason.
The blocking difference is `outputs`: esslivedata's `sc.DataArray` templates
must first migrate to `ArraySpec` (already planned independently in
scipp/esslivedata#889). The import edge is free — the esslivedata backend
already depends on essreduce, and its dashboard is decoupled via the
serialized-spec announcement, not via imports.

## Consequences

- Generic UIs (including a command-line interface) can be generated from
  `WorkflowSpec` alone, and from `SerializedWorkflowSpec` across process
  boundaries, with no knowledge of the workflow implementation.
- essreduce gains a pydantic dependency.
- `ess.reduce.parameter`, `ess.reduce.workflow`, and the widgets built on them
  are superseded and will be removed in a later hard break; they are untouched
  for now. The graph-derived parameter discovery they provide is dropped, not
  ported.
- `ess.nmx.configurations` and esslivedata migrate to the shared vocabulary
  and spec incrementally, per package, with no coordination requirement — a
  package that never migrates costs the others nothing.
- The executor binding and spec enumeration remain to be designed when a
  concrete consumer needs them; the spec layer does not constrain either
  beyond being addressable by `(name, version)`.
