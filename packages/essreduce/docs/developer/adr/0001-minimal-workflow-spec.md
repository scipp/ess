# ADR 0001: Minimal implementation-independent workflow specifications

- Status: proposed
- Deciders: Simon
- Date: 2026-09-14

## Context

Three mechanisms currently describe workflow interfaces to users, with
overlapping purpose and no shared shape:

- `ess.reduce.parameter` / `ess.reduce.workflow`: per-sciline-key `Parameter`
  dataclasses in a global registry, with parameters discovered by walking the
  pipeline graph from selected outputs. Drives the ipywidgets GUI.
- `ess.livedata.config.workflow_spec.WorkflowSpec`: one pydantic params model
  and one pydantic outputs model per workflow, driving the live-data dashboard.
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
a web dashboard, a command-line tool, or a GUI application — be generated from a workflow
description alone. Compute is deliberately abstracted away: the same spec must
make sense whether the workflow runs as a local sciline pipeline, behind a web
service, or as a cluster job. Compute is not part of this work, but it shapes
the design: nothing implementation-bound may appear in the spec.

## Decision

A new module `ess.reduce.spec` defines the spec layer. Its only dependency
beyond the standard library is pydantic (a new essreduce dependency); the
scipp-facing pieces are quarantined in one submodule.

### The spec is pure interface: no factory, no keys, no registry

`WorkflowSpec` holds identity (`name`, `version`), display metadata (`title`,
`description`, both mandatory), a params model, an outputs model, and an
optional `code_revision`. Nothing else. In particular it holds *no* workflow
factory and *no* sciline keys: a spec describes *what a user can configure and
what they get back*, not how it is computed. Binding a spec to an executor —
conceptually a mapping from spec identity to
`Callable[[BaseModel], BaseModel]`, or a remote service holding the same
spec — is a parallel mechanism, intentionally undefined here. This is what
keeps the spec valid across local, service, and cluster execution.

One requirement follows for workflow packages: the module that defines a spec
must be importable without importing the workflow code. A service then loads
and validates every spec it knows without importing sciline pipelines or
instrument code, and the process that binds a spec to code is the only one
that pays for that import. esslivedata already separates the two for this
reason.

How specs are enumerated (module-level tuples, entry points, esslivedata's
per-instrument registration) is out of scope. Any mechanism works against the
same spec type; prescribing one here would recreate the catalog problem that
sank the `ess.schemas` plan. Entry points split by role, specs in one group
and factories in another under the same name, are the natural fit for the
requirement above and are expected to become the convention once the first
service adopts this spec.

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

### Outputs are a typed model in the same vocabulary

Outputs are likewise a pydantic model class (`outputs: type[BaseModel]`,
mandatory). Field title and description are the display metadata; a field may
be optional when the workflow does not always produce it; declaration order is
meaningful (consumers show outputs in order, primary output first). Array and
file outputs are data fields (next section); small values such as a beam
centre or a fitted scale factor are `Quantity`, a scalar or short vector with a
unit, as plain data.

An earlier form of this decision declared outputs as a dictionary of
structural descriptions, with arrays typed by `ArraySpec` and everything else
untyped. That broke "outputs can be inputs" for exactly the values that most
often feed the next workflow. With both sides as models over one vocabulary,
chaining is a type check between an output field and a parameter field, and
where a framework stores an output (inline in a record, or in a data store) is
decided by the field's type and is not a spec concept. Output *selection*
(choosing which sciline targets to compute) is still not modeled: like
parameter slicing, it is an implementation notion. Livedata-specific output
machinery (`OutputView`, `Temporality`, windowing) stays in esslivedata.

### Data fields: inputs are parameters

There is no separate input section. A parameter or output that holds data
rather than a literal is a **data field**: a field of type `Ref`, annotated
with the `Format` of the bytes (raw NeXus file, scipp object, opaque file) and,
for scipp data, an `ArraySpec` describing dims, unit, coordinate units, and
whether the data is binned. The `binned` flag tells consumers which outputs are
event data that must not be plotted directly. A scalar with a unit is the 0-d
case.

A reference is plain data naming data that exists elsewhere: an output of an
earlier run (`OutputRef`: record, output name, optionally one element of a
collection by key), or a dataset the framework did not compute (`DatasetRef`:
an identity string whose meaning, a catalogue PID or a local file's identity,
belongs to the framework). A field may be a union of a literal and a reference,
for values a user may type in or take from a previous run. Collections,
`list[...]` and `dict[str, ...]` of one declared type, are allowed on both
sides, and a reference may name one element of a collection output.

The spec says nothing about how a workflow gets at the bytes. Whether a
reference becomes a local path or an in-memory object is decided where the
workflow is called, by the executor binding that the ADR leaves out of scope,
and the workflow asks there for the form it wants: a path for a NeXus file it
loads by component, an object for a curve it fits. A framework that adds an
in-memory fast path for chained runs therefore changes no spec and no workflow
interface. An earlier form of this decision typed a data field as a union of the
reference and the materialized value, a path or a scipp object, so that one
model served both the request and the call. That made the model wrong in both
phases, needed a validator that accepted anything not plain data, hid that
member from the JSON Schema, and put a materialization instruction into what
was meant to be pure interface.

The format serves the consumers of the spec: a framework compares the
producer's output annotation with the consumer's parameter annotation before
chaining, a picker lists candidates of matching format, a UI selects a plotter
from the `ArraySpec`. A dataset's format is not checked at submission; a dataset
that is not what the field declares fails when the workflow reads it. Every
difference between an input and a parameter — resolution, provenance, which
widget a UI shows — is behaviour a framework selects by the field's type; the
spec only declares the type. Helpers find the data fields of a model and the
references in a plain request value, so a framework never re-derives the
annotation's meaning. The structural check of a scipp object against its
`ArraySpec` needs scipp and lives in `ess.reduce.spec.conversions`, called by
whoever runs the workflow on the outputs it returns.

### Two forms, one-way projection

`WorkflowSpec` is the in-process form: it holds the params and outputs model
*classes*, so same-process consumers get full pydantic validation including
custom validators. `spec.serialize()` projects onto `SerializedWorkflowSpec`,
a plain-data pydantic model with both models as JSON Schema
(`model_json_schema()`), which round-trips through JSON and is what a service
announces to remote consumers. Data fields appear in the schema under a
`dataField` key with their format and array structure; the schema is the
entire cross-process surface, sufficient to
render a form, offer a picker for data fields, and select a plotter for an
output.

There is deliberately no inverse. Validators do not survive JSON Schema, so a
deserialized spec would be a lie about its own validation. Instead, validation
authority sits with the process owning the model classes: in-process UIs
validate directly; remote UIs validate optimistically against the schema and
the owning service accepts or rejects authoritatively. This matches the
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

`code_revision` is provenance, not identity: an optional git commit or package
version of the code the spec describes, so that a record made from a
development branch is honest about what ran. The interface version stays
`version`.

### Shared parameter vocabulary, scipp-free

`ess.reduce.spec.parameters` provides constrained unit enums, range/edges
models with cross-field validation (`stop > start`, log-scale positivity) —
the models previously duplicated between esslivedata and package-specific
code — and `Quantity`. They contain no scipp: conversion of validated values
into scipp objects (`edges_to_variable`, `range_to_variables`) lives in
`ess.reduce.spec.conversions`, imported by workflow implementations only. This
keeps the vocabulary JSON-Schema-clean and the spec layer importable without
touching scipp. Value defaults (start/stop/bin counts) are set by workflow
authors at the use site, not by the vocabulary — sensible values are a
workflow/instrument decision, and a generic default is a wrong default.

### Convergence with esslivedata

Explicit goal: `ess.livedata.config.workflow_spec.WorkflowSpec` eventually
inherits from this spec, adding its live-data fields (`instrument`, `group`,
`source_names`, `aux_sources`, reset flags). The base spec's field names and
semantics (`name`, `version`, `title`, `description`, `params`, `outputs`) are
a strict subset of esslivedata's today for exactly this reason, and
esslivedata already declares outputs as a model class with title and
description as field metadata. The remaining difference is field types:
esslivedata's outputs are `sc.DataArray` fields with default-factory templates
used for plotter selection; here they are data fields constrained by
`ArraySpec`, which serializes. The migration (already planned independently in
scipp/esslivedata#889) changes field types only; esslivedata's `Temporality`
annotation coexists with the data-field annotation in the same `Annotated`.
The import edge is free — the esslivedata backend already depends on
essreduce, and its dashboard is decoupled via the serialized-spec announcement,
not via imports.

## Consequences

- Generic UIs (including a command-line interface) can be generated from
  `WorkflowSpec` alone, and from `SerializedWorkflowSpec` across process
  boundaries, with no knowledge of the workflow implementation.
- A framework that chains workflows validates a reference by looking up the
  producer's output field and comparing its data-field annotation with the
  consumer's; how strict that comparison is (format only, or full `ArraySpec`
  compatibility) is the framework's rule.
- A workflow receives references and resolves them through whatever runs it;
  the contract for that resolution belongs to the executor binding, not to the
  spec. A runner calls `check_array` on array outputs at completion.
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
  beyond being addressable by `(name, version)` and importable without
  workflow code.

Foreseen extensions, each one optional spec field or one field-level
annotation, deliberately not added until a consumer exists: declared failure
reasons, an intermediate flag for retention, and declared keys of a collection
output.
