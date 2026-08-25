# Module 1 deterministic 10,000-molecule sample design

**Date:** 2026-08-24  
**Status:** Approved 2026-08-24
**Target:** `nvMolKit + Nemotron Notebook` Brev Launchable  
**Launchable ID:** `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`

## Goal

Make the Module 1 instruction below work when an attendee changes only
`SAMPLE_SIZE` and reruns Steps 1 through 3:

> Change `SAMPLE_SIZE` in Step 1 and rerun through this cell. How does the
> speed ratio change?

Every integer from `96` through `10_000` must be supported. The reference run
will remain `SAMPLE_SIZE = 96` so attendees can compare larger batch sizes.

## Root cause

Step 3 calculates similarity between three anchors and the selected library.
Its result has `3 * SAMPLE_SIZE` elements and does not require an all-pairs
matrix.

The failure occurs in Step 1. The bundled snapshot contains 96 records, and
`load_reframe` intentionally rejects a larger snapshot request. The notebook
therefore advertises an exercise that its local data source cannot perform.

Automatically changing to the live ReFRAME export is not acceptable. It would
make the attendee path depend on network access, make 110 concurrent instances
contact one external source, and expose the lesson to source and schema drift.
The public export inspected on 2026-08-24 used `ikey`, while the notebook
loader requires `canonical_ikey`.

## Scope

Add a source used only by Module 1, extend the shared loader with that explicit
source, update Module 1 instructions, and add tests. Update only the saved
target Launchable after the source passes review and acceptance.

Do not change:

- Modules 2 or 3 or their 96-row scientific contracts;
- the hosted Nemotron model or agent behavior;
- API-key storage, the setup renderer, or the saved setup body;
- any other Brev Launchable;
- Step 4's allocation safety limit.

## Data asset

Keep `notebooks/data/reframe_teaching_snapshot.csv` unchanged. Add
`notebooks/data/reframe_module1_snapshot_10k.csv` containing exactly 10,000
unique, RDKit-valid ReFRAME structures for Module 1 only.

The asset must:

1. Make its first 96 canonical identifiers match the sequence returned by the
   current Module 1 call, including its three anchor terms and
   `random_state=2026`.
2. Retain imatinib, linezolid, and ritonavir in the first three records.
3. Exclude all 96 existing identifiers, then add 9,904 records from a fixed
   ReFRAME export after normalizing `ikey` to `canonical_ikey`.
4. Remove records with missing SMILES or identifiers, duplicate identifiers,
   or SMILES that RDKit cannot parse.
5. Sort eligible added records by `canonical_ikey`, then take the first 9,904.
6. Convert spreadsheet-formula page links such as `=HYPERLINK(...)` to plain
   HTTPS URLs.
7. Record the source URL, retrieval date, raw export SHA-256, curation rules,
   final row count, final snapshot SHA-256, and approved redistribution basis
   in `notebooks/data/reframe_module1_snapshot_10k.provenance.json`.

The fixed source is the ReFRAME export retrieved on 2026-08-24 from
`https://reframedb.org/assets/csv/reframe_smiles_list.csv`, with raw SHA-256
`5a6043d65f7592f14cf3d879352a69089f3abac2b6c798dde5a283d05f90b522`.

The workshop organizer confirmed on 2026-08-24 that they hold written
permission to redistribute the 10,000-row ReFRAME snapshot through this public
GitHub repository and Brev Launchable. The permission record remains with the
organizer and must not be committed if it contains private information.

The default attendee path must read only the committed snapshot. It must make
no network request.

## Loader behavior

The existing `source="snapshot"` behavior and 96-row file remain unchanged for
Modules 2 and 3. Add `source="snapshot_10k"` for Module 1. That source will:

1. Require `sample_size` to be a built-in integer from `96` through `10_000`.
2. Reject values outside that range with a clear supported-range error.
3. Load and validate the committed 10,000-row snapshot.
4. Return the first `sample_size` records in their committed order.
5. Return exactly the requested number of records.
6. Fail with a data-integrity error if the committed snapshot is missing,
   malformed, duplicated, incomplete, or contains an invalid molecule.

This prefix selection makes samples nested: the 512-row sample contains the
96-row sample, the 2,048-row sample contains the 512-row sample, and the
10,000-row sample contains all smaller samples.

The existing snapshot, live, and explicitly configured CSV paths retain their
current semantics. They are not used by this exercise.

## Notebook behavior

Module 1 will keep:

```python
DATA_SOURCE = "snapshot_10k"
SAMPLE_SIZE = 96
```

The Step 1 and Step 3 instructions will state that attendees can try `512`,
`2_048`, and `10_000`, changing only `SAMPLE_SIZE` and rerunning Steps 1
through 3.

Step 2 will continue to fingerprint the full selected sample. Step 3 will
continue to compare the three anchors with that sample. Neither step will
materialize an `N x N` matrix.

Before Step 4, the notebook will tell attendees to restore
`SAMPLE_SIZE = 96`. Step 4's RDKit clustering constructs a quadratic condensed
distance vector and remains outside the 10,000-row exercise.

## Error handling

- `SAMPLE_SIZE < 96` or `SAMPLE_SIZE > 10_000` with `snapshot_10k`: report
  that the supported exercise range is 96 through 10,000.
- Corrupt bundled data: fail closed with a snapshot-integrity error.
- Missing anchors: fail the snapshot acceptance test before publication.
- Live-source failure: retain the current generic live-source error; do not
  fall back silently.

## Tests

### Data and loader tests

- The committed snapshot has exactly 10,000 rows.
- All required columns are present.
- SMILES and canonical identifiers are nonempty.
- Canonical identifiers are unique.
- RDKit parses every SMILES.
- All three anchors are present in the first 96 records.
- Requests for `96`, `97`, `512`, `2_048`, `9_999`, and `10_000` return
  exactly that many rows.
- Each tested smaller sample is a prefix of the next sample.
- Snapshot loading performs no network read.
- Requests for `95` and `10_001` fail with the supported-range error.
- Boolean, floating-point, string, and NumPy integer values fail the exact-type
  check.
- The existing 96-compound set remains unchanged.
- Existing `source="snapshot"` requests return the same ordered rows as before.
- Module 3's ordered 96-row pool and first-24 reference baseline remain
  unchanged.

### Notebook tests

- Module 1 uses `DATA_SOURCE = "snapshot_10k"` and keeps `SAMPLE_SIZE = 96`.
- The exercise names the supported values and rerun boundary.
- Steps 1 through 3 execute at `96`, `512`, `2_048`, and `10_000` from a clean
  kernel.
- Each Step 3 similarity result has shape `(3, SAMPLE_SIZE)`.
- Both RDKit and nvMolKit timing rows are produced on the target GPU.
- No Step 1 through Step 3 code allocates a dense `SAMPLE_SIZE x SAMPLE_SIZE`
  result.
- Step 4 retains its current memory guard and reset instruction.

### Target acceptance

Run the complete Steps 1 through 3 sequence on a fresh deployment of only
Launchable `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`. Verify the four representative
sample sizes `96`, `512`, `2_048`, and `10_000` in JupyterLab, and record
elapsed time plus peak host and GPU memory for the 10,000-row run. A browser
run must pass before the target saved Launchable is considered updated for
future deployments.

## Publication boundary

Implementation, tests, and a fresh target deployment must pass before the
reviewed source commit is selected for future instances of the target
Launchable. Updating the source repository alone does not prove that the saved
Launchable works. No other saved Launchable may be edited or redeployed.
