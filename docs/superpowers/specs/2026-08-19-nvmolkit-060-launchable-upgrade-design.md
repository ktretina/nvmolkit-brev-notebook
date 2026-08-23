# nvMolKit 0.6.0 Notebook Launchable Upgrade Design

## Decision

Upgrade only the **nvMolKit + Nemotron Notebook** Launchable to an exact
`nvmolkit==0.6.0` runtime. Integrate the valid technical intent from GitHub PR
`#1` and PR `#2` through a coordinated repository change instead of merging
either PR as written.

The conversational ACS/OpenClaw Launchable is outside this design and must not
change.

## Target user

The target user is a chemistry workshop participant who can run guided Python
notebooks but should not need to debug GPU library compatibility, interpret
internal result containers, or manage Brev infrastructure. The participant
needs one reproducible environment, clear CPU/GPU provenance, bounded choices,
and truthful output that remains useful when a small benchmark does not favor
the GPU path.

## Goals

1. Make all four notebooks executable with the exact nvMolKit 0.6.0 release.
2. Normalize the nvMolKit 0.5 and 0.6 fused-Butina result shapes behind one
   repository-owned adapter.
3. Preserve exact cluster-assignment, centroid, receipt, and artifact
   validation at every consumer.
4. Remove the false 10,000-by-10,000 matrix estimate from the fused-Butina
   advanced path without implying that GPU out-of-memory errors are impossible.
5. Keep runtime comparison language neutral and retain CPU/GPU provenance.
6. Requalify the complete notebook experience on the target L4 environment
   before treating the Launchable as updated.
7. Preserve Kevin Boyd's contribution credit and close both PRs with precise
   integration notes after the replacement change reaches `main`.

## Non-goals

- Do not change the Nemotron model, API-key field, Jupyter port, access policy,
  GPU requirement, disk size, or notebook order.
- Do not add an automatic dependency range or follow an unreleased nvMolKit
  0.6.1 contract.
- Do not claim universal speedup, memory safety, biological activity, or
  production reliability.
- Do not edit or deploy the conversational Launchable.
- Do not use a private Brev Console API.

## Source architecture

### Exact dependency

`requirements.txt` pins `nvmolkit==0.6.0`. GPU acceptance checks the same exact
version. The exact pin contains the open upstream API-contract risk and avoids
an unreviewed 0.6.1 behavior change.

### Shared fused-Butina adapter

Add `notebooks/nvmolkit_compat.py`. It returns one normalized result tuple with:

- one integer label for every input molecule;
- member indices for every cluster;
- one centroid index for every cluster.

The adapter accepts either supported result shape:

- nvMolKit 0.5: `(member_lists, cumulative_sizes, centroids)`;
- nvMolKit 0.6: `(cluster_ids, centroids)`, where returned objects may expose
  `.numpy()`.

It converts data to host NumPy arrays once and validates all invariants:

- molecule count and label-vector length agree;
- labels are integer, non-negative, and contiguous after normalization;
- clusters form an exact partition with no missing, duplicate, or out-of-range
  molecule index;
- centroid count equals cluster count;
- every centroid belongs to its corresponding cluster.

The adapter is shape-based rather than version-branching. This avoids an
implicit `packaging` dependency and keeps malformed output fail-closed.

Module 3 executes controller-owned analysis through isolated Python. The
controller therefore embeds the exact source of the same self-contained
normalizer into the validated analysis program. It does not weaken `-I`, add a
mutable `PYTHONPATH`, or permit an attendee-selected import.

### Consumers

Use the adapter in every fused-Butina execution path:

1. Module 1 direct nvMolKit notebook;
2. Module 3 controller-rendered analysis in `notebooks/workshop_llm_agent.py`;
3. the companion workflow in `chemistry_workflow.py`;
4. command receipts and their exact-source tests;
5. GPU acceptance tests.

Module 2 does not call fused Butina, so its computation and text remain
unchanged. If the shared workshop-agent version changes, only its exact helper
version lock changes. It still runs during full notebook qualification.

### Scientific behavior

nvMolKit 0.6 changes cutoff boundary behavior, implementation, tie-breaking,
and cluster ordering. Tests must compare partitions in a label-invariant way
unless a public artifact explicitly promises an order. Live qualification must
recheck assignment completeness, centroid membership, representative choice,
Module 3 panel results, and companion-demo evidence.

## Module 1 messaging and advanced path

Remove only the hypothetical square-matrix estimate and 512 MiB check from the
advanced fused-Butina path. Keep the valid RDKit condensed-distance guard.

Explain that fused Butina avoids an `N x N` pairwise matrix and uses packed
fingerprints plus linear working buffers. State that the full notebook and CUDA
allocator still consume GPU memory, so the change is not a no-OOM guarantee.

Use neutral comparison output:

`Observed runtime ratio (RDKit CPU / nvMolKit GPU; >1 favors nvMolKit): ...`

For clustering, identify the compared work:

- `RDKit CPU: condensed distance + Butina`;
- `nvMolKit GPU: fused fingerprint clustering`.

Do not print `speedup` when a measured ratio may be below one.

## Tests and qualification

### Test-first local work

Before production edits, add focused failures for:

- both nvMolKit result shapes;
- malformed shapes, incomplete assignments, noncontiguous labels, and invalid
  centroids;
- all three consumers using the shared adapter;
- the exact 0.6.0 dependency and GPU gate;
- neutral ratio and hardware wording;
- removal of only the incorrect fused-memory guard;
- accurate fused-memory explanation;
- clean notebook JSON and no saved output/widget state.

After each focused GREEN, run adjacent tests. Finish with the complete serial
test suite using `MPLBACKEND=Agg`, Ruff checks for changed Python, notebook code
compilation, Bash syntax, `git diff --check`, and a secret scan.

### Live L4 qualification

Use only the target Notebook Launchable and one fresh environment. Record the
exact source commit and verify a clean checkout, CPython 3.12, one L4, CUDA,
nvMolKit 0.6.0, protected-key permissions, and Jupyter health.

Then run:

1. `RUN_GPU_TESTS=1` acceptance;
2. Module 1 GPU path and its bounded advanced 10,000-row path;
3. Module 2 reference and one hosted policy call;
4. Module 3 plan, explicit approval, one analysis attempt, audit state,
   canonical receipt, gallery, and replay with zero new calls or executions;
5. the full companion plan, six approvals, objective, conclusion, receipts,
   and figures;
6. an organization-only browser check through the port-8888 Secure Link.

Do not call a local or static check live proof.

## GitHub integration

Implement the replacement on the existing isolated branch. The final source
commit records Kevin Boyd as co-author. After independent specification and
quality reviews, push the accepted source to `main`. Add a concise comment to
each PR that identifies the integrated commit, explains the corrections, and
close the PR as superseded. Do not merge either flawed patch merely to obtain
credit.

## Brev update boundary

The only target is Launchable ID `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`.

Before any Brev command, verify the current CLI version and exact command help.
Use a dry run to inspect the stored definition without changing organization or
authentication state. Confirm that the definition still uses:

- the intended repository source;
- the current saved `launchable/setup.sh` body;
- exactly one required `NVIDIA_API_KEY` field with no default;
- fixed model `nvidia/nemotron-3-nano-30b-a3b`;
- port 8888 with organization-only access;
- the four-notebook order.

A repository push alone is not proof that the saved Launchable definition was
updated. If an authoring change is required, use the supported Brev Console
workflow. Do not use undocumented endpoints. A fresh deployment proves the
new definition only after the live L4 gates pass.

## Failure and rollback

- Any local, GPU, hosted, browser, or secret-handling failure blocks the public
  update.
- Keep the prior exact commit available as rollback source.
- Do not patch an existing VM and describe it as a Launchable update.
- If nvMolKit 0.6.0 cannot pass the scientific gates, restore the Launchable to
  the prior 0.5.0 commit and report the upgrade as blocked rather than weakening
  validation.
