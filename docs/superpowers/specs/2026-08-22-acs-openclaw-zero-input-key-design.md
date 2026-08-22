# ACS OpenClaw Zero-Input Credential Design

## Decision

Remove the attendee-facing API-key parameter from only the conversational ACS
OpenClaw Launchable, `env-3Hlp4pHBlTTlfDxfH41KkGhTeCV`.

Reuse the repository's accepted zero-input workshop pattern: keep a secret-free
Console bootstrap template in Git, render one private organizer copy through a
hidden prompt, and save only that rendered body in the existing Launchable
configuration. Do not create a credential broker, parameter default, or new
runtime secret service.

## Target experience

An attendee signs in to Brev, deploys the Launchable without Setup values,
opens **Open Chemistry Agent**, and runs the four published prompts. The
attendee never creates, sees, enters, copies, or downloads a provider
credential.

## Credential flow

1. `launchable/acs_console_bootstrap.sh.in` contains one unmistakable
   credential sentinel and no credential value.
2. An ACS-specific offline renderer reads one `nvapi-` workshop key with
   `getpass`, validates it, shell-quotes it, and replaces exactly one sentinel.
3. The renderer creates a new mode-`0600` file outside the repository. It
   refuses repository paths, symlinked parents, existing targets, empty values,
   non-`nvapi-` values, multiline values, and NUL bytes.
4. The organizer saves the private rendered body in the existing Brev
   Launchable setup-script field and removes every Launch parameter. The
   rendered file is then deleted.
5. At deployment, the bootstrap keeps the key out of Git and install child
   processes. It exports the key only for the final unified setup process.
6. The existing unified setup writes the key to its owned mode-`0600`
   transient file, clears the environment variable, and invokes Phase Zero.
7. The existing Phase Zero validation checks file ownership, type, mode,
   non-emptiness, and the `nvapi-` prefix, gives it only to NemoClaw onboarding,
   and deletes the transient file. NemoClaw/OpenShell keeps the provider
   credential on the host and routes credential-free sandbox requests.

No credential value may enter Git, tests, documentation, logs, command
arguments, URLs, environment listings, browser output, chat, artifacts, or
verification reports. Tests use fake `nvapi-` canaries only.

## Repository changes

- Add one bounded renderer for the ACS Console bootstrap.
- Replace the bootstrap environment-input boundary with the single sentinel.
- Keep `acs_nemoclaw_launchable_setup.sh`, `nemoclaw_phase_zero.sh`, the
  chemistry runner, verifier, and live-operation controller unchanged.
- Update the Console authoring sheet and the conversational attendee steps to
  specify no Launch parameters and no attendee key entry.
- Regenerate the public bootstrap twice: first with the existing pin for local
  verification, then in a second commit pinned to the reviewed source commit.

## Failure behavior

The unrendered public bootstrap must fail before Git, installation, or secret
storage. A missing, empty, malformed, multiply substituted, or unsafe render
must fail without printing the key or leaving a private output file. Setup must
not publish its ready marker unless the existing credential and runtime gates
all pass.

## Acceptance evidence

1. A failing test first proves the current authoring sheet and attendee page
   still require an API-key field.
2. Renderer and bootstrap tests prove the secret-free repository, safe private
   render, fail-closed behavior, child-process shielding, and exact handoff to
   the existing setup.
3. Existing focused setup, Phase Zero, page, verifier, and live-operation suites
   remain green.
4. The saved Launchable preview and a no-parameter dry run show no Setup value.
5. One fresh L4 deployment reaches the ready marker, then one isolated browser
   session completes the four prompts in order and downloads the verified ZIP.
6. Browser text, trajectory, image metadata, and archive scans find no
   credential names or values, internal setup paths, raw dashboard, or admin UI.
7. The task-owned paid instance is stopped after the pass or first unresolved
   blocker.

## Security boundary

The acceptance target is the attendee browser experience. Brev and NemoClaw
administrators retain the platform access defined by those products. The shared
workshop credential must be time-bounded, usage-limited, monitored, and revoked
after the workshop. This design does not claim that a person with administrator
or root access to a deployed VM cannot recover host credentials.

## Rejected alternatives

- A Launch-parameter default is rejected because it keeps a credential field
  and NVIDIA documentation advises against reusable credential defaults.
- A new broker, vault, or proxy is rejected as unnecessary infrastructure.
- A literal credential in source, public bootstrap, documentation, or command
  arguments is forbidden.

## Rollback

If the fresh deployment cannot use the saved credential safely, restore the
previous saved setup body and required parameter only as an operator-controlled
rollback, keep access restricted, stop the test instance, and report NOT READY.
