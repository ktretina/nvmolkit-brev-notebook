# Zero-Input Workshop Key Design

## Decision

Remove all attendee-facing API-key inputs from only the **nvMolKit + Nemotron
Notebook** Launchable, ID `env-3HJtJW3qHg4Dw1I3xt75BfpBmZW`.

The repository will contain a secret-free setup template. The Brev Console will
hold the separately rendered setup-script body with the approved workshop key.
The user has already updated that saved Console body and removed
`NVIDIA_INFERENCE_API_KEY` from Setup values.

Do not change any other Launchable or deployed environment.

## Target user

The target user is one of about 110 concurrent workshop attendees. The attendee
should deploy the Launchable with a Brev coupon, open JupyterLab, and run the
notebooks without creating an NVIDIA API account, creating an API key, copying a
key, or entering a key in Brev.

## Goals

1. Present no API-key field during deployment.
2. Keep the workshop key out of Git, notebooks, tests, documentation, logs,
   screenshots, and review artifacts.
3. Preserve the fixed Inference Hub endpoint, Nemotron model, protected key-file
   path, notebook behavior, and `parallel_tool_calls=False` contract.
4. Keep the repository setup source useful as a redacted operator template.
5. State the security boundary accurately: a person who controls a deployed VM
   can recover the workshop key.
6. Require key rotation or revocation after the workshop.

## Non-goals

- Do not build a credential broker, proxy, or new service.
- Do not add another Launch parameter or attendee prompt.
- Do not put a reusable credential in a repository file or Launch parameter
  default.
- Do not change notebook chemistry, model selection, Jupyter configuration,
  hardware, storage, access policy, or notebook order.
- Do not edit the conversational Launchable or any other Brev resource.

## Repository setup template

Keep `launchable/setup.sh` as the operator-facing template because existing
documentation and tests already identify that path. Replace the environment
variable input logic with one unmistakable sentinel:

`__NVIDIA_INFERENCE_API_KEY__`

The repository version must fail before dependency installation when the
sentinel has not been replaced. A rendered Console copy must:

- contain an `sk-` Inference Hub workshop key;
- never print the key;
- write it atomically to
  `${HOME}/.config/nvmolkit/NVIDIA_INFERENCE_API_KEY`;
- set the directory to mode `0700` and the file to mode `0600`;
- clear the shell variable after persistence;
- preserve the rest of the current setup and Jupyter health checks.

The documentation must tell the operator to replace the sentinel only in a
private working copy before saving the script in the Brev Console. It must not
suggest that the rendered script be committed, attached to a ticket, pasted
into chat, or retained in a shared artifact.

## Attendee experience

The Launchable has no Setup values. The workshop organizer provisions the
approved Inference Hub credential in the saved Console setup body. Attendees do
not see a credential field and do not need an NVIDIA API account or API key.

Modules 2 and 3 continue to load the protected key file automatically. Module 1
continues to use nvMolKit directly and makes no hosted model call. Reference
mode remains available as a deterministic recovery path.

## Security boundary and lifecycle

Embedding a shared key in the saved setup script simplifies onboarding but does
not make the key secret from a deployer. Each attendee controls the deployed VM
and can read the protected key file. The key must therefore be workshop-only,
quota-limited where possible, monitored during the event, and rotated or
revoked after the event. VM deletion removes that VM's persisted copy but is not
a substitute for key revocation.

## Tests and acceptance

Use fake `sk-` values only in tests.

1. Assert that Launchable documentation specifies zero Setup values and no
   attendee key entry.
2. Assert that the repository template contains the sentinel and no real
   workshop credential.
3. Assert that the unrendered template fails closed before installation.
4. Render a temporary test copy with a fake key and verify atomic persistence,
   permissions, no output leakage, and no reliance on credential environment
   variables.
5. Preserve the setup-script size limit of 16,384 bytes and Bash syntax checks.
6. Run the focused setup, documentation, notebook inventory, and helper tests,
   followed by the complete deterministic suite.
7. Treat the user's saved Console update as configuration evidence only. A
   fresh deployment plus hosted notebook and browser tests is required before
   claiming that future instances work end to end.

## Rollback

If a fresh deployment cannot persist or use the provisioned key, stop the
Launchable update. Restore the prior saved Console setup body and required key
field only as a temporary operator-controlled rollback; do not change another
Launchable or weaken notebook validation.
