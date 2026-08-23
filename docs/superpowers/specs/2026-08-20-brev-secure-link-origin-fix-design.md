# Brev Secure Link Origin Fix Design

**Date:** 2026-08-20

## Goal

Restore OpenClaw dashboard auto-login on the running ACS chemistry instance and prevent the same failure in future deployments of the Launchable.

Success requires both of these outcomes:

1. The dashboard on instance `80fyx449k` completes its WebSocket connection through the current `brevlab.com` Secure Link.
2. The Launchable source accepts both current and legacy Brev Secure Link domains without weakening its origin checks.

## Root cause

The token bootstrap succeeds, but the proxy rejects the following WebSocket origin before OpenClaw can authenticate it:

`https://open-chemistry-agent-80fyx449k.brevlab.com`

The deployed proxy accepts only hosts ending in `.apps.run.brev.nvidia.com`. It returns HTTP 403 for the current `.brevlab.com` host.

## Selected approach

Use a strict allowlist containing the two observed Brev Secure Link hostname forms:

- `open-chemistry-agent-<lowercase-alphanumeric-instance-id>.brevlab.com`
- `open-chemistry-agent-<lowercase-alphanumeric-instance-id>.apps.run.brev.nvidia.com`

The proxy will continue to require exactly one `Host` header and one `Origin` header. The origin must be HTTPS and must equal the host exactly. After validation, the proxy will rewrite the origin to the private OpenClaw dashboard origin, `http://127.0.0.1:18789`, and tunnel the WebSocket.

The proxy will continue to reject missing or duplicate headers, mismatched origins, HTTP origins, ports, uppercase or non-alphanumeric instance IDs, and hostname suffix confusion.

## Rejected alternatives

### Accept any same-origin HTTPS hostname

This would handle future Brev domain changes automatically, but it would remove the Brev-specific hostname boundary. The added flexibility is not worth the weaker validation.

### Make accepted domains a Launchable parameter

This would be flexible, but it would expose a security-sensitive value as deployment configuration and add a new setup failure mode. The two known platform domains are stable enough to keep in reviewed source.

## Current-instance repair

The repair will target only organization `agents-in-ls`, instance `80fyx449k`, and the OpenClaw Secure Link proxy on port `18788`.

The repair procedure will:

1. Reconfirm the instance identity and healthy state.
2. Back up the deployed proxy source.
3. Install the locally tested proxy source through an atomic file replacement.
4. Verify JavaScript syntax before changing the process.
5. Confirm that the recorded proxy PID belongs to the expected Node executable and proxy source.
6. Stop only that proxy process.
7. Obtain the gateway token inside the remote shell without printing it and start the replacement proxy.
8. Verify the bootstrap HTTP response and a `.brevlab.com` WebSocket upgrade.
9. Restore the backup and restart the previous proxy if any verification step fails.

The procedure will not restart OpenClaw, the NemoClaw sandbox, or the Brev instance. Existing dashboard connections may disconnect briefly while the proxy restarts.

## Future-deployment repair

Implementation will occur in the existing isolated worktree for branch `acs-fall-2026-launchable` so the user's dirty `main` worktree is untouched.

The durable change will:

1. Add failing tests for the current `.brevlab.com` hostname.
2. Update the proxy allowlist while preserving all fail-closed checks.
3. Run the focused Node and Python proxy tests, followed by the relevant setup-script tests.
4. Commit and push the reviewed source fix to `origin/acs-fall-2026-launchable`.
5. Generate a second commit that repins `launchable/acs_console_bootstrap.sh` to the immutable source-fix commit, then push it.

The existing Brev Launchable definition currently embeds a bootstrap pinned to commit `da8db3bbf4599d8d8fa41f3b9f6ebd51cf4ddb1f`. Therefore, repository publication alone cannot change future deployments. The Brev Console setup-script field must be replaced with the newly generated bootstrap. No supported, callable Launchable-authoring interface is available in this task, so implementation will provide the exact paste-ready file and verify its byte size and pinned commit. The user must perform the Console update unless an authenticated supported authoring tool becomes available.

## Tests and acceptance

Automated tests must prove:

- `.brevlab.com` upgrades are accepted and tunnel data.
- `.apps.run.brev.nvidia.com` upgrades remain accepted.
- Accepted external origins are rewritten to the private backend origin.
- Invalid and ambiguous origins receive HTTP 403 before any backend connection.
- Missing dashboard tokens still fail closed.
- The generated bootstrap contains one full reviewed commit SHA and no placeholder.
- The Console setup script remains at or below Brev's 16,384-byte limit.

Live acceptance must prove:

- The proxy returns HTTP 200 for the dashboard bootstrap.
- A WebSocket request with the exact live `.brevlab.com` host and origin passes the proxy rather than returning its 403 response.
- The OpenClaw gateway remains running.
- No credential or gateway token is printed in logs or command output.

Browser auto-login is considered verified only after the user reloads the Secure Link and the dashboard connects without manual token entry. Transport verification from the instance is necessary but does not replace that browser check.

## Rollback

For the running instance, restore the backed-up proxy source and restart only the proxy with the existing gateway token.

For future deployments, keep the existing Console bootstrap available until the new pinned bootstrap passes live acceptance. If the new definition fails, restore the previous Console script pinned to `da8db3bbf4599d8d8fa41f3b9f6ebd51cf4ddb1f`.
