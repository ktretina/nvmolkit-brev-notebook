# Brev Secure Link Origin Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the live ACS OpenClaw dashboard and make future Launchable deployments accept both current and legacy Brev Secure Link domains without weakening origin validation.

**Architecture:** Extend the proxy's strict hostname allowlist with `brevlab.com` while retaining single-header, HTTPS, exact-origin, and backend-origin-rewrite checks. Publish an immutable source-fix commit, repin the Console bootstrap to it, and install the same tested proxy on the running instance with scoped rollback.

**Tech Stack:** Node.js `http`, `net`, and `node:test`; Python `pytest`; Bash; Git; Brev CLI v0.6.332.

---

## File map

- Modify `tests/openclaw_secure_link_proxy.test.mjs`: test both accepted Brev domains and retain the rejection matrix.
- Modify `launchable/openclaw_secure_link_proxy.mjs`: extend the strict hostname allowlist.
- Modify `launchable/acs_console_bootstrap.sh`: pin future Console deployments to the source-fix commit.
- Create `/private/tmp/acs-openclaw-proxy-hotfix-80fyx449k.sh`: one-use live repair script; do not commit it.

### Task 1: Add the failing current-domain test

**Files:**
- Modify: `tests/openclaw_secure_link_proxy.test.mjs:19-22`
- Modify: `tests/openclaw_secure_link_proxy.test.mjs:317-343`

- [ ] **Step 1: Define both accepted hosts**

Replace the single host definition with:

```javascript
const CURRENT_SECURE_LINK_HOST =
  "open-chemistry-agent-80fyx449k.brevlab.com";
const LEGACY_SECURE_LINK_HOST =
  "open-chemistry-agent-4z4yqg7de.apps.run.brev.nvidia.com";
const SECURE_LINK_HOSTS = [CURRENT_SECURE_LINK_HOST, LEGACY_SECURE_LINK_HOST];
const SECURE_LINK_HOST = LEGACY_SECURE_LINK_HOST;
const SECURE_LINK_ORIGIN = `https://${SECURE_LINK_HOST}`;
const BACKEND_ORIGIN = "http://127.0.0.1:18789";
```

- [ ] **Step 2: Make the existing successful tunnel assertion run once per host**

Replace the socket creation, handshake, data-frame exchange, close, and final assertions at lines 317-343 with:

```javascript
for (const secureLinkHost of SECURE_LINK_HOSTS) {
  const socket = net.connect({ host: "127.0.0.1", port: proxyPort });
  await once(socket, "connect");
  const key = Buffer.from(`0123456789-${upgrades.length}`).toString("base64");
  const handshake = readHandshake(socket);
  socket.write(
    "GET /socket?keep=1 HTTP/1.1\r\n" +
      `Host: ${secureLinkHost}\r\n` +
      `Origin: https://${secureLinkHost}\r\n` +
      "Upgrade: websocket\r\n" +
      "Connection: Upgrade\r\n" +
      `Sec-WebSocket-Key: ${key}\r\n` +
      "Sec-WebSocket-Version: 13\r\n" +
      "X-ACS-Test: websocket-header\r\n\r\n",
  );
  const handshakeResult = await handshake;
  assert.match(handshakeResult.headers, /^HTTP\/1\.1 101 Switching Protocols/m);
  const reply = readTextFrame(socket, handshakeResult.remainder);
  socket.write(encodeMaskedTextFrame("through-proxy"));
  assert.equal(await reply, "backend-reply");
  socket.destroy();
}

assert.equal(upgrades.length, SECURE_LINK_HOSTS.length);
assert.deepEqual(
  upgrades.map(({ url, headers }) => ({
    url,
    host: headers.host,
    origin: headers.origin,
    testHeader: headers["x-acs-test"],
  })),
  SECURE_LINK_HOSTS.map((host) => ({
    url: "/socket?keep=1",
    host,
    origin: BACKEND_ORIGIN,
    testHeader: "websocket-header",
  })),
);
```

- [ ] **Step 3: Prove the new case fails**

Run `node --test tests/openclaw_secure_link_proxy.test.mjs`.

Expected: FAIL because `.brevlab.com` receives `HTTP/1.1 403 Forbidden` instead of `101 Switching Protocols`.

### Task 2: Implement and validate the allowlist fix

**Files:**
- Modify: `launchable/openclaw_secure_link_proxy.mjs:15-16`
- Test: `tests/openclaw_secure_link_proxy.test.mjs`

- [ ] **Step 1: Extend only the hostname pattern**

Use:

```javascript
const SECURE_LINK_HOST_PATTERN =
  /^open-chemistry-agent-[a-z0-9]+(?:\.brevlab\.com|\.apps\.run\.brev\.nvidia\.com)$/;
```

Do not change `singleRawHeader`, `origin === ` exact matching, the HTTPS requirement, or the backend-origin rewrite.

- [ ] **Step 2: Run focused tests**

Run:

```bash
node --test tests/openclaw_secure_link_proxy.test.mjs
python3 -m pytest tests/test_openclaw_secure_link_proxy.py tests/test_acs_nemoclaw_launchable_setup.py tests/test_acs_console_bootstrap.py -q
node --check launchable/openclaw_secure_link_proxy.mjs
git diff --check
```

Expected: all tests pass; syntax and whitespace checks exit zero.

- [ ] **Step 3: Commit only the source and test**

Run:

```bash
git add launchable/openclaw_secure_link_proxy.mjs tests/openclaw_secure_link_proxy.test.mjs
git commit -m "fix: accept current Brev Secure Link origin"
git rev-parse HEAD
```

Expected: a new source-fix commit. Record its full 40-character SHA for Task 3.

### Task 3: Repin the future-deployment bootstrap

**Files:**
- Read: `launchable/acs_console_bootstrap.sh.in`
- Modify: `launchable/acs_console_bootstrap.sh:7`
- Test: `tests/test_acs_console_bootstrap.py`

- [ ] **Step 1: Regenerate from the template**

Run this mechanical one-marker substitution using the source-fix commit at the current `HEAD`:

```bash
fix_commit="$(git rev-parse HEAD)"
test "${#fix_commit}" -eq 40
test "$(rg -o '@REVIEWED_PUBLIC_COMMIT_SHA@' launchable/acs_console_bootstrap.sh.in | wc -l | tr -d ' ')" -eq 1
sed "s/@REVIEWED_PUBLIC_COMMIT_SHA@/${fix_commit}/" \
  launchable/acs_console_bootstrap.sh.in \
  > /private/tmp/acs_console_bootstrap.sh
mv /private/tmp/acs_console_bootstrap.sh launchable/acs_console_bootstrap.sh
```

The saved file is identical to the template except for the one resolved commit marker.

- [ ] **Step 2: Verify the immutable pin and size**

Run:

```bash
fix_commit="$(git rev-parse HEAD)"
grep -Fx "readonly repo_commit=\"${fix_commit}\"" launchable/acs_console_bootstrap.sh
! rg -n '@REVIEWED_PUBLIC_COMMIT_SHA@' launchable/acs_console_bootstrap.sh
LC_ALL=C wc -c < launchable/acs_console_bootstrap.sh
python3 -m pytest tests/test_acs_console_bootstrap.py -q
```

Expected: exact pin found, no placeholder, size at most 16,384 bytes, all tests pass.

- [ ] **Step 3: Commit only the generated bootstrap**

Run `git add launchable/acs_console_bootstrap.sh` and `git commit -m "chore: repin ACS Launchable bootstrap"`.

### Task 4: Publish the reviewed branch

**Files:**
- Verify: branch `acs-fall-2026-launchable`

- [ ] **Step 1: Verify clean state and commit contents**

Run `git status --short`, `git log -3 --oneline --decorate`, and `git diff origin/acs-fall-2026-launchable...HEAD --check`.

Expected: clean worktree with design, source-fix, and repin commits; clean diff.

- [ ] **Step 2: Push without force**

Run `git push origin HEAD:acs-fall-2026-launchable`.

Expected: `origin/acs-fall-2026-launchable` advances to the repin commit.

- [ ] **Step 3: Verify publication**

Run `git ls-remote origin refs/heads/acs-fall-2026-launchable`. Verify the result equals local `HEAD`. Extract the pinned SHA from `launchable/acs_console_bootstrap.sh` and run `git merge-base --is-ancestor "$pinned_sha" HEAD`.

Expected: both checks exit zero.

### Task 5: Repair only instance `80fyx449k`

**Files:**
- Copy: `launchable/openclaw_secure_link_proxy.mjs`
- Create temporarily: `/private/tmp/acs-openclaw-proxy-hotfix-80fyx449k.sh`
- Modify remotely: the deployed `source-da8db3b.../launchable/openclaw_secure_link_proxy.mjs`

- [ ] **Step 1: Reconfirm target and command contracts**

Check `brev --version`, `brev copy --help`, `brev exec --help`, and `brev ls --org agents-in-ls --json`. Require ID `80fyx449k`, name `acs-fall-2026-gpu-chemistry-agent-lab-d46bba`, `RUNNING`, `READY`, and `HEALTHY`. Do not switch the active organization.

- [ ] **Step 2: Stage and syntax-check the tested proxy**

Copy the tested local proxy to `80fyx449k:/tmp/openclaw_secure_link_proxy.80fyx449k.new.mjs`, then run the remote Node binary with `--check` on that exact file.

Expected: exit zero and no source replacement yet.

- [ ] **Step 3: Run a rollback-capable scoped repair**

The temporary repair script must:

1. Use `set -Eeuo pipefail` and `umask 077`.
2. Require the exact deployed proxy, candidate, PID file, Node binary, NemoClaw binary, and sandbox name.
3. Confirm the recorded PID is numeric and its second `/proc/<pid>/cmdline` field equals the exact proxy path.
4. Back up the proxy as `openclaw_secure_link_proxy.mjs.pre-brevlab-hotfix` and atomically install the candidate with mode 600.
5. Obtain `nemoclaw acs-chemistry-agent gateway-token --quiet` only inside the remote shell, never print it, and unset it after starting Node.
6. Send TERM only to the confirmed proxy PID, wait at most 30 seconds, start the replacement, write its PID atomically, and require it to stay alive.
7. Require HTTP 200 from `http://127.0.0.1:18788/`.
8. On any post-replacement failure, restore the backup and restart the old proxy before exiting nonzero.

Execute this script only with `brev exec 80fyx449k @/private/tmp/acs-openclaw-proxy-hotfix-80fyx449k.sh`.

Expected: `Proxy hotfix installed and HTTP-ready.` No token appears in output.

### Task 6: Verify live transport and prepare Console handoff

**Files:**
- Verify remotely: proxy listener, private forward, and OpenClaw gateway
- Deliver: `launchable/acs_console_bootstrap.sh`

- [ ] **Step 1: Verify current origin acceptance**

Send a synthetic WebSocket upgrade to `127.0.0.1:18788` with:

```text
Host: open-chemistry-agent-80fyx449k.brevlab.com
Origin: https://open-chemistry-agent-80fyx449k.brevlab.com
```

Expected: not the proxy's `HTTP/1.1 403 Forbidden`; `101` proves the full handshake.

- [ ] **Step 2: Verify fail-closed behavior**

Repeat with an origin ending in `.attacker.example`.

Expected: `HTTP/1.1 403 Forbidden`.

- [ ] **Step 3: Verify processes without printing command lines**

Require Node to own port `18788`, the private forward to own `127.0.0.1:18789`, and `openclaw-gateway` to remain running. Do not print process command lines because they may contain transient tokens.

- [ ] **Step 4: Deliver the required Console update**

Provide the clickable `launchable/acs_console_bootstrap.sh`, its byte count, and its pinned source-fix SHA. Tell the user to edit the existing ACS chemistry Launchable, replace only the setup-script field, save it, and leave its required API-key parameter and Secure Link ports unchanged.

State that future deployments remain pinned to the old code until this Console save occurs. Do not automate the save through private endpoints or browser credentials.

- [ ] **Step 5: Request browser acceptance**

Ask the user to reload the live Secure Link in a fresh tab. Expected: automatic token fill followed by a connected dashboard without manual token or password entry.
