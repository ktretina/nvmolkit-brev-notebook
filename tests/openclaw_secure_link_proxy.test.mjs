import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import crypto from "node:crypto";
import { once } from "node:events";
import http from "node:http";
import net from "node:net";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  createSecureLinkProxy,
  PROXY_DEFAULTS,
} from "../launchable/openclaw_secure_link_proxy.mjs";


const PROXY_PATH = fileURLToPath(
  new URL("../launchable/openclaw_secure_link_proxy.mjs", import.meta.url),
);
const SECURE_LINK_HOST =
  "open-chemistry-agent-4z4yqg7de.apps.run.brev.nvidia.com";
const SECURE_LINK_ORIGIN = `https://${SECURE_LINK_HOST}`;
const BACKEND_ORIGIN = "http://127.0.0.1:18789";


function listen(server, port) {
  return new Promise((resolve, reject) => {
    const onError = (error) => reject(error);
    server.once("error", onError);
    server.listen(port, "127.0.0.1", () => {
      server.off("error", onError);
      resolve(server.address().port);
    });
  });
}


function closeServer(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
}


function request(port, path, headers = {}) {
  return new Promise((resolve, reject) => {
    const outgoing = http.request(
      {
        host: "127.0.0.1",
        port,
        method: "GET",
        path,
        headers,
      },
      (response) => {
        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () => {
          resolve({
            statusCode: response.statusCode,
            headers: response.headers,
            body: Buffer.concat(chunks).toString("utf8"),
          });
        });
      },
    );
    outgoing.on("error", reject);
    outgoing.end();
  });
}


function encodeMaskedTextFrame(text) {
  const payload = Buffer.from(text, "utf8");
  assert.ok(payload.length <= 125);
  const mask = Buffer.from([0x11, 0x22, 0x33, 0x44]);
  const masked = Buffer.alloc(payload.length);
  for (let index = 0; index < payload.length; index += 1) {
    masked[index] = payload[index] ^ mask[index % mask.length];
  }
  return Buffer.concat([
    Buffer.from([0x81, 0x80 | payload.length]),
    mask,
    masked,
  ]);
}


function decodeMaskedTextFrame(buffer) {
  if (buffer.length < 6) {
    return null;
  }
  const length = buffer[1] & 0x7f;
  if ((buffer[1] & 0x80) === 0 || length > 125 || buffer.length < 6 + length) {
    return null;
  }
  const mask = buffer.subarray(2, 6);
  const payload = Buffer.alloc(length);
  for (let index = 0; index < length; index += 1) {
    payload[index] = buffer[6 + index] ^ mask[index % mask.length];
  }
  return payload.toString("utf8");
}


function encodeTextFrame(text) {
  const payload = Buffer.from(text, "utf8");
  assert.ok(payload.length <= 125);
  return Buffer.concat([Buffer.from([0x81, payload.length]), payload]);
}


function readTextFrame(socket, initial = Buffer.alloc(0)) {
  return new Promise((resolve, reject) => {
    let buffered = initial;
    const cleanup = () => {
      socket.off("data", onData);
      socket.off("error", onError);
      socket.off("close", onClose);
    };
    const check = () => {
      if (buffered.length < 2) {
        return;
      }
      const length = buffered[1] & 0x7f;
      if (length > 125 || buffered.length < 2 + length) {
        return;
      }
      cleanup();
      resolve(buffered.subarray(2, 2 + length).toString("utf8"));
    };
    const onData = (chunk) => {
      buffered = Buffer.concat([buffered, chunk]);
      check();
    };
    const onError = (error) => {
      cleanup();
      reject(error);
    };
    const onClose = () => {
      cleanup();
      reject(new Error("socket closed before a WebSocket data frame arrived"));
    };
    socket.on("data", onData);
    socket.on("error", onError);
    socket.on("close", onClose);
    check();
  });
}


function readHandshake(socket) {
  return new Promise((resolve, reject) => {
    let buffered = Buffer.alloc(0);
    const cleanup = () => {
      socket.off("data", onData);
      socket.off("error", onError);
      socket.off("close", onClose);
    };
    const onData = (chunk) => {
      buffered = Buffer.concat([buffered, chunk]);
      const boundary = buffered.indexOf("\r\n\r\n");
      if (boundary === -1) {
        return;
      }
      cleanup();
      resolve({
        headers: buffered.subarray(0, boundary + 4).toString("latin1"),
        remainder: buffered.subarray(boundary + 4),
      });
    };
    const onError = (error) => {
      cleanup();
      reject(error);
    };
    const onClose = () => {
      cleanup();
      reject(new Error("socket closed before the WebSocket handshake completed"));
    };
    socket.on("data", onData);
    socket.on("error", onError);
    socket.on("close", onClose);
  });
}


async function requestUpgradeHandshake(port, headers) {
  const socket = net.connect({ host: "127.0.0.1", port });
  await once(socket, "connect");
  const key = Buffer.from("invalid-upgrade").toString("base64");
  const handshake = readHandshake(socket);
  socket.write(
    "GET /socket HTTP/1.1\r\n" +
      headers.map(([name, value]) => `${name}: ${value}\r\n`).join("") +
      "Upgrade: websocket\r\n" +
      "Connection: Upgrade\r\n" +
      `Sec-WebSocket-Key: ${key}\r\n` +
      "Sec-WebSocket-Version: 13\r\n\r\n",
  );
  const result = await handshake;
  socket.destroy();
  return result.headers;
}


test("bootstraps once, proxies HTTP, and tunnels WebSocket data", async (t) => {
  assert.deepEqual(PROXY_DEFAULTS, {
    listenHost: "0.0.0.0",
    listenPort: 18788,
    backendHost: "127.0.0.1",
    backendPort: 18789,
  });
  const upgrades = [];
  const backendSockets = new Set();
  const backend = http.createServer((incoming, response) => {
    response.writeHead(200, {
      "content-type": "application/json",
      "x-mock-backend": "yes",
    });
    response.end(
      JSON.stringify({
        method: incoming.method,
        url: incoming.url,
        testHeader: incoming.headers["x-acs-test"] ?? null,
      }),
    );
  });
  backend.on("connection", (socket) => {
    backendSockets.add(socket);
    socket.on("close", () => backendSockets.delete(socket));
  });
  backend.on("upgrade", (incoming, socket, head) => {
    upgrades.push({ url: incoming.url, headers: incoming.headers });
    const accept = crypto
      .createHash("sha1")
      .update(`${incoming.headers["sec-websocket-key"]}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
      .digest("base64");
    socket.write(
      "HTTP/1.1 101 Switching Protocols\r\n" +
        "Upgrade: websocket\r\n" +
        "Connection: Upgrade\r\n" +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    let buffered = head;
    const receive = (chunk) => {
      buffered = Buffer.concat([buffered, chunk]);
      const text = decodeMaskedTextFrame(buffered);
      if (text !== null) {
        assert.equal(text, "through-proxy");
        socket.write(encodeTextFrame("backend-reply"));
        socket.off("data", receive);
      }
    };
    socket.on("data", receive);
    if (head.length > 0) {
      receive(Buffer.alloc(0));
    }
  });

  const backendPort = await listen(backend, 0);
  t.after(async () => {
    for (const socket of backendSockets) {
      socket.destroy();
    }
    await closeServer(backend);
  });

  const token = "acs-test-token +/&";
  const proxy = createSecureLinkProxy({ token, backendPort });
  const proxySockets = new Set();
  proxy.on("connection", (socket) => {
    proxySockets.add(socket);
    socket.on("close", () => proxySockets.delete(socket));
  });
  const proxyPort = await listen(proxy, 0);
  t.after(async () => {
    for (const socket of proxySockets) {
      socket.destroy();
    }
    await closeServer(proxy);
  });

  const bootstrap = await request(proxyPort, "/");
  assert.equal(bootstrap.statusCode, 200);
  assert.match(bootstrap.headers["cache-control"], /no-store/);
  assert.match(bootstrap.headers["content-type"], /^text\/html/);
  assert.match(bootstrap.body, /const TOKEN="acs-test-token \+\/&";/);
  assert.ok(
    bootstrap.body.includes(
      "location.replace('/?__acs_boot=1#token=' + encodeURIComponent(TOKEN))",
    ),
  );

  const booted = await request(proxyPort, "/?__acs_boot=1");
  assert.deepEqual(JSON.parse(booted.body), {
    method: "GET",
    url: "/",
    testHeader: null,
  });

  const bootedWithQuery = await request(
    proxyPort,
    "/?alpha=1&__acs_boot=1&beta=two",
  );
  assert.equal(JSON.parse(bootedWithQuery.body).url, "/?alpha=1&beta=two");

  const staticResponse = await request(proxyPort, "/assets/app.js?cache=1", {
    "x-acs-test": "static-header",
  });
  assert.equal(staticResponse.statusCode, 200);
  assert.equal(staticResponse.headers["x-mock-backend"], "yes");
  assert.deepEqual(JSON.parse(staticResponse.body), {
    method: "GET",
    url: "/assets/app.js?cache=1",
    testHeader: "static-header",
  });

  const socket = net.connect({ host: "127.0.0.1", port: proxyPort });
  await once(socket, "connect");
  const key = Buffer.from("0123456789abcdef").toString("base64");
  const handshake = readHandshake(socket);
  socket.write(
    "GET /socket?keep=1 HTTP/1.1\r\n" +
      `Host: ${SECURE_LINK_HOST}\r\n` +
      `Origin: ${SECURE_LINK_ORIGIN}\r\n` +
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

  assert.equal(upgrades.length, 1);
  assert.equal(upgrades[0].url, "/socket?keep=1");
  assert.equal(upgrades[0].headers.host, SECURE_LINK_HOST);
  assert.equal(upgrades[0].headers.origin, BACKEND_ORIGIN);
  assert.equal(upgrades[0].headers["x-acs-test"], "websocket-header");
});


test("rejects invalid Brev WebSocket origins before a backend connection", async (t) => {
  let backendConnectionCount = 0;
  const backendSockets = new Set();
  const backend = net.createServer((socket) => {
    backendConnectionCount += 1;
    backendSockets.add(socket);
    socket.on("close", () => backendSockets.delete(socket));
    socket.write(
      "HTTP/1.1 101 Switching Protocols\r\n" +
        "Connection: Upgrade\r\n" +
        "Upgrade: websocket\r\n\r\n",
    );
  });
  const backendPort = await listen(backend, 0);

  const proxy = createSecureLinkProxy({ token: "test-token", backendPort });
  const proxySockets = new Set();
  proxy.on("connection", (socket) => {
    proxySockets.add(socket);
    socket.on("close", () => proxySockets.delete(socket));
  });
  const proxyPort = await listen(proxy, 0);
  t.after(async () => {
    for (const socket of proxySockets) {
      socket.destroy();
    }
    for (const socket of backendSockets) {
      socket.destroy();
    }
    await closeServer(proxy);
    await closeServer(backend);
  });

  const invalidRequests = [
    {
      name: "missing Host",
      headers: [["Origin", SECURE_LINK_ORIGIN]],
    },
    {
      name: "missing Origin",
      headers: [["Host", SECURE_LINK_HOST]],
    },
    {
      name: "duplicate Host",
      headers: [
        ["Host", SECURE_LINK_HOST],
        [
          "Host",
          "open-chemistry-agent-attacker.apps.run.brev.nvidia.com",
        ],
        ["Origin", SECURE_LINK_ORIGIN],
      ],
    },
    {
      name: "duplicate Origin",
      headers: [
        ["Host", SECURE_LINK_HOST],
        ["Origin", SECURE_LINK_ORIGIN],
        [
          "Origin",
          "https://open-chemistry-agent-attacker.apps.run.brev.nvidia.com",
        ],
      ],
    },
    {
      name: "mismatched Origin",
      headers: [
        ["Host", SECURE_LINK_HOST],
        [
          "Origin",
          "https://open-chemistry-agent-other.apps.run.brev.nvidia.com",
        ],
      ],
    },
    {
      name: "HTTP Origin",
      headers: [
        ["Host", SECURE_LINK_HOST],
        ["Origin", `http://${SECURE_LINK_HOST}`],
      ],
    },
    {
      name: "Origin suffix confusion",
      headers: [
        ["Host", SECURE_LINK_HOST],
        ["Origin", `${SECURE_LINK_ORIGIN}.attacker.example`],
      ],
    },
    {
      name: "Host suffix confusion",
      headers: [
        ["Host", `${SECURE_LINK_HOST}.attacker.example`],
        ["Origin", `${SECURE_LINK_ORIGIN}.attacker.example`],
      ],
    },
    {
      name: "non-alphanumeric instance ID",
      headers: [
        [
          "Host",
          "open-chemistry-agent-4z4y-qg7de.apps.run.brev.nvidia.com",
        ],
        [
          "Origin",
          "https://open-chemistry-agent-4z4y-qg7de.apps.run.brev.nvidia.com",
        ],
      ],
    },
    {
      name: "uppercase instance ID",
      headers: [
        [
          "Host",
          "open-chemistry-agent-4Z4YQG7DE.apps.run.brev.nvidia.com",
        ],
        [
          "Origin",
          "https://open-chemistry-agent-4Z4YQG7DE.apps.run.brev.nvidia.com",
        ],
      ],
    },
    {
      name: "Host with port",
      headers: [
        ["Host", `${SECURE_LINK_HOST}:443`],
        ["Origin", `${SECURE_LINK_ORIGIN}:443`],
      ],
    },
  ];

  const observed = [];
  for (const invalidRequest of invalidRequests) {
    const headers = await requestUpgradeHandshake(
      proxyPort,
      invalidRequest.headers,
    );
    observed.push({
      name: invalidRequest.name,
      status: headers.split("\r\n", 1)[0],
    });
  }

  assert.deepEqual(
    observed,
    invalidRequests.map(({ name }) => ({
      name,
      status: "HTTP/1.1 403 Forbidden",
    })),
  );
  assert.equal(backendConnectionCount, 0);
});


test("fails closed when ACS_DASHBOARD_TOKEN is missing", async () => {
  const environment = { ...process.env };
  delete environment.ACS_DASHBOARD_TOKEN;
  const child = spawn(process.execPath, [PROXY_PATH], {
    env: environment,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString("utf8");
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString("utf8");
  });
  const [code] = await once(child, "exit");

  assert.notEqual(code, 0);
  assert.equal(stdout, "");
  assert.match(stderr, /ACS_DASHBOARD_TOKEN is required/);
});
