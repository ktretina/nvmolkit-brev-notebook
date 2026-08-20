#!/usr/bin/env node

import http from "node:http";
import net from "node:net";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";


export const PROXY_DEFAULTS = Object.freeze({
  listenHost: "0.0.0.0",
  listenPort: 18788,
  backendHost: "127.0.0.1",
  backendPort: 18789,
});
const SECURE_LINK_HOST_PATTERN =
  /^open-chemistry-agent-[a-z0-9]+(?:\.brevlab\.com|\.apps\.run\.brev\.nvidia\.com)$/;
const PRIVATE_DASHBOARD_ORIGIN = "http://127.0.0.1:18789";


function javascriptString(value) {
  return JSON.stringify(value)
    .replaceAll("<", "\\u003c")
    .replaceAll("\u2028", "\\u2028")
    .replaceAll("\u2029", "\\u2029");
}


function bootstrapHtml(token) {
  return (
    "<!doctype html><meta charset=utf-8>" +
    `<script>const TOKEN=${javascriptString(token)};` +
    "location.replace('/?__acs_boot=1#token=' + encodeURIComponent(TOKEN))" +
    "</script>"
  );
}


function stripBootstrapMarker(rawPath) {
  const queryStart = rawPath.indexOf("?");
  if (queryStart === -1) {
    return { found: false, path: rawPath };
  }
  const pathname = rawPath.slice(0, queryStart);
  const parameters = rawPath.slice(queryStart + 1).split("&");
  const preserved = parameters.filter((parameter) => parameter !== "__acs_boot=1");
  if (preserved.length === parameters.length) {
    return { found: false, path: rawPath };
  }
  return {
    found: true,
    path: preserved.length > 0 ? `${pathname}?${preserved.join("&")}` : pathname,
  };
}


function writeBadGateway(response) {
  if (response.headersSent) {
    response.destroy();
    return;
  }
  response.writeHead(502, {
    "cache-control": "no-store",
    "content-type": "text/plain; charset=utf-8",
  });
  response.end("Bad Gateway\n");
}


function singleRawHeader(request, expectedName) {
  const values = [];
  for (let index = 0; index < request.rawHeaders.length; index += 2) {
    if (request.rawHeaders[index].toLowerCase() === expectedName) {
      values.push(request.rawHeaders[index + 1]);
    }
  }
  return values.length === 1 ? values[0] : null;
}


function isAllowedSecureLinkUpgrade(request) {
  const host = singleRawHeader(request, "host");
  const origin = singleRawHeader(request, "origin");
  return (
    host !== null &&
    origin !== null &&
    SECURE_LINK_HOST_PATTERN.test(host) &&
    origin === `https://${host}`
  );
}


function rejectUpgrade(clientSocket) {
  clientSocket.end(
    "HTTP/1.1 403 Forbidden\r\n" +
      "Connection: close\r\n" +
      "Content-Length: 0\r\n\r\n",
  );
}


function forwardUpgrade(request, clientSocket, head, backendHost, backendPort) {
  const backendSocket = net.connect({ host: backendHost, port: backendPort });
  const closeBackend = () => backendSocket.destroy();

  clientSocket.on("error", closeBackend);
  clientSocket.on("close", closeBackend);
  backendSocket.on("error", () => clientSocket.destroy());
  backendSocket.once("connect", () => {
    let requestHead = `${request.method} ${request.url} HTTP/${request.httpVersion}\r\n`;
    for (let index = 0; index < request.rawHeaders.length; index += 2) {
      const name = request.rawHeaders[index];
      const value =
        name.toLowerCase() === "origin"
          ? PRIVATE_DASHBOARD_ORIGIN
          : request.rawHeaders[index + 1];
      requestHead += `${name}: ${value}\r\n`;
    }
    backendSocket.write(`${requestHead}\r\n`, "latin1");
    if (head.length > 0) {
      backendSocket.write(head);
    }
    clientSocket.pipe(backendSocket);
    backendSocket.pipe(clientSocket);
  });
}


export function createSecureLinkProxy({
  token,
  backendHost = PROXY_DEFAULTS.backendHost,
  backendPort = PROXY_DEFAULTS.backendPort,
} = {}) {
  if (typeof token !== "string" || token.length === 0) {
    throw new Error("ACS_DASHBOARD_TOKEN is required.");
  }

  const html = bootstrapHtml(token);
  const server = http.createServer((request, response) => {
    const rawPath = request.url ?? "/";
    const marker = stripBootstrapMarker(rawPath);
    const pathname = rawPath.split("?", 1)[0];

    if (request.method === "GET" && pathname === "/" && !marker.found) {
      response.writeHead(200, {
        "cache-control": "no-store, max-age=0",
        "content-length": Buffer.byteLength(html),
        "content-type": "text/html; charset=utf-8",
        expires: "0",
        pragma: "no-cache",
        "referrer-policy": "no-referrer",
        "x-content-type-options": "nosniff",
      });
      response.end(html);
      return;
    }

    const backendRequest = http.request(
      {
        host: backendHost,
        port: backendPort,
        method: request.method,
        path: marker.path,
        headers: request.headers,
      },
      (backendResponse) => {
        response.writeHead(backendResponse.statusCode ?? 502, backendResponse.headers);
        backendResponse.pipe(response);
      },
    );
    backendRequest.on("error", () => writeBadGateway(response));
    request.on("aborted", () => backendRequest.destroy());
    request.pipe(backendRequest);
  });

  server.on("upgrade", (request, clientSocket, head) => {
    if (!isAllowedSecureLinkUpgrade(request)) {
      rejectUpgrade(clientSocket);
      return;
    }
    forwardUpgrade(request, clientSocket, head, backendHost, backendPort);
  });
  return server;
}


function run() {
  const token = process.env.ACS_DASHBOARD_TOKEN;
  if (typeof token !== "string" || token.length === 0) {
    console.error("ACS_DASHBOARD_TOKEN is required.");
    process.exitCode = 1;
    return;
  }

  const server = createSecureLinkProxy({ token });
  const sockets = new Set();
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
  });
  server.once("error", () => {
    console.error("OpenClaw Secure Link proxy could not start.");
    process.exitCode = 1;
  });
  server.listen(PROXY_DEFAULTS.listenPort, PROXY_DEFAULTS.listenHost, () => {
    console.log(
      `OpenClaw Secure Link proxy listening on ${PROXY_DEFAULTS.listenHost}:${PROXY_DEFAULTS.listenPort}.`,
    );
  });

  let stopping = false;
  process.on("SIGTERM", () => {
    if (stopping) {
      return;
    }
    stopping = true;
    server.close(() => {
      process.exitCode = 0;
    });
    for (const socket of sockets) {
      socket.end();
    }
    setTimeout(() => {
      for (const socket of sockets) {
        socket.destroy();
      }
    }, 1_000).unref();
  });
}


const isEntryPoint =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href;
if (isEntryPoint) {
  run();
}
