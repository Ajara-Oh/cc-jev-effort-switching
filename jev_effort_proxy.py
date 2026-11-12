#!/usr/bin/env python3
"""Anthropic API proxy that lets Jev pick Claude's reasoning effort per step.

For each POST /v1/messages on a model that supports mid-conversation effort,
the proxy asks Jev which effort the next step needs. If that differs from the
level in force, it inserts an effort-only system message before the newest
message. Inserted markers are remembered by a hash of the preceding messages
and replayed at the same positions on later requests, so the cached prefix
never changes.

Any failure (no key, Jev error or timeout, unsupported model, upstream 400 on
the modified request) falls back to the unmodified request.
"""

import gzip
import hashlib
import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("JEV_CONFIG", os.path.join(ROOT, "config.json"))
BETA = "mid-conversation-output-config-2026-07-01"
HEALTH = b"jev-effort-proxy-v1\n"
LEVELS = ["low", "medium", "high", "xhigh", "max"]
SUPPORTED_MODEL = re.compile(r"claude-(opus-5(-5)?|fable-5-[12]|mythos-5-1)(?![-.]?\d)")
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

DEFAULTS = {
    "api_key": "",
    "jev_model": "jev-latest",
    "jev_url": "https://api.typesafe.ai/v1/systemone",
    "jev_timeout_s": 3.0,
    "upstream": os.environ.get("JEV_UPSTREAM", "https://api.anthropic.com"),
    "levels": ["low", "medium", "high", "xhigh"],
    "min_confidence": 0.45,
    "log_file": os.path.join(ROOT, "effort.log"),
    "debug": False,
}

CRITERIA = {
    "low": "Routine, mechanical next step: reading or listing files, running a command "
           "whose outcome is predictable, applying an edit that has already been decided, "
           "or reporting results that are already clear.",
    "medium": "Ordinary progress where the path forward is clear: straightforward edits, "
              "simple searches, following an existing plan step by step.",
    "high": "The next step needs real reasoning: planning a change, interpreting "
            "unexpected output, non-trivial debugging, or a fresh request that is "
            "moderately complex.",
    "xhigh": "The agent appears stuck or the problem is hard: repeated or recurring "
             "failures, contradictory evidence, the same error seen again, a subtle "
             "correctness/design/concurrency question, or a large ambiguous request.",
    "max": "Exceptionally hard: many failed attempts in a row and no working hypothesis, "
           "or a deep problem where mistakes would be very costly.",
}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:
        log(f"config error: {e}", force=True)
    cfg["api_key"] = cfg["api_key"] or os.environ.get("JEV_API_KEY", "")
    cfg["levels"] = [l for l in LEVELS if l in cfg["levels"]] or ["medium"]
    cfg["debug"] = cfg["debug"] or os.environ.get("JEV_DEBUG") == "1"
    return cfg


CFG = DEFAULTS
_log_lock = threading.Lock()


def log(msg, force=False):
    if not (force or CFG.get("debug")):
        return
    with _log_lock:
        try:
            with open(os.path.expanduser(CFG["log_file"]), "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        except OSError:
            pass


# Markers: hash of the messages before a marker -> its effort level.

class InjectionStore:
    def __init__(self, cap=20000):
        self.d = OrderedDict()
        self.cap = cap
        self.lock = threading.Lock()

    def get(self, h):
        with self.lock:
            v = self.d.get(h)
            if v is not None:
                self.d.move_to_end(h)
            return v

    def put(self, h, level):
        with self.lock:
            self.d[h] = level
            self.d.move_to_end(h)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)


STORE = InjectionStore()


def _strip_cache_control(x):
    if isinstance(x, dict):
        return {k: _strip_cache_control(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [_strip_cache_control(v) for v in x]
    return x


def _chain(h, obj):
    # Claude Code moves cache_control between requests; it must not change the hash.
    blob = json.dumps(_strip_cache_control(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(h + blob.encode()).digest()


def _root_hash(body):
    return _chain(b"", {"model": body.get("model"), "system": body.get("system")})


def _marker(level):
    return {"role": "system", "content": [], "output_config": {"effort": level}}


# What Jev sees.

def _text_of(content, limit):
    if isinstance(content, str):
        s = content
    else:
        parts = []
        for b in content or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "tool_result":
                parts.append(_text_of(b.get("content"), limit))
        s = "\n".join(parts)
    s = re.sub(r"<system-reminder>.*?</system-reminder>", "", s, flags=re.S).strip()
    return s if len(s) <= limit else s[: limit // 2] + " …[cut]… " + s[-limit // 2:]


def _is_prompt(msg):
    """A user prompt, as opposed to a message carrying tool results."""
    if msg.get("role") != "user":
        return False
    c = msg.get("content")
    if isinstance(c, str):
        return True
    return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c or [])


def build_state(messages, current_level):
    real = [m for m in messages if m.get("role") in ("user", "assistant")]
    prompt_idx = max((i for i, m in enumerate(real) if _is_prompt(m)), default=0)
    task = _text_of(real[prompt_idx].get("content"), 2500) if real else ""
    since = real[prompt_idx + 1:]

    steps, errors_in_a_row = [], 0
    for m in since:
        blocks = m.get("content") if isinstance(m.get("content"), list) else []
        if m["role"] == "assistant":
            for b in blocks:
                if b.get("type") == "tool_use":
                    steps.append({"tool_call": b.get("name"),
                                  "input": json.dumps(b.get("input", {}))[:300]})
                elif b.get("type") == "text" and b.get("text", "").strip():
                    steps.append({"assistant_said": _text_of([b], 300)})
        else:
            for b in blocks:
                if b.get("type") == "tool_result":
                    err = bool(b.get("is_error")) or bool(
                        re.search(r"(Traceback|Error|FAILED|error:|exit code [1-9])",
                                  _text_of(b.get("content"), 4000)))
                    errors_in_a_row = errors_in_a_row + 1 if err else 0
                    steps.append({"tool_result": _text_of(b.get("content"), 400),
                                  "looks_like_error": err})

    return {
        "user_request": task,
        "phase": "fresh user request" if not since else "mid-task, processing tool results",
        "steps_taken_since_request": sum(1 for s in steps if "tool_call" in s),
        "consecutive_failed_tool_results": errors_in_a_row,
        "current_effort": current_level,
        "recent_steps": steps[-10:],
    }


def ask_jev(state):
    levels = CFG["levels"]
    if len(levels) == 1:
        return levels[0], 1.0
    body = json.dumps({
        "model": CFG["jev_model"],
        "state": state,
        "questions": {"effort": {
            "type": "choice",
            "instructions": "A coding agent is working on `user_request`. Choose the "
                            "reasoning effort it should use for its NEXT step, based on "
                            "`recent_steps` and `consecutive_failed_tool_results`. Prefer "
                            "less effort for routine steps and more when it is stuck.",
            "criteria": {l: CRITERIA[l] for l in levels},
        }},
    }).encode()
    u = urllib.parse.urlsplit(CFG["jev_url"])
    Conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = Conn(u.netloc, timeout=CFG["jev_timeout_s"])
    try:
        conn.request("POST", u.path, body, {
            "Authorization": f"Bearer {CFG['api_key']}",
            "Content-Type": "application/json",
        })
        r = conn.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"jev {r.status}: {data[:200]!r}")
        ans = json.loads(data)["answers"]["effort"]
        return ans["choice"], float(ans.get("confidence", 0))
    finally:
        conn.close()


def rewrite(body):
    """Return (body, modified) for a /v1/messages request body."""
    if not CFG["api_key"] or not SUPPORTED_MODEL.search(str(body.get("model", ""))):
        return body, False
    msgs = body.get("messages") or []
    if not msgs or msgs[-1].get("role") != "user":
        return body, False

    current = (body.get("output_config") or {}).get("effort") or "default"
    out, h, replayed = [], _root_hash(body), 0
    for m in msgs[:-1]:
        lvl = STORE.get(h)
        if lvl:
            out.append(_marker(lvl))
            current, replayed = lvl, replayed + 1
        out.append(m)
        if m.get("role") == "system" and (m.get("output_config") or {}).get("effort"):
            current = m["output_config"]["effort"]
        h = _chain(h, m)

    lvl = STORE.get(h)  # set if this is a retry of a request already decided
    if lvl is None:
        t0 = time.monotonic()
        try:
            choice, conf = ask_jev(build_state(msgs, current))
            ms = int((time.monotonic() - t0) * 1000)
            log(f"jev -> {choice} (conf {conf:.2f}, {ms}ms), current {current}")
            if choice != current and choice in LEVELS and conf >= CFG["min_confidence"]:
                lvl = choice
                STORE.put(h, lvl)
        except Exception as e:
            log(f"jev failed, keeping {current}: {e}", force=True)
    if lvl:
        out.append(_marker(lvl))
    out.append(msgs[-1])

    if not lvl and not replayed:
        return body, False
    return dict(body, messages=out), True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Claude Code aborts and resets connections routinely.
    _BENIGN = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
               TimeoutError, socket.timeout, OSError)

    def log_message(self, fmt, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except self._BENIGN:
            pass

    def finish(self):
        try:
            super().finish()
        except self._BENIGN:
            pass

    def handle_error(self, request, client_address):
        # Never write to stderr: it is the terminal Claude Code draws on.
        log(f"connection error {client_address}: {sys.exc_info()[1]!r}", force=True)

    def _forward(self, method, body, headers):
        u = urllib.parse.urlsplit(CFG["upstream"])
        Conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = Conn(u.netloc, timeout=600)
        conn.request(method, u.path.rstrip("/") + self.path, body, headers)
        return conn, conn.getresponse()

    def _relay(self, conn, resp):
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while chunk := resp.read1(65536):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        finally:
            conn.close()

    def _handle(self):
        if self.command == "GET" and self.path == "/__jev_health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(HEALTH)))
            self.end_headers()
            self.wfile.write(HEALTH)
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}

        new_raw = None
        if self.command == "POST" and urllib.parse.urlsplit(self.path).path == "/v1/messages" and raw:
            try:
                gz = self.headers.get("Content-Encoding", "").lower() == "gzip"
                body, modified = rewrite(json.loads(gzip.decompress(raw) if gz else raw))
                if modified:
                    new_raw = json.dumps(body, separators=(",", ":")).encode()
                    if gz:
                        new_raw = gzip.compress(new_raw)
            except Exception as e:
                log(f"rewrite error, passing through: {e}", force=True)

        if new_raw is not None:
            h2 = dict(headers)
            key = next((k for k in h2 if k.lower() == "anthropic-beta"), "anthropic-beta")
            betas = [b for b in h2.get(key, "").split(",") if b.strip()]
            if BETA not in betas:
                betas.append(BETA)
            h2[key] = ",".join(betas)
            conn, resp = self._forward(self.command, new_raw, h2)
            if resp.status != 400:
                return self._relay(conn, resp)
            err = resp.read()
            conn.close()
            log(f"upstream rejected modified request, retrying plain: {err[:300]!r}", force=True)

        conn, resp = self._forward(self.command, raw, headers)
        self._relay(conn, resp)

    def _safe(self):
        try:
            self._handle()
        except self._BENIGN:
            pass
        except Exception as e:
            log(f"proxy error: {e}", force=True)
            try:
                self.send_error(502, str(e))
            except Exception:
                pass

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _safe


def main():
    global CFG
    CFG = load_config()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print(srv.server_address[1], flush=True)
    log(f"listening on {srv.server_address[1]}, jev {'on' if CFG['api_key'] else 'OFF (no api_key)'}",
        force=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
