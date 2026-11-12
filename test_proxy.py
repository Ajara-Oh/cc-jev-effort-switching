"""Offline tests: no API requests, no credentials."""
import http.client
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import jev_effort_proxy as proxy

ROOT = Path(__file__).resolve().parent


def user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


class RewriteTests(unittest.TestCase):
    def setUp(self):
        proxy.CFG = dict(proxy.DEFAULTS, api_key="test")
        proxy.STORE = proxy.InjectionStore()

    def request(self, msgs):
        return {"model": "claude-opus-5-5", "system": "s", "messages": msgs}

    def test_markers_replay_at_same_position(self):
        msgs = [user("a")]
        with mock.patch.object(proxy, "ask_jev", return_value=("low", 0.9)):
            first, modified = proxy.rewrite(self.request(msgs))
        self.assertTrue(modified)
        self.assertEqual(first["messages"], [proxy._marker("low"), user("a")])

        # Next turn: the cache_control breakpoint moved and Jev keeps the level.
        msgs = [dict(user("a"), cache_control={"type": "ephemeral"}), assistant("b"), user("c")]
        with mock.patch.object(proxy, "ask_jev", return_value=("low", 0.9)):
            second, _ = proxy.rewrite(self.request(msgs))
        self.assertEqual(second["messages"][0], proxy._marker("low"))
        self.assertEqual(second["messages"][1:], msgs)

    def test_low_confidence_leaves_request_alone(self):
        body = self.request([user("a")])
        with mock.patch.object(proxy, "ask_jev", return_value=("xhigh", 0.1)):
            self.assertEqual(proxy.rewrite(body), (body, False))

    def test_unsupported_model_passes_through(self):
        body = dict(self.request([user("a")]), model="claude-haiku-5-5")
        with mock.patch.object(proxy, "ask_jev") as ask:
            self.assertEqual(proxy.rewrite(body), (body, False))
        ask.assert_not_called()


class ServiceTests(unittest.TestCase):
    def test_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            for name in ("jev_proxy_service.py", "jev_effort_proxy.py", "claude-jev"):
                shutil.copy(ROOT / name, root)
            env = dict(os.environ, JEV_PORT=str(port), JEV_API_KEY="",
                       JEV_CONFIG=str(root / "missing.json"))

            def ensure():
                return subprocess.check_output(
                    ["python3", str(root / "jev_proxy_service.py"), "ensure"],
                    env=env, text=True, timeout=15).strip()

            def ready():
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                try:
                    conn.request("GET", "/__jev_health")
                    response = conn.getresponse()
                    return response.status == 200 and response.read() == proxy.HEALTH
                except OSError:
                    return False
                finally:
                    conn.close()

            def children(pid):
                return Path(f"/proc/{pid}/task/{pid}/children").read_text().split()

            pid = None
            try:
                with ThreadPoolExecutor(max_workers=6) as pool:
                    self.assertEqual(list(pool.map(lambda _: ensure(), range(6))), [str(port)] * 6)
                pid = int((root / "proxy-service.lock").read_text())
                self.assertTrue(ready(), "proxy must outlive the launchers")
                self.assertEqual(os.getsid(pid), pid, "supervisor must be detached")
                first = children(pid)
                self.assertEqual(len(first), 1, "concurrent launchers must share one proxy")

                os.kill(int(first[0]), signal.SIGKILL)
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    now = children(pid)
                    if now and now != first and ready():
                        break
                    time.sleep(0.1)
                else:
                    self.fail("proxy did not restart after a crash")

                fake = root / "claude"
                fake.write_text('#!/bin/sh\nprintf "%s\\n" "$ANTHROPIC_BASE_URL" "$@"\n')
                fake.chmod(0o755)
                output = subprocess.check_output(
                    ["bash", str(root / "claude-jev"), "--resume", "test session"],
                    env=dict(env, PATH=f"{root}:{env['PATH']}"), text=True, timeout=15)
                self.assertEqual(output.splitlines(),
                                 [f"http://127.0.0.1:{port}", "--resume", "test session"])
                self.assertTrue(ready(), "exiting claude must not stop the proxy")
                self.assertEqual(int((root / "proxy-service.lock").read_text()), pid)
            finally:
                if pid is None and (root / "proxy-service.lock").exists():
                    pid = int((root / "proxy-service.lock").read_text())
                if pid:
                    os.kill(pid, signal.SIGTERM)
                    deadline = time.monotonic() + 8
                    while ready() and time.monotonic() < deadline:
                        time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
