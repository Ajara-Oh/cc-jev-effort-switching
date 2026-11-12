#!/usr/bin/env python3
"""Detached supervisor that keeps one proxy running on a fixed port.

`ensure` starts the supervisor if needed, waits for the proxy and prints its
port. `run` is the supervisor itself; concurrent copies exit on the lock.
"""
import fcntl
import http.client
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("JEV_PORT", 37491))


def healthy():
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=0.5)
    try:
        conn.request("GET", "/__jev_health")
        response = conn.getresponse()
        return response.status == 200 and response.read(64) == b"jev-effort-proxy-v1\n"
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def occupied():
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


def supervise():
    os.umask(0o077)
    with (ROOT / "proxy-service.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        lock.seek(0)
        lock.truncate()
        lock.write(f"{os.getpid()}\n")
        lock.flush()

        child = None
        stopping = False

        def stop(*_):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while not stopping:
                if (child is None or child.poll() is not None) and not occupied():
                    child = subprocess.Popen(
                        [sys.executable, str(ROOT / "jev_effort_proxy.py"), str(PORT)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, close_fds=True,
                    )
                time.sleep(1)
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def ensure():
    os.umask(0o077)
    with (ROOT / "proxy-service.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "run"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True, close_fds=True,
        )
    for _ in range(100):
        if healthy():
            print(PORT)
            return
        time.sleep(0.1)
    raise SystemExit(f"Jev proxy not ready on {PORT}; see {ROOT / 'proxy-service.log'}")


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        supervise()
    elif sys.argv[1:] == ["ensure"]:
        ensure()
    else:
        raise SystemExit("usage: jev_proxy_service.py ensure|run")
