"""Start and verify claude-code-proxy before simulation."""

import subprocess
import time
import os
import sys
import atexit
import logging
import webbrowser
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger("llm_api.proxy")

PROXY_DIR = os.path.expanduser("~/claude-code-proxy")
HEALTH_URL = "http://localhost:42069/health"
AUTH_URL = "http://localhost:42069/auth/status"
LOGIN_URL = "http://localhost:42069/auth/login"

_proxy_process = None


def _is_running():
    """Check if proxy is already responding."""
    try:
        urlopen(HEALTH_URL, timeout=2)
        return True
    except (URLError, OSError):
        return False


def _is_authenticated():
    """Check if proxy has valid OAuth tokens."""
    import json
    try:
        resp = urlopen(AUTH_URL, timeout=2)
        data = json.loads(resp.read())
        return data.get("authenticated", False)
    except (URLError, OSError):
        return False


def _stop_proxy():
    """Cleanup: stop proxy on exit."""
    global _proxy_process
    if _proxy_process and _proxy_process.poll() is None:
        _proxy_process.terminate()
        _proxy_process.wait(timeout=5)
        logger.info("Proxy stopped")


def ensure_proxy():
    """Ensure proxy is running and authenticated before simulation starts."""
    global _proxy_process

    if not _is_running():
        logger.info("Starting claude-code-proxy...")
        _proxy_process = subprocess.Popen(
            ["node", "server/server.js"],
            cwd=PROXY_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(_stop_proxy)

        for _ in range(15):
            time.sleep(1)
            if _is_running():
                break
        else:
            logger.error("Proxy failed to start within 15s")
            sys.exit(1)

        logger.info("Proxy started (port 42069)")

    if not _is_authenticated():
        logger.info("Authentication required. Opening browser...")
        webbrowser.open(LOGIN_URL)

        print("\n[claudecode] Please authenticate in the browser.")
        print("[claudecode] Waiting for authentication...\n")

        for _ in range(120):
            time.sleep(1)
            if _is_authenticated():
                break
        else:
            logger.error("Authentication timed out (120s)")
            sys.exit(1)

    logger.info("Proxy ready and authenticated")
