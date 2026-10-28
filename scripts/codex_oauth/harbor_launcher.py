"""Harbor-specific launcher for the Codex (ChatGPT subscription) bridge.

Where ``codex_oauth.launcher`` fronts a generic command by exporting
``OPENAI_BASE_URL``/``OPENAI_API_KEY`` into that child's environment, Harbor needs
something different: the ``harbor trial start`` CLI process itself never calls the
LLM — the *containerized agent* (e.g. OpenHands) does, several process hops away,
inside a Docker sandbox. So the bridge address must be threaded in as Harbor CLI
flags that Harbor propagates to the agent:

  * ``--base-url <container_url>/v1``  -> config.agent.base_url -> LLM_BASE_URL,
    and triggers Harbor's auto-allowlist of the bridge host under no-network /
    allowlist task policies;
  * ``--ae OPENAI_API_KEY=<bridge_secret>`` -> config.agent.env, satisfying the
    agent's credential-PRESENCE gate (Harbor never validates the key) with the
    bridge's shared-auth secret rather than a real OpenAI key.

The address MUST be the CONTAINER-reachable one (host.docker.internal on Docker
Desktop, the docker bridge gateway on native Linux) — ``CodexBridge.container_base_url``
already resolves this.

Usage::

    PYTHONPATH=scripts python3 -m codex_oauth.harbor_launcher -- \
        python3 -m harbor trial start -p <task> -a openhands -m openai/gpt-6.1-sol
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from typing import Callable, Sequence

from .launcher import BridgeStartupError, CodexBridge

logger = logging.getLogger(__name__)

_BASE_URL_FLAGS = ("--base-url",)
_AGENT_ENV_FLAGS = ("--ae", "--agent-env")


def _has_base_url(argv: Sequence[str]) -> bool:
    return any(a in _BASE_URL_FLAGS for a in argv)


def _has_agent_env(argv: Sequence[str], name: str) -> bool:
    """True if the argv already carries ``--ae NAME=...`` / ``--agent-env NAME=...``."""
    prefix = f"{name}="
    for i, a in enumerate(argv):
        if a in _AGENT_ENV_FLAGS and i + 1 < len(argv) and argv[i + 1].startswith(prefix):
            return True
    return False


def build_harbor_command(
    harbor_argv: Sequence[str],
    *,
    container_base_url: str,
    bridge_secret: str,
    api_suffix: str = "/v1",
) -> list[str]:
    """Return ``harbor_argv`` with the bridge's --base-url and --ae injected.

    A caller-supplied --base-url or OPENAI_API_KEY agent-env is respected (never
    overridden), so an operator can still point at a different endpoint/key.

    The injected ``--ae OPENAI_API_KEY=`` only satisfies Harbor's credential
    presence gate when the model is ``openai/``-prefixed (that provider's
    candidate key name is OPENAI_API_KEY). A different prefix (e.g. ``anthropic/``)
    would need its own provider key name here, or the run fails the presence check.
    """
    argv = list(harbor_argv)
    base_url = container_base_url.rstrip("/") + api_suffix
    if not _has_base_url(argv):
        argv += ["--base-url", base_url]
    if not _has_agent_env(argv, "OPENAI_API_KEY"):
        argv += ["--ae", f"OPENAI_API_KEY={bridge_secret}"]
    return argv


def split_wrapped_command(argv: Sequence[str]) -> list[str]:
    """Strip a leading ``--`` separator from the wrapped command, if present."""
    argv = list(argv)
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def run(
    harbor_argv: Sequence[str],
    *,
    bridge: CodexBridge | None = None,
    runner: Callable[[list[str]], int] = subprocess.call,
    **bridge_kwargs: object,
) -> int:
    """Start the bridge, run the (flag-injected) Harbor command, always tear down.

    ``bridge`` and ``runner`` are injectable for tests; by default a fresh
    ``CodexBridge`` is created from ``bridge_kwargs`` and the command is run via
    ``subprocess.call``.
    """
    bridge = bridge or CodexBridge(**bridge_kwargs)  # type: ignore[arg-type]
    try:
        bridge.start()
        cmd = build_harbor_command(
            list(harbor_argv),
            container_base_url=bridge.container_base_url,
            bridge_secret=bridge.bridge_secret,
        )
        logger.info("running harbor via codex bridge: base_url=%s/v1",
                    bridge.container_base_url)
        return runner(cmd)
    finally:
        bridge.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m codex_oauth.harbor_launcher",
        description=(
            "Start the Codex subscription bridge, run a `harbor trial start ...` "
            "command with --base-url/--ae pointed at it (container-reachable), then "
            "tear the bridge down."
        ),
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--container-host", default=None,
                        help="Hostname a container uses to reach the bridge "
                             "(default host.docker.internal / docker gateway).")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- <harbor trial start ...>")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    command = split_wrapped_command(args.command)
    if not command:
        parser.error("no command given (use: ... -- harbor trial start ...)")

    bridge_kwargs = dict(port=args.port, host=args.host,
                         container_host=args.container_host, log_level=args.log_level)
    try:
        return run(command, **bridge_kwargs)  # type: ignore[arg-type]
    except BridgeStartupError as e:
        print(f"[codex-harbor-launcher] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
