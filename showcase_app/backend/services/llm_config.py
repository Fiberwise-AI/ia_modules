"""
Shared LLM and agent configuration loader.

Reads provider, model, API key, workspace, tools, and logging config from
environment variables (loaded by main.py via dotenv).

Environment variables (see .env.example):
    DEFAULT_LLM_PROVIDER  - CLI type: "opencode" or "anthropic"
    LLM_PROVIDER          - provider name passed to CLI (e.g. "zai-coding-plan")
    LLM_MODEL             - model name (e.g. "glm-5.1")
    LLM_API_KEY           - API key for the provider
    PATTERN_MAX_TOKENS    - max tokens (default 2000)
    PATTERN_TEMPERATURE   - temperature (default 0.7)

    AGENT_CWD             - shared workspace for agents (default: project root)
    AGENT_LOGS_DIR        - directory for agent NDJSON logs (default: ./logs/agents)
    AGENT_TIMEOUT         - timeout in seconds (default: 120)

Tools are defined per agent role, not globally. Mode is derived from tools.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from ia_modules.agents.subprocess_executor import (
    get_shared_executor,
    set_shared_executor as set_shared_agent_executor,  # re-exported for main.py
)
from ia_modules.pipeline.llm_step import LLMStep
from ia_modules.pipeline.services import ServiceRegistry

logger = logging.getLogger(__name__)

# Default project root (two levels up from this file: services/ -> backend/ -> showcase_app/)
_DEFAULT_CWD = str(Path(__file__).resolve().parent.parent.parent)


@lru_cache(maxsize=1)
def get_llm_config() -> Dict[str, Any]:
    """Read LLM configuration from environment.

    Returns a dict with: cli_type, provider, model, api_key.
    All read directly from env vars — no magic mapping.

    DEFAULT_LLM_PROVIDER  → cli_type ("opencode" or "claude_code")
    LLM_PROVIDER          → provider name passed to CLI (e.g. "zai-coding-plan")
    LLM_MODEL             → model (e.g. "glm-5.1")
    LLM_API_KEY           → API key
    """
    cli_type = os.getenv("DEFAULT_LLM_PROVIDER", "opencode").strip().lower()
    provider = os.getenv("LLM_PROVIDER", "").strip() or None
    model = os.getenv("LLM_MODEL", "").strip() or None
    api_key = os.getenv("LLM_API_KEY", "").strip() or None

    config: Dict[str, Any] = {
        "cli_type": cli_type,
    }

    if provider:
        config["provider"] = provider
    if model:
        config["model"] = model
    if api_key:
        config["api_key"] = api_key

    return config


_WRITE_TOOLS = {"Edit", "Write", "Bash"}


def derive_mode(tools: list[str]) -> str:
    """Derive agent mode from its tool list.

    If any write tool (Edit, Write, Bash) is present → "execute".
    Otherwise → "research".
    """
    return "execute" if _WRITE_TOOLS.intersection(tools) else "research"


@lru_cache(maxsize=1)
def get_agent_config() -> Dict[str, Any]:
    """Read agent workspace configuration from environment.

    Cached: env vars are loaded once at app startup (dotenv in main.py) and
    the cwd/logs_dir resolution + mkdir calls are idempotent, so memoizing
    skips the per-call Path.resolve + mkdir syscalls. Cached for the whole
    process lifetime.

    Returns a dict with: cwd, logs_dir, timeout_seconds.
    Tools and mode are defined per agent role, not here.
    """
    cwd_raw = os.getenv("AGENT_CWD", "").strip()
    if cwd_raw:
        # Resolve relative paths against the project root (showcase_app/)
        cwd = str(Path(_DEFAULT_CWD).joinpath(cwd_raw).resolve())
    else:
        cwd = _DEFAULT_CWD
    logs_raw = os.getenv("AGENT_LOGS_DIR", "").strip()
    if logs_raw:
        logs_dir = str(Path(_DEFAULT_CWD).joinpath(logs_raw).resolve())
    else:
        logs_dir = str(Path(_DEFAULT_CWD) / "logs" / "agents")
    timeout = int(os.getenv("AGENT_TIMEOUT", "120"))

    # Ensure directories exist (idempotent; only runs once thanks to lru_cache)
    Path(cwd).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    return {
        "cwd": cwd,
        "logs_dir": logs_dir,
        "timeout_seconds": timeout,
    }


class LLMCallResult:
    """Result of an llm_call — carries text and the NDJSON job_id."""
    __slots__ = ("text", "job_id", "event_count")

    def __init__(self, text: Optional[str], job_id: Optional[str] = None, event_count: int = 0):
        self.text = text
        self.job_id = job_id
        self.event_count = event_count

    def __bool__(self):
        return self.text is not None

    def __str__(self):
        return self.text or ""


async def llm_call(
    system_prompt: str,
    user_message: str,
    timeout: int = 120,
    step_name: str = "_shared_llm",
    **extra_config: Any,
) -> LLMCallResult:
    """Make an LLM call via LLMStep using env-configured provider.

    Returns an LLMCallResult with .text and .job_id.
    .text is None on failure (caller should fall back).
    .job_id is the NDJSON log directory name (for reading events later).
    Uses AGENT_CWD as working directory so opencode.json is written there.
    """
    env_config = get_llm_config()
    agent_cfg = get_agent_config()
    step_config = {
        "system_prompt": system_prompt,
        "timeout_seconds": timeout,
        "cwd": agent_cfg["cwd"],
        "logs_dir": agent_cfg["logs_dir"],
        **env_config,
        **extra_config,
    }

    executor = get_shared_executor()

    step = LLMStep(step_name, step_config)
    # Attach a minimal service registry carrying the shared executor
    # so AgentStep._get_executor() finds it (standalone steps have no
    # orchestrator wiring services for them).
    step_services = ServiceRegistry()
    step_services.register("agent_executor", executor)
    step.services = step_services

    try:
        result = await step.run({"prompt": user_message})
    except Exception as e:
        logger.warning("LLM call failed (step=%s): %s", step_name, e, exc_info=True)
        return LLMCallResult(None)

    text = result.get("text", "").strip() or None
    return LLMCallResult(text, result.get("agent_job_id"), result.get("event_count", 0))
