"""HTTP clients for Anthropic, OpenAI, and OpenRouter Messages-style endpoints.

This module is the paper-faithful client surface used by the auditing pipeline.
It exposes:

- ``call_anthropic`` — Anthropic Messages API with prompt caching and adaptive
  thinking support.
- ``call_openrouter`` — OpenRouter chat-completions, with provider-order
  routing and reasoning passthrough.
- ``call_openai`` — OpenAI chat-completions (GPT-family targets).
- ``sanitize_messages`` — message-shape normalization shared by all callers
  (orphan tool_use repair, role coercion, content shape unification).
- ``APIClients`` — convenience wrapper holding three semaphores
  (auditor / target / judge) and routing by model_id prefix.

The retry policy, concurrency limits, prompt caching and sanitization are
algorithmically identical to the paper-time client. Training-only paths
(Tinker, Qwen3 OAI fork, finetuning beta header, multi-key round-robin) have
been removed.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_AUDITOR_CONCURRENT",
    "DEFAULT_TARGET_CONCURRENT",
    "DEFAULT_JUDGE_CONCURRENT",
    "get_api_key",
    "sanitize_messages",
    "call_anthropic",
    "call_openrouter",
    "call_openai",
    "APIClients",
]


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

DEFAULT_AUDITOR_CONCURRENT = 150
DEFAULT_TARGET_CONCURRENT = 100
DEFAULT_JUDGE_CONCURRENT = 75

# Models that explicitly reject assistant-message prefill. If sanitization
# strips an empty user message and exposes a trailing assistant message we
# append a "Please continue." nudge for these models only.
_NO_PREFILL_MODELS = {
    "claude-opus-4-6",
    "claude-opus-4-20250514",
    "claude-sonnet-4-6",
}


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _get_retry_after(response: httpx.Response) -> Optional[float]:
    """Extract a numeric retry-after header value if present."""
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return None


def _calculate_backoff(attempt: int, base: float = 2.0, max_wait: float = 60.0) -> float:
    """Exponential backoff with multiplicative [0.5, 1.5] jitter."""
    wait = min(base ** (attempt + 1), max_wait)
    jitter = random.uniform(0.5, 1.5)
    return wait * jitter


# ---------------------------------------------------------------------------
# API key resolution (one key per provider)
# ---------------------------------------------------------------------------


def get_api_key(provider: str = "anthropic") -> Optional[str]:
    """Return the configured API key for ``provider``.

    Looks first in the process environment, then in a ``.env`` file located
    next to this module's package root. Only the public, plain-named env vars
    are consulted: ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
    ``OPENROUTER_API_KEY``.
    """
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = key_map.get(provider, provider)

    key = os.environ.get(env_var)
    if key:
        return key

    # Fallback: look for a .env file next to the repo root.
    env_paths = [
        Path(__file__).parent.parent / ".env",  # repo root
        Path(__file__).parent / ".env",         # core/.env
    ]
    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(f"{env_var}="):
                            return line.split("=", 1)[1].strip()
            except OSError:
                continue
    return None


# ---------------------------------------------------------------------------
# Message sanitization
# ---------------------------------------------------------------------------


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Normalize messages to a shape the Anthropic Messages API accepts.

    Specifically:
    - Convert OpenAI ``role="tool"`` messages to Anthropic ``role="user"``
      with a ``tool_result`` content block.
    - Convert OpenAI-style ``tool_calls`` on assistant messages to Anthropic
      ``tool_use`` content blocks.
    - Strip trailing whitespace from assistant messages (the API rejects
      "final assistant content cannot end with trailing whitespace").
    - Drop empty messages, except a final empty assistant message (used as a
      prefill anchor by some callers).
    - Merge consecutive ``role="user"`` messages whose content is entirely
      ``tool_result`` blocks (deep-copying on first merge to avoid mutating
      the caller's data).
    - Strip orphaned ``tool_use`` blocks — if an assistant ``tool_use`` is
      not followed by a matching ``tool_result``, the ``tool_use`` block is
      removed (the API would otherwise 400).
    """
    sanitized: list[dict] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # OpenAI role="tool" -> Anthropic role="user" + tool_result block.
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            tool_content = content if isinstance(content, str) else str(content)
            sanitized.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_content,
                }],
            })
            continue

        # OpenAI-style assistant tool_calls -> Anthropic tool_use blocks.
        if role == "assistant" and msg.get("tool_calls"):
            blocks: list[dict] = []
            text = content if isinstance(content, str) else ""
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                tc_name = func.get("name", "") if isinstance(func, dict) else str(func)
                tc_args = func.get("arguments", {}) if isinstance(func, dict) else {}
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except Exception:
                        tc_args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"tool_{i}"),
                    "name": tc_name,
                    "input": tc_args,
                })
            sanitized.append({"role": "assistant", "content": blocks})
            continue

        # Strip trailing whitespace from assistant text (API hard-rejects it).
        if role == "assistant" and isinstance(content, str):
            content = content.rstrip()
            msg = dict(msg, content=content)

        # Detect effectively-empty messages.
        is_empty = False
        if content is None:
            is_empty = True
        elif isinstance(content, str) and content.strip() == "":
            is_empty = True
        elif isinstance(content, list):
            has_content = False
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text", "").strip():
                        has_content = True
                        break
                    if item.get("type") in ("tool_use", "tool_result"):
                        has_content = True
                        break
                elif isinstance(item, str) and item.strip():
                    has_content = True
                    break
            is_empty = not has_content

        if is_empty:
            is_final_assistant = (i == len(messages) - 1) and role == "assistant"
            if not is_final_assistant:
                logger.debug(f"Skipping empty message at index {i}: role={role}")
                continue

        sanitized.append(msg)

    # Merge consecutive user messages composed entirely of tool_result blocks.
    merged: list[dict] = []
    for msg in sanitized:
        if (merged
                and msg.get("role") == "user"
                and merged[-1].get("role") == "user"
                and isinstance(msg.get("content"), list)
                and isinstance(merged[-1].get("content"), list)
                and all(isinstance(c, dict) and c.get("type") == "tool_result"
                        for c in msg["content"])
                and all(isinstance(c, dict) and c.get("type") == "tool_result"
                        for c in merged[-1]["content"])):
            if not merged[-1].get("_merged"):
                merged[-1] = copy.deepcopy(merged[-1])
                merged[-1]["_merged"] = True
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)

    for msg in merged:
        msg.pop("_merged", None)

    # Strip orphaned tool_use blocks (no matching tool_result on the next msg).
    fixed: list[dict] = []
    for i, msg in enumerate(merged):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            tool_use_ids = [
                b.get("id") for b in msg["content"]
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            if tool_use_ids:
                next_msg = merged[i + 1] if i + 1 < len(merged) else None
                result_ids: set = set()
                if (next_msg
                        and next_msg.get("role") == "user"
                        and isinstance(next_msg.get("content"), list)):
                    result_ids = {
                        b.get("tool_use_id") for b in next_msg["content"]
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
                orphaned_ids = set(tool_use_ids) - result_ids
                if orphaned_ids:
                    clean_msg = copy.deepcopy(msg)
                    clean_msg["content"] = [
                        b for b in clean_msg["content"]
                        if not (isinstance(b, dict)
                                and b.get("type") == "tool_use"
                                and b.get("id") in orphaned_ids)
                    ]
                    if not clean_msg["content"]:
                        clean_msg["content"] = [{"type": "text", "text": "(continued)"}]
                    fixed.append(clean_msg)
                    continue
        fixed.append(msg)

    return fixed


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


async def call_anthropic(
    model_id: str,
    system: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = 4000,
    api_key: Optional[str] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_retries: int = 25,
    thinking_budget: Optional[object] = None,
    temperature: Optional[float] = None,
) -> Optional[dict]:
    """Call the Anthropic Messages API.

    Args:
        model_id: e.g. ``"claude-haiku-4-5-20251001"``.
        system: System prompt (a single string).
        messages: List of message dicts in Anthropic or OpenAI shape;
            sanitized internally.
        tools: Optional Anthropic-format tool list. The last tool is marked
            ``cache_control: ephemeral`` so the entire tools block is cached.
        max_tokens: Output token budget.
        api_key: Override; defaults to ``ANTHROPIC_API_KEY``.
        semaphore: Optional concurrency gate.
        max_retries: Retry budget. Retries on 429/500/502/503/529.
        thinking_budget: ``int`` for fixed budget, ``"adaptive"`` for adaptive
            thinking, ``None`` to disable.

    Returns:
        Parsed JSON dict, or ``None`` on permanent failure / exhausted retries.
    """
    if api_key is None:
        api_key = get_api_key("anthropic")
    if not api_key:
        logger.error("No Anthropic API key found")
        return None

    clean_messages = sanitize_messages(messages)
    if not clean_messages:
        logger.error("No valid messages after sanitization")
        return None

    # Defensive second pass: catch any orphaned tool_use blocks that slipped
    # through. Observed in production as 400 errors despite the orphan fixer.
    for di, dm in enumerate(clean_messages):
        if dm.get("role") != "assistant" or not isinstance(dm.get("content"), list):
            continue
        tu_ids = [
            b.get("id") for b in dm["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        if not tu_ids:
            continue
        nxt = clean_messages[di + 1] if di + 1 < len(clean_messages) else None
        result_ids: set = set()
        if nxt and nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
            result_ids = {
                b.get("tool_use_id") for b in nxt["content"]
                if isinstance(b, dict) and b.get("type") == "tool_result"
            }
        orphans = set(tu_ids) - result_ids
        if orphans:
            logger.error(
                f"ORPHAN_VALIDATOR: sanitize_messages left {len(orphans)} orphaned "
                f"tool_use blocks at clean_messages[{di}]: {orphans}. Stripping."
            )
            fixed_msg = copy.deepcopy(dm)
            fixed_msg["content"] = [
                b for b in fixed_msg["content"]
                if not (isinstance(b, dict)
                        and b.get("type") == "tool_use"
                        and b.get("id") in orphans)
            ]
            if not fixed_msg["content"]:
                fixed_msg["content"] = [{"type": "text", "text": "(continued)"}]
            clean_messages[di] = fixed_msg

    # Some models (Opus 4.6 family) don't support assistant-message prefill.
    # If sanitization left a trailing assistant message, append a nudge.
    if (clean_messages[-1].get("role") == "assistant"
            and model_id in _NO_PREFILL_MODELS):
        clean_messages = clean_messages + [{
            "role": "user",
            "content": "Please continue.",
        }]

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "anthropic-beta": "prompt-caching-2024-07-31",
    }

    # Prompt caching: mark the system block ephemeral so the system+tools
    # prefix is cached across calls. Empty text blocks must NOT carry
    # cache_control (the API rejects that).
    if system and system.strip():
        system_block = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
    else:
        system_block = []

    body: dict = {
        "model": model_id,
        "max_tokens": max_tokens,
        "system": system_block,
        "messages": clean_messages,
    }
    if tools:
        # Cache the full tools block by marking only the last tool ephemeral.
        cached_tools = [dict(t) for t in tools]
        cached_tools[-1] = dict(cached_tools[-1])
        cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
        body["tools"] = cached_tools
    if thinking_budget is not None:
        if thinking_budget == "adaptive":
            body["thinking"] = {"type": "adaptive"}
        else:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
    # Temperature must be 1.0 when thinking is enabled (Anthropic API constraint).
    if temperature is not None and thinking_budget is None:
        body["temperature"] = float(temperature)

    async def make_request() -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                # Thinking calls need longer timeouts — first byte is delayed.
                if thinking_budget is not None:
                    http_timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
                    wall_timeout = 600.0
                else:
                    http_timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
                    wall_timeout = 180.0
                async with httpx.AsyncClient() as client:
                    response = await asyncio.wait_for(
                        client.post(
                            "https://api.anthropic.com/v1/messages",
                            headers=headers,
                            json=body,
                            timeout=http_timeout,
                        ),
                        timeout=wall_timeout,
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 529):
                    wait_time = _get_retry_after(e.response)
                    if wait_time is None:
                        wait_time = _calculate_backoff(attempt, base=2.0, max_wait=60.0)
                        if e.response.status_code == 529:
                            wait_time = max(wait_time, 10.0)
                        elif e.response.status_code == 429:
                            wait_time = max(wait_time, 5.0)
                    if e.response.status_code == 429:
                        logger.warning(
                            f"429 rate limit on model={model_id[:40]} "
                            f"(attempt {attempt+1}/{max_retries}, wait {wait_time:.1f}s)"
                        )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Anthropic error {e.response.status_code}: {e.response.text[:200]}")
                    return None
            except (httpx.TimeoutException, asyncio.TimeoutError):
                wait_time = _calculate_backoff(attempt, base=2.0, max_wait=30.0)
                logger.warning(
                    f"Anthropic timeout on model={model_id[:40]} "
                    f"(attempt {attempt+1}, wait {wait_time:.1f}s)"
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Anthropic exception: {e}")
                wait_time = _calculate_backoff(attempt, base=2.0, max_wait=30.0)
                await asyncio.sleep(wait_time)
        logger.error(
            f"call_anthropic EXHAUSTED {max_retries} retries for model={model_id[:40]}"
        )
        return None

    if semaphore:
        async with semaphore:
            return await make_request()
    return await make_request()


# ---------------------------------------------------------------------------
# OpenRouter chat completions
# ---------------------------------------------------------------------------


async def call_openrouter(
    model_id: str,
    system: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = 4000,
    api_key: Optional[str] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_retries: int = 20,
    provider_order: Optional[list[str]] = None,
    allow_fallbacks: bool = True,
    temperature: Optional[float] = None,
) -> Optional[dict]:
    """Call OpenRouter's OpenAI-compatible chat-completions endpoint.

    Args:
        model_id: e.g. ``"x-ai/grok-4.1-fast"``.
        system: System prompt.
        messages: Anthropic-format messages; converted to OpenAI shape.
        tools: Anthropic-format tools; converted to OpenAI function-calling.
        max_tokens: Output budget.
        api_key: Override; defaults to ``OPENROUTER_API_KEY``.
        semaphore: Optional concurrency gate.
        max_retries: Retry budget. Retries on 403 (3 attempts), 429/500/502/503.
        provider_order: Optional preferred provider list.
        allow_fallbacks: If False, only the listed providers are used.
    """
    if api_key is None:
        api_key = get_api_key("openrouter")
    if not api_key:
        logger.error("No OpenRouter API key found")
        return None

    # Convert Anthropic-shaped messages to OpenAI shape, tracking pending
    # tool_call IDs so tool results can be emitted with the right role.
    oai_messages: list[dict] = [{"role": "system", "content": system}]
    pending_tool_call_ids: set = set()

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "") or ""
            tool_content = content if isinstance(content, str) else str(content)
            if pending_tool_call_ids and tool_call_id and tool_call_id in pending_tool_call_ids:
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content,
                })
                pending_tool_call_ids.discard(tool_call_id)
            else:
                oai_messages.append({"role": "user", "content": f"[Tool result: {tool_content}]"})
        elif isinstance(content, str):
            oai_msg: dict = {"role": role, "content": content}
            if role == "assistant" and msg.get("tool_calls"):
                valid_tcs = []
                for tc in msg["tool_calls"]:
                    tc_copy = dict(tc)
                    if not tc_copy.get("id"):
                        tc_copy["id"] = f"call_{len(valid_tcs)}"
                    valid_tcs.append(tc_copy)
                oai_msg["tool_calls"] = valid_tcs
                pending_tool_call_ids = {tc["id"] for tc in valid_tcs}
                if not oai_msg["content"]:
                    oai_msg["content"] = ""
            oai_messages.append(oai_msg)
        elif isinstance(content, list):
            text_parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
                    elif c.get("type") == "tool_result":
                        text_parts.append(c.get("content", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
            oai_messages.append({"role": role, "content": "\n".join(text_parts)})

    # Merge consecutive same-role messages so OpenAI's strict alternation is
    # not violated (only for user/assistant; system stays first).
    merged: list[dict] = []
    for msg in oai_messages:
        if (merged
                and msg["role"] == merged[-1]["role"]
                and msg["role"] in ("user", "assistant")):
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)
    oai_messages = merged

    # Convert tools to OpenAI function-calling format.
    oai_tools = None
    if tools:
        oai_tools = []
        for t in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body: dict = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": oai_messages,
        # Request reasoning summaries from reasoning models (e.g., Grok).
        "reasoning": {"enabled": True},
    }
    if temperature is not None:
        body["temperature"] = float(temperature)
    if oai_tools:
        body["tools"] = oai_tools
    if provider_order:
        body["provider"] = {
            "order": provider_order,
            "allow_fallbacks": allow_fallbacks,
        }

    async def make_request() -> Optional[dict]:
        import time as _time
        for attempt in range(max_retries):
            try:
                req_t0 = _time.time()
                async with httpx.AsyncClient() as client:
                    response = await asyncio.wait_for(
                        client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=body,
                            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
                        ),
                        timeout=300.0,
                    )
                    response.raise_for_status()
                    result = response.json()
                    req_dt = _time.time() - req_t0
                    usage = result.get("usage", {}) or {}
                    comp_tok = usage.get("completion_tokens", 0)
                    provider = result.get("provider", "")
                    tps = comp_tok / req_dt if req_dt > 0 and comp_tok > 0 else 0
                    if req_dt > 30:
                        logger.warning(
                            f"OpenRouter call: {req_dt:.1f}s, {comp_tok} tok, "
                            f"{tps:.1f} tok/s, provider={provider}"
                        )
                    return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    # 403 typically = spend limit; long backoff, only 3 attempts.
                    error_body = e.response.text[:200]
                    logger.error(f"OpenRouter error 403: {error_body}")
                    if attempt < 2:
                        wait_time = 30.0
                        logger.warning(
                            f"OpenRouter 403 (attempt {attempt+1}/3, wait {wait_time:.1f}s)"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("OpenRouter 403 after 3 attempts. Returning None.")
                        return None
                elif e.response.status_code in (429, 500, 502, 503):
                    wait_time = _get_retry_after(e.response)
                    if wait_time is None:
                        wait_time = _calculate_backoff(attempt, base=2.0, max_wait=60.0)
                        if e.response.status_code == 429:
                            wait_time = max(wait_time, 5.0)
                    logger.warning(
                        f"OpenRouter {e.response.status_code} "
                        f"(attempt {attempt+1}/{max_retries}, wait {wait_time:.1f}s)"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"OpenRouter error {e.response.status_code}: {e.response.text[:200]}"
                    )
                    return None
            except (httpx.TimeoutException, asyncio.TimeoutError):
                wait_time = _calculate_backoff(attempt, base=2.0, max_wait=30.0)
                logger.warning(
                    f"OpenRouter timeout (attempt {attempt+1}, wait {wait_time:.1f}s)"
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"OpenRouter exception: {e}")
                wait_time = _calculate_backoff(attempt, base=2.0, max_wait=30.0)
                await asyncio.sleep(wait_time)
        return None

    if semaphore:
        async with semaphore:
            return await make_request()
    return await make_request()


# ---------------------------------------------------------------------------
# OpenAI chat completions
# ---------------------------------------------------------------------------


async def call_openai(
    model_id: str,
    system: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = 4000,
    api_key: Optional[str] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_retries: int = 5,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Optional[dict]:
    """Call OpenAI's chat-completions endpoint directly.

    Args:
        model_id: e.g. ``"gpt-4.1-mini"``.
        system: System prompt.
        messages: Anthropic-format messages; converted to OpenAI shape.
        tools: Anthropic-format tools; converted to OpenAI function-calling.
        max_tokens: Output budget (sent as ``max_completion_tokens``).
        api_key: Override; defaults to ``OPENAI_API_KEY``.
        semaphore: Optional concurrency gate.
        max_retries: Retry budget. Retries on 429/500/502/503.
        reasoning_effort: Optional ``reasoning_effort`` body field. Stripped
            automatically when ``tools`` are present (GPT-5.4+ rejects the
            combination).
    """
    if api_key is None:
        api_key = get_api_key("openai")
    if not api_key:
        logger.error("No OpenAI API key found")
        return None

    oai_messages: list[dict] = [{"role": "system", "content": system}]
    pending_tool_call_ids: set = set()

    def _append_user_text(text) -> None:
        if text is None:
            return
        text_str = text if isinstance(text, str) else str(text)
        if text_str.strip():
            oai_messages.append({"role": "user", "content": text_str})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            if isinstance(content, str):
                if content.strip():
                    text_parts.append(content)
                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                        elif c.get("type") == "tool_use":
                            tc_id = c.get("id") or f"toolu_{len(tool_calls)}"
                            name = c.get("name") or "unknown_tool"
                            args = c.get("input") or {}
                            tool_calls.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            })
                    elif isinstance(c, str):
                        if c.strip():
                            text_parts.append(c)

            oai_msg: dict = {
                "role": "assistant",
                "content": "\n".join(t for t in text_parts if t),
            }
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
                if not oai_msg["content"]:
                    oai_msg["content"] = ""
                pending_tool_call_ids = {tc["id"] for tc in tool_calls}
            oai_messages.append(oai_msg)
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = content if isinstance(content, str) else str(content)
            if pending_tool_call_ids and tool_call_id in pending_tool_call_ids:
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content,
                })
                pending_tool_call_ids.discard(tool_call_id)
            else:
                _append_user_text(tool_content)
            continue

        if role == "user":
            if isinstance(content, str):
                _append_user_text(content)
            elif isinstance(content, list):
                text_parts = []
                tool_results: list[dict] = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "tool_result":
                            tool_results.append(c)
                        elif c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                    elif isinstance(c, str):
                        text_parts.append(c)
                for tr in tool_results:
                    tool_call_id = tr.get("tool_use_id", "")
                    tr_content = tr.get("content", "")
                    if pending_tool_call_ids and tool_call_id in pending_tool_call_ids:
                        oai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tr_content if isinstance(tr_content, str) else str(tr_content),
                        })
                        pending_tool_call_ids.discard(tool_call_id)
                    else:
                        text_parts.append(
                            tr_content if isinstance(tr_content, str) else str(tr_content)
                        )
                if text_parts:
                    _append_user_text("\n".join(t for t in text_parts if t))
            else:
                _append_user_text(content if isinstance(content, str) else str(content))
            continue

        # Fallback path: emit as user text.
        if isinstance(content, str):
            _append_user_text(content)
        elif isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    text_parts.append(c)
            _append_user_text("\n".join(text_parts))

    oai_tools = None
    if tools:
        oai_tools = []
        for t in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ``max_completion_tokens`` is forward-compatible with GPT-5+ and o-series.
    body: dict = {
        "model": model_id,
        "max_completion_tokens": max_tokens,
        "messages": oai_messages,
    }
    if oai_tools:
        body["tools"] = oai_tools
    if temperature is not None:
        body["temperature"] = float(temperature)
    if reasoning_effort is not None:
        # GPT-5.4+ rejects reasoning_effort when tools are present.
        if oai_tools:
            logger.debug(
                f"Stripping reasoning_effort={reasoning_effort} because tools are present"
            )
        else:
            body["reasoning_effort"] = reasoning_effort

    async def make_request() -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=body,
                        timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503):
                    wait_time = _get_retry_after(e.response)
                    if wait_time is None:
                        wait_time = _calculate_backoff(attempt, base=2.0, max_wait=60.0)
                        if e.response.status_code == 429:
                            wait_time = max(wait_time, 5.0)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"OpenAI error {e.response.status_code}: {e.response.text[:200]}"
                    )
                    return None
            except (httpx.TimeoutException, asyncio.TimeoutError):
                wait_time = _calculate_backoff(attempt, base=2.0, max_wait=30.0)
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"OpenAI exception: {e}")
                wait_time = _calculate_backoff(attempt, base=2.0, max_wait=30.0)
                await asyncio.sleep(wait_time)
        return None

    if semaphore:
        async with semaphore:
            return await make_request()
    return await make_request()


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic response adapter
# ---------------------------------------------------------------------------


def _convert_openai_response_to_anthropic(raw: Optional[dict]) -> Optional[dict]:
    """Adapt an OpenAI chat-completion response to Anthropic message shape.

    The audit loop expects::

        {"content": [{"type": "text", ...}, {"type": "tool_use", ...}],
         "stop_reason": "end_turn" | "tool_use" | "max_tokens"}
    """
    if not raw or not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if not choices:
        return None
    message = choices[0].get("message", {}) or {}
    finish_reason = choices[0].get("finish_reason", "stop")

    content_blocks: list[dict] = []

    text = message.get("content") or ""
    if text.strip():
        content_blocks.append({"type": "text", "text": text})

    tool_calls_raw = message.get("tool_calls") or []
    for tc in tool_calls_raw:
        func = tc.get("function", {}) or {}
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}
        if name:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": name,
                "input": args if isinstance(args, dict) else {},
            })

    if tool_calls_raw:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    return {"content": content_blocks, "stop_reason": stop_reason}


# ---------------------------------------------------------------------------
# APIClients: routing + per-role concurrency
# ---------------------------------------------------------------------------


# Heuristic: GPT-family target models that should route through OpenAI direct
# even when no ``openai/`` prefix is given (mirrors paper-time auto-detection).
_OPENAI_FAMILY_PREFIXES = ("gpt-", "o1", "o3", "o4")


def _is_openai_target(model_id: str) -> bool:
    low = model_id.lower()
    return any(low.startswith(p) for p in _OPENAI_FAMILY_PREFIXES)


class APIClients:
    """Three-semaphore client wrapper with prefix-based routing.

    Routing rules:

    - ``openrouter/<model>`` → :func:`call_openrouter`
    - ``openai/<model>`` (or bare ``gpt-*``/``o*``) → :func:`call_openai`
      then :func:`_convert_openai_response_to_anthropic`
    - default → :func:`call_anthropic`

    Three independent semaphores control concurrency for the auditor, target,
    and judge roles. Target Anthropic calls are converted to OpenAI-shaped
    responses so callers can treat all targets uniformly downstream.
    """

    def __init__(
        self,
        auditor_concurrent: int = DEFAULT_AUDITOR_CONCURRENT,
        target_concurrent: int = DEFAULT_TARGET_CONCURRENT,
        judge_concurrent: int = DEFAULT_JUDGE_CONCURRENT,
        use_openai_for_target: bool = False,
        provider_order: Optional[list[str]] = None,
    ):
        self.auditor_key = get_api_key("anthropic")
        self.judge_key = get_api_key("anthropic")
        self.openrouter_key = get_api_key("openrouter")
        self.openai_key = get_api_key("openai")

        self.use_openai_for_target = use_openai_for_target
        self.provider_order = provider_order

        self.auditor_semaphore = asyncio.Semaphore(auditor_concurrent)
        self.target_semaphore = asyncio.Semaphore(target_concurrent)
        self.judge_semaphore = asyncio.Semaphore(judge_concurrent)

        if not self.auditor_key:
            logger.warning(
                "No Anthropic API key found for auditor — auditor calls may fail"
            )
        if not self.judge_key:
            logger.warning(
                "No Anthropic API key found for judge — judging calls may fail"
            )
        if not self.openrouter_key:
            logger.warning(
                "No OpenRouter API key found — OpenRouter target calls may fail"
            )
        if not self.openai_key:
            logger.warning(
                "No OpenAI API key found — OpenAI target calls may fail"
            )

        logger.info(
            f"APIClients initialized: auditor={auditor_concurrent}, "
            f"target={target_concurrent}, judge={judge_concurrent}"
        )

    async def call_auditor(
        self,
        model_id: str,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4000,
        temperature: Optional[float] = None,
    ) -> Optional[dict]:
        """Call the auditor model, routing by model_id prefix."""
        if model_id.startswith("openrouter/"):
            or_model = model_id[len("openrouter/"):]
            raw = await call_openrouter(
                model_id=or_model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                api_key=self.openrouter_key,
                semaphore=self.auditor_semaphore,
                provider_order=self.provider_order,
                temperature=temperature,
            )
            return _convert_openai_response_to_anthropic(raw)
        if model_id.startswith("openai/") or _is_openai_target(model_id):
            oai_model = model_id[len("openai/"):] if model_id.startswith("openai/") else model_id
            reasoning_effort = None
            if ":reasoning=" in oai_model:
                oai_model, reasoning_effort = oai_model.rsplit(":reasoning=", 1)
            raw = await call_openai(
                model_id=oai_model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                api_key=self.openai_key,
                semaphore=self.auditor_semaphore,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )
            return _convert_openai_response_to_anthropic(raw)
        return await call_anthropic(
            model_id=model_id,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            api_key=self.auditor_key,
            semaphore=self.auditor_semaphore,
            temperature=temperature,
        )

    async def call_judge(
        self,
        model_id: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 16000,
        thinking_budget: Optional[object] = 10_000,
    ) -> Optional[dict]:
        """Call the judge model (always Anthropic, with thinking by default)."""
        # Ensure max_tokens > thinking_budget when thinking is enabled.
        if isinstance(thinking_budget, int) and max_tokens <= thinking_budget:
            max_tokens = thinking_budget + 6000
        if thinking_budget == "adaptive":
            max_tokens = max(max_tokens, 16000)
        return await call_anthropic(
            model_id=model_id,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            api_key=self.judge_key,
            semaphore=self.judge_semaphore,
            thinking_budget=thinking_budget,
        )

    async def call_target(
        self,
        model_id: str,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4000,
        provider_order: Optional[list[str]] = None,
        allow_fallbacks: bool = True,
    ) -> Optional[dict]:
        """Call the target model, routing by prefix.

        Returns an OpenAI-compatible chat-completions dict so downstream code
        can read ``choices[0].message`` uniformly regardless of provider.
        """
        clean_id = model_id
        if clean_id.startswith("openrouter/"):
            clean_id = clean_id[len("openrouter/"):]
        elif clean_id.startswith("openai/"):
            clean_id = clean_id[len("openai/"):]
        elif clean_id.startswith("anthropic/"):
            clean_id = clean_id[len("anthropic/"):]

        # Anthropic targets — call Anthropic, then convert to OpenAI shape.
        if (model_id.startswith("anthropic/")
                or (clean_id.startswith("claude-") and not self.use_openai_for_target)):
            response = await call_anthropic(
                model_id=clean_id,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                api_key=self.auditor_key,
                semaphore=self.target_semaphore,
            )
            if response and isinstance(response, dict) and "content" in response:
                text = ""
                tool_calls: list[dict] = []
                for block in response.get("content", []):
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text += block.get("text", "")
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            })
                msg: dict = {"role": "assistant", "content": text}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                return {"choices": [{"message": msg}]}
            return response

        # OpenAI direct (either explicit prefix already stripped, or flag set).
        if self.use_openai_for_target or model_id.startswith("openai/") or _is_openai_target(clean_id):
            return await call_openai(
                model_id=clean_id,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                api_key=self.openai_key,
                semaphore=self.target_semaphore,
            )

        # Default: OpenRouter.
        effective_provider_order = provider_order or self.provider_order
        return await call_openrouter(
            model_id=clean_id,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            api_key=self.openrouter_key,
            semaphore=self.target_semaphore,
            provider_order=effective_provider_order,
            allow_fallbacks=allow_fallbacks,
        )
