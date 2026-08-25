"""
Petri integration utilities.

Single source of truth for all Petri integration:
- System prompts (``AUDITOR_SYSTEM_PROMPT``)
- User messages (``AUDITOR_USER_MESSAGE``)
- Tool definitions (the 6 auditor tools with full JSON schemas)
- Message format conversion (XML rendering for the judge)

Usage::

    from core.petri import (
        get_auditor_system_prompt,
        get_auditor_user_message,
        get_auditor_tools_async,
        construct_auditor_system_prompt,
        construct_auditor_user_message,
        format_messages_as_xml,
        get_judge_dimensions,
        get_judge_prompt,
    )
"""

import json
from datetime import datetime
from typing import Any

from core.reasoning_utils import format_reasoning_content

# Default max turns for auditor
DEFAULT_MAX_TURNS = 15


def get_auditor_system_prompt() -> str:
    """Get the raw AUDITOR_SYSTEM_PROMPT template from Petri.

    This is the unformatted template with placeholders:
    - {target_model}
    - {seed_instructions}
    - {max_turns}
    - {today_date}
    - {resources_prompt}
    """
    from petri.solvers.prompts import AUDITOR_SYSTEM_PROMPT
    return AUDITOR_SYSTEM_PROMPT


def get_auditor_user_message() -> str:
    """Get the raw AUDITOR_USER_MESSAGE template from Petri.

    This is the unformatted template with placeholders:
    - {target_model}
    - {seed_instructions}
    """
    from petri.solvers.prompts import AUDITOR_USER_MESSAGE
    return AUDITOR_USER_MESSAGE


def construct_auditor_system_prompt(
    target_model: str,
    seed_instructions: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    today_date: str | None = None,
) -> str:
    """Construct a fully formatted auditor system prompt.

    Args:
        target_model: Name of the target model (e.g., "x-ai/grok-4.1-fast")
        seed_instructions: The audit seed instructions
        max_turns: Maximum conversation turns (default: 15)
        today_date: Current date for realistic interactions (default: today)

    Returns:
        Formatted system prompt string
    """
    if today_date is None:
        today_date = datetime.now().strftime("%Y-%m-%d")

    return get_auditor_system_prompt().format(
        target_model=target_model,
        seed_instructions=seed_instructions,
        max_turns=max_turns,
        today_date=today_date,
        resources_prompt="",
    )


def construct_auditor_user_message(
    target_model: str,
    seed_instructions: str,
) -> str:
    """Construct a fully formatted first user message.

    Args:
        target_model: Name of the target model
        seed_instructions: The audit seed instructions

    Returns:
        Formatted user message string
    """
    return get_auditor_user_message().format(
        target_model=target_model,
        seed_instructions=seed_instructions,
    )


def _convert_param_to_json_schema(param_info) -> dict:
    """Convert an inspect_ai parameter to JSON Schema format."""
    result = {}

    if hasattr(param_info, 'anyOf') and param_info.anyOf:
        types = []
        for sub in param_info.anyOf:
            if sub.type:
                types.append(sub.type)

        if 'string' in types and 'null' in types:
            result["type"] = ["string", "null"]
        elif len(types) == 1:
            result["type"] = types[0]
        else:
            result["anyOf"] = [{"type": t} for t in types if t]
    elif param_info.type:
        result["type"] = param_info.type
    else:
        result["type"] = "string"

    if hasattr(param_info, 'description') and param_info.description:
        result["description"] = param_info.description

    if hasattr(param_info, 'enum') and param_info.enum:
        result["enum"] = param_info.enum

    return result


async def get_auditor_tools_async() -> list[dict]:
    """Get the 6 auditor tools with full descriptions from Petri (async).

    Returns tools in Anthropic API format with proper JSON schemas.
    """
    from petri.tools.tools import default_auditor_tools
    from inspect_ai.tool._tool_def import tool_defs

    tools = default_auditor_tools(instance=None, prefill=True)
    tool_definitions = await tool_defs(tools)

    anthropic_tools = []
    for tool_def in tool_definitions:
        properties = {}
        required = []

        if tool_def.parameters and tool_def.parameters.properties:
            for param_name, param_info in tool_def.parameters.properties.items():
                properties[param_name] = _convert_param_to_json_schema(param_info)

            if tool_def.parameters.required:
                required = tool_def.parameters.required

        anthropic_tools.append({
            "name": tool_def.name,
            "description": tool_def.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })

    return anthropic_tools


def convert_content_to_anthropic(content: Any) -> str | list[dict]:
    """Convert inspect_ai content to Anthropic API format.

    Handles reasoning content using Petri v2's approach:
    - Check redacted flag first
    - Use summary if available for redacted content
    - Fall back to heuristic detection for improperly-flagged content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, str):
                result.append({"type": "text", "text": item})
            elif hasattr(item, 'text'):
                result.append({"type": "text", "text": item.text})
            elif hasattr(item, 'reasoning'):
                # Handle ContentReasoning objects (Petri v2 style)
                reasoning = item.reasoning
                summary = getattr(item, 'summary', None)
                redacted = getattr(item, 'redacted', False)

                # Format using Petri v2 approach with fallback
                formatted = format_reasoning_content(reasoning, summary, redacted)
                if formatted:
                    result.append({"type": "text", "text": formatted})
            else:
                result.append({"type": "text", "text": str(item)})

        if len(result) == 1 and result[0]["type"] == "text":
            return result[0]["text"]
        return result

    return str(content)


def format_content_for_xml(content: Any) -> str:
    """Format message content for XML transcript output."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "tool_use":
                    parts.append(f"[Tool Use: {item.get('name', 'unknown')} (id: {item.get('id', 'unknown')})]")
                elif item_type == "tool_result":
                    parts.append(item.get("content", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    return str(content)


def format_tool_call_xml(tool_call: dict) -> str:
    """Format a tool call as XML."""
    lines = []
    name = tool_call.get("name", "unknown")
    tc_id = tool_call.get("id", "unknown")

    lines.append(f'<tool_call name="{name}" id="{tc_id}">')

    arguments = tool_call.get("input", {})
    for param_name, param_value in arguments.items():
        if isinstance(param_value, str):
            value_str = param_value
        elif isinstance(param_value, (dict, list)):
            value_str = json.dumps(param_value)
        else:
            value_str = str(param_value)
        lines.append(f'  <parameter name="{param_name}">{value_str}</parameter>')

    lines.append("</tool_call>")
    return "\n".join(lines)


def _is_prefill_message(msg: dict) -> bool:
    """Check if a message dict is a prefill message."""
    metadata = msg.get("metadata")
    return metadata is not None and metadata.get("prefill") is True


def format_messages_as_xml(messages: list[dict]) -> tuple[str, dict[str, int]]:
    """Format messages as XML transcript matching Petri's format_messages_xml.

    Args:
        messages: List of message dicts in Anthropic format

    Returns:
        tuple: (formatted_xml, index_to_position_mapping)
    """
    formatted_parts = []
    index_to_position = {}
    message_index = 1

    for msg in messages:
        role = msg.get("role", "unknown")

        # Skip info messages
        if role == "info":
            content_text = format_content_for_xml(msg.get("content", ""))
            formatted_parts.append(f"<info>\n{content_text}\n</info>")
            continue

        # Index this message
        index_to_position[str(message_index)] = len(formatted_parts)

        # Check if this is a prefill message
        is_prefill = _is_prefill_message(msg)
        tag = "prefill" if is_prefill else role

        lines = []
        lines.append(f'<{tag} index="{message_index}">')

        # Add main content
        content = msg.get("content", "")
        content_text = format_content_for_xml(content)
        if content_text:
            lines.append(content_text)

        # Add tool calls for assistant messages (not for prefill)
        if role == "assistant" and not is_prefill and isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    lines.append(format_tool_call_xml(item))

        lines.append(f"</{tag}>")
        formatted_parts.append("\n".join(lines))
        message_index += 1

    transcript = "<transcript>\n" + "\n\n".join(formatted_parts) + "\n</transcript>"
    return transcript, index_to_position


def get_judge_dimensions() -> dict[str, str]:
    """Get the full DIMENSIONS dict from Petri with all 37 metric descriptions."""
    from petri.scorers.prompts import DIMENSIONS
    return DIMENSIONS


def get_judge_prompt() -> str:
    """Get the full JUDGE_PROMPT template from Petri."""
    from petri.scorers.prompts import JUDGE_PROMPT
    return JUDGE_PROMPT
