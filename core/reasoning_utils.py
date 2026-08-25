"""
Reasoning content formatting (Petri v2 style).

Used by ``core/petri.py`` to render assistant reasoning in transcripts.
``core/audit.py`` has its own ``strip_encrypted_thinking`` for the audit
loop's transcript-cleanup pass; these two helpers are intentionally
distinct and not wired together.
"""

from typing import Optional


def is_encrypted_text(text: str, sample_size: int = 100) -> bool:
    """Heuristic: returns True for base64-like gibberish (no spaces).

    Encrypted thinking from providers like OpenRouter/Grok shows up as
    long contiguous strings without spaces. Real text typically has
    >10% space ratio; encrypted content has <2%.
    """
    if not text or len(text) < 20:
        return False
    sample = text[:sample_size]
    space_ratio = sample.count(' ') / len(sample) if sample else 1
    return space_ratio < 0.02


def format_reasoning_content(
    reasoning: Optional[str],
    summary: Optional[str] = None,
    redacted: bool = False,
) -> str:
    """Format reasoning content following Petri v2's approach.

    Priority:
      1. ``redacted=True`` and ``summary`` present → ``<thinking_summary>...</thinking_summary>``
      2. ``redacted=True`` and no summary → ``<thinking_redacted/>``
      3. ``reasoning`` looks encrypted (heuristic) → ``<thinking_redacted/>``
      4. otherwise → ``<thinking>...</thinking>``
    """
    if redacted:
        if summary and summary.strip():
            return f"<thinking_summary>\n{summary.strip()}\n</thinking_summary>"
        return "<thinking_redacted/>"
    if not reasoning:
        return ""
    if is_encrypted_text(reasoning):
        return "<thinking_redacted/>"
    return f"<thinking>\n{reasoning}\n</thinking>"


__all__ = ["format_reasoning_content", "is_encrypted_text"]
