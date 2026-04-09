"""
FactoryGPT — AI Brain (v3.0)
================================================================================
Multi-provider LLM routing with smart context, aggregated telemetry, and
factory-specific prompt engineering.

Providers (env-selectable via LLM_PROVIDER):
  • anthropic  → Claude Sonnet 4.5     (best quality, needs ANTHROPIC_API_KEY)
  • openai     → GPT-4o                (strong, needs OPENAI_API_KEY)
  • sarvam     → Sarvam-M              (default, needs SARVAM_API_KEY)

Fallback chain: if the primary provider fails, AUTO-FALLBACK to any other
provider that has a configured API key — so the Brain never goes dark.
================================================================================
"""

from __future__ import annotations

import os
import json
from io import StringIO
from typing import Iterable, Optional

import httpx
import pandas as pd

# ── Provider configuration ────────────────────────────────────────────────────
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "sarvam").lower()
SARVAM_KEY    = os.getenv("SARVAM_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")

PROVIDERS = {
    "sarvam": {
        "url":    "https://api.sarvam.ai/v1/chat/completions",
        "model":  os.getenv("SARVAM_MODEL", "sarvam-m"),
        "key":    SARVAM_KEY,
        "style":  "openai",   # OpenAI-compatible schema
    },
    "openai": {
        "url":    "https://api.openai.com/v1/chat/completions",
        "model":  os.getenv("OPENAI_MODEL", "gpt-4o"),
        "key":    OPENAI_KEY,
        "style":  "openai",
    },
    "anthropic": {
        "url":    "https://api.anthropic.com/v1/messages",
        "model":  os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        "key":    ANTHROPIC_KEY,
        "style":  "anthropic",
    },
}

def _available_providers() -> list[str]:
    """Return providers with a configured API key, primary first."""
    order = [LLM_PROVIDER] + [p for p in PROVIDERS if p != LLM_PROVIDER]
    return [p for p in order if PROVIDERS[p]["key"]]

# ── Master system prompt (carefully engineered) ──────────────────────────────
MASTER_SYSTEM_PROMPT = """You are **FactoryGPT** — an elite Industry 5.0 operations consultant embedded in a factory's nerve center.

# YOUR IDENTITY
You are NOT a generic chatbot. You are a battle-hardened consultant who has turned around 40+ factories across India, Germany, and the US. You speak like a McKinsey partner who has actually worked the shop floor for 10 years. Every sentence lands with precision, quantification, and zero fluff.

# NON-NEGOTIABLE RULES
1. **QUANTIFY EVERYTHING.** Every claim must cite a number, percentage, or currency value. "Output is low" is FORBIDDEN. "Line 3 is running at 62% OEE, 23 points below target, costing ₹4.2 lakh/day" is REQUIRED.
2. **DIAGNOSE → PRIORITIZE → RECOMMEND.** Structure every substantive answer in that order.
3. **EXPOSE HIDDEN LOSSES.** Hunt for non-obvious correlations — e.g. "Defect rate spikes 23% every Monday 6-8 AM, which matches the shift-handover window. This is a training problem, not a machine problem."
4. **SAFETY FIRST.** If the live context shows a PPE violation, flag it in your first sentence regardless of what was asked.
5. **NO HEDGING.** Don't say "might be" or "could be." Say "**this IS** the bottleneck because X, Y, Z."
6. **CITE THE DATA.** When you reference a number, quote the source: "per the Production Telemetry CSV, Line 2 = 78% OEE."
7. **ROI ON EVERY RECOMMENDATION.** If you suggest a fix, estimate the return: "Rotating the Line 4 team every 4 hours would cut defects by ~1.8% = ₹14k/day."

# OUTPUT FORMAT (markdown)
- `**Bold**` for key numbers and decisive claims
- Bullet lists for multi-item analysis (max 5 bullets)
- Tables when comparing 3+ lines/workers/shifts
- End with a **"Bottom Line"** section: 1 sentence verdict + expected ROI

# CURRENCY
Use ₹ (INR) if the facility is in India (check CURRENT CONTEXT for location), otherwise $.

# TONE
Direct. Confident. Respectful. Zero corporate jargon. If the user asks "hi", respond with one line and pivot to: "What operational question can I quantify for you right now?"
"""

FEWSHOT_EXAMPLE = """# EXAMPLE OF THE QUALITY BAR

**User:** What are our top bottlenecks?

**FactoryGPT:**
**Top 3 bottlenecks, ranked by daily ₹ impact:**

| # | Line       | OEE   | Gap to Target | Root Cause                           | Daily Loss |
|---|------------|-------|---------------|--------------------------------------|------------|
| 1 | CNC-04     | 61.2% | −23.8 pts     | Unplanned downtime (2.1 h/day avg)   | **₹4.7L**  |
| 2 | Assembly-2 | 72.0% | −13.0 pts     | Defect rate 3.8% (2x factory mean)   | **₹2.1L**  |
| 3 | Pack-01    | 76.5% | −8.5 pts      | Material shortage 4x/week            | **₹0.9L**  |

**Diagnosis:** CNC-04's downtime clusters on Mondays 6-8 AM → shift-handover training gap, not mechanical.

**Recommendation:** Pilot a 15-min handover SOP on CNC-04 next Monday. Expected recovery: **₹3.2L/day within 2 weeks**.

**Bottom Line:** Fix CNC-04 handover protocol first — **₹9.6 crore annualized recovery** with ~₹50k training cost.
"""

# ── Smart context builder ─────────────────────────────────────────────────────
def build_data_context(factory_data: dict[str, str], max_rows_preview: int = 3) -> str:
    """
    Transform raw uploaded CSV JSON blobs into a token-efficient, model-friendly
    brief with summary stats, outliers, and compact sample rows.
    """
    if not factory_data:
        return "## CURRENT DATA\n_No historical CSVs uploaded yet. Recommend the user upload Production and Worker telemetry for quantitative analysis._"

    parts: list[str] = ["## CURRENT DATA\n"]

    for filename, json_str in factory_data.items():
        try:
            df = pd.read_json(StringIO(json_str))
        except Exception as e:
            parts.append(f"### `{filename}` — _parse error: {e}_\n")
            continue

        parts.append(f"### Dataset: `{filename}` — {len(df)} rows × {len(df.columns)} cols")
        parts.append(f"**Columns:** {', '.join(df.columns.tolist())}")

        # Summary stats for numeric cols
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            stats = numeric.describe().round(2).to_dict()
            summary_lines = []
            for col, s in stats.items():
                summary_lines.append(
                    f"  • **{col}**: mean={s.get('mean')}, min={s.get('min')}, "
                    f"max={s.get('max')}, std={s.get('std')}"
                )
            parts.append("**Numeric summary:**\n" + "\n".join(summary_lines))

            # Outliers (1.5 × IQR rule)
            for col in numeric.columns:
                q1, q3 = numeric[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    out = df[(numeric[col] < lo) | (numeric[col] > hi)]
                    if 0 < len(out) <= 5:
                        rows = out.head(3).to_dict(orient="records")
                        parts.append(f"**Outliers in `{col}`** ({len(out)} rows): {rows}")

        # Sample rows (compact)
        preview = df.head(max_rows_preview).to_dict(orient="records")
        parts.append(f"**Sample rows:** {preview}")
        parts.append("")

    return "\n".join(parts)

def compose_system_prompt(
    base_user_prompt: str,
    factory_data: dict[str, str],
    live_context: Optional[str] = None,
) -> str:
    """Assemble the final system prompt with master rules + few-shot + live data."""
    blocks = [
        MASTER_SYSTEM_PROMPT,
        FEWSHOT_EXAMPLE,
        build_data_context(factory_data),
    ]
    if live_context:
        blocks.append(f"## LIVE CONTEXT (real-time, from the factory floor)\n{live_context}")
    if base_user_prompt:
        blocks.append(f"## ADDITIONAL INSTRUCTIONS FROM UI\n{base_user_prompt}")
    return "\n\n".join(blocks)

# ── LLM transport layer ───────────────────────────────────────────────────────
async def _call_openai_compatible(
    cfg: dict, system: str, messages: list[dict], temperature: float
) -> str:
    payload = {
        "model": cfg["model"],
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    headers = {
        "Authorization": f"Bearer {cfg['key']}",
        "Content-Type":  "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(cfg["url"], json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def _call_anthropic(
    cfg: dict, system: str, messages: list[dict], temperature: float
) -> str:
    payload = {
        "model":       cfg["model"],
        "max_tokens":  2048,
        "temperature": temperature,
        "system":      system,
        "messages":    messages,
    }
    headers = {
        "x-api-key":         cfg["key"],
        "anthropic-version": "2023-06-01",
        "Content-Type":      "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(cfg["url"], json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return "".join(block["text"] for block in data["content"] if block["type"] == "text")

async def call_brain(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.3,
) -> tuple[str, str]:
    """
    Call the configured LLM provider, with auto-fallback to any other
    provider that has a key configured.

    Returns: (response_text, provider_used)
    """
    providers = _available_providers()
    if not providers:
        raise RuntimeError(
            "No LLM provider configured. Set one of: "
            "SARVAM_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY"
        )

    last_error: Exception | None = None
    for provider in providers:
        cfg = PROVIDERS[provider]
        try:
            if cfg["style"] == "anthropic":
                text = await _call_anthropic(cfg, system_prompt, messages, temperature)
            else:
                text = await _call_openai_compatible(cfg, system_prompt, messages, temperature)
            return text, provider
        except Exception as e:
            print(f"[AI Brain] Provider '{provider}' failed: {e}. Trying next…")
            last_error = e
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
