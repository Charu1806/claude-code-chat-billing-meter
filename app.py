"""
AI Token Cost Explorer — Streamlit App
• Tabs 1 & 2: Live API calls — Anthropic, Google (Gemini), or Groq (Mistral/Llama)
• Tabs 3, 4, 5: Pure cost math — all providers, no API keys required
"""

import json
import os
import anthropic
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Live model registry — models available for actual API calls ───────────────
# sdk: "anthropic" | "openai_compat"
# For openai_compat: base_url routes to the right provider endpoint
LIVE_MODELS = {
    # Anthropic
    "claude-opus-4-8": {
        "label": "Claude Opus 4.8",
        "provider": "Anthropic",
        "sdk": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "supports_thinking": True,
        "supports_tools": True,
        "context_window": 1_000_000,
    },
    "claude-sonnet-4-6": {
        "label": "Claude Sonnet 4.6",
        "provider": "Anthropic",
        "sdk": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "supports_thinking": True,
        "supports_tools": True,
        "context_window": 1_000_000,
    },
    "claude-haiku-4-5": {
        "label": "Claude Haiku 4.5",
        "provider": "Anthropic",
        "sdk": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "supports_thinking": False,
        "supports_tools": True,
        "context_window": 200_000,
    },
    # Google Gemini — via OpenAI-compatible endpoint
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash",
        "provider": "Google",
        "sdk": "openai_compat",
        "key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_model": "gemini-2.5-flash",
        "supports_thinking": False,
        "supports_tools": True,
        "context_window": 1_000_000,
    },
    "gemini-2.5-pro": {
        "label": "Gemini 2.5 Pro",
        "provider": "Google",
        "sdk": "openai_compat",
        "key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_model": "gemini-2.5-pro",
        "supports_thinking": False,
        "supports_tools": True,
        "context_window": 1_000_000,
    },
    # Groq — Mistral
    "mistral-saba": {
        "label": "Mistral Saba (via Groq)",
        "provider": "Groq / Mistral",
        "sdk": "openai_compat",
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "api_model": "mistral-saba-24b",
        "supports_thinking": False,
        "supports_tools": True,
        "context_window": 32_768,
    },
    # Groq — Meta Llama
    "llama-3.3-70b": {
        "label": "Llama 3.3 70B (via Groq)",
        "provider": "Groq / Meta",
        "sdk": "openai_compat",
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "api_model": "llama-3.3-70b-versatile",
        "supports_thinking": False,
        "supports_tools": True,
        "context_window": 128_000,
    },
    "llama-3.1-8b": {
        "label": "Llama 3.1 8B (via Groq)",
        "provider": "Groq / Meta",
        "sdk": "openai_compat",
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "api_model": "llama-3.1-8b-instant",
        "supports_thinking": False,
        "supports_tools": True,
        "context_window": 128_000,
    },
}

# Group order for the selector
LIVE_MODEL_GROUPS = {
    "Anthropic": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "Google": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "Groq / Mistral": ["mistral-saba"],
    "Groq / Meta": ["llama-3.3-70b", "llama-3.1-8b"],
}

# ── Model pricing config (USD per 1M tokens) — used for cost calculator tabs ──
MODEL_PRICING = {
    "claude-opus-4-8":   {"name": "Claude Opus 4.8",    "provider": "Anthropic",      "input": 5.00,  "output": 25.00, "context_window": 1_000_000, "notes": "Most capable Anthropic model"},
    "claude-sonnet-4-6": {"name": "Claude Sonnet 4.6",  "provider": "Anthropic",      "input": 3.00,  "output": 15.00, "context_window": 1_000_000, "notes": "Best speed/intelligence balance"},
    "claude-haiku-4-5":  {"name": "Claude Haiku 4.5",   "provider": "Anthropic",      "input": 1.00,  "output": 5.00,  "context_window": 200_000,   "notes": "Fastest & cheapest Anthropic"},
    "gemini-2.5-flash":  {"name": "Gemini 2.5 Flash",   "provider": "Google",         "input": 0.15,  "output": 0.60,  "context_window": 1_000_000, "notes": "Best Google price/perf"},
    "gemini-2.5-pro":    {"name": "Gemini 2.5 Pro",     "provider": "Google",         "input": 1.25,  "output": 10.00, "context_window": 1_000_000, "notes": "Top Google model"},
    "gemini-1.5-flash":  {"name": "Gemini 1.5 Flash",   "provider": "Google",         "input": 0.075, "output": 0.30,  "context_window": 1_000_000, "notes": "Very cheap, high volume"},
    "mistral-large-2":   {"name": "Mistral Large 2",    "provider": "Mistral",        "input": 2.00,  "output": 6.00,  "context_window": 128_000,   "notes": "Flagship Mistral model"},
    "mistral-small-3.1": {"name": "Mistral Small 3.1",  "provider": "Mistral",        "input": 0.10,  "output": 0.30,  "context_window": 128_000,   "notes": "Low-cost, multilingual"},
    "mistral-nemo":      {"name": "Mistral Nemo",       "provider": "Mistral",        "input": 0.15,  "output": 0.15,  "context_window": 128_000,   "notes": "12B model, Apache 2"},
    "codestral":         {"name": "Codestral",          "provider": "Mistral",        "input": 0.30,  "output": 0.90,  "context_window": 256_000,   "notes": "Code specialist"},
    "grok-3":            {"name": "Grok 3",             "provider": "xAI",            "input": 3.00,  "output": 15.00, "context_window": 131_072,   "notes": "xAI flagship"},
    "grok-3-mini":       {"name": "Grok 3 Mini",        "provider": "xAI",            "input": 0.30,  "output": 0.50,  "context_window": 131_072,   "notes": "Fast & cheap xAI"},
    "grok-2":            {"name": "Grok 2",             "provider": "xAI",            "input": 2.00,  "output": 10.00, "context_window": 131_072,   "notes": "Previous xAI flagship"},
    "gpt-4o":            {"name": "GPT-4o",             "provider": "OpenAI",         "input": 2.50,  "output": 10.00, "context_window": 128_000,   "notes": "OpenAI flagship"},
    "gpt-4o-mini":       {"name": "GPT-4o mini",        "provider": "OpenAI",         "input": 0.15,  "output": 0.60,  "context_window": 128_000,   "notes": "Cheap & fast OpenAI"},
    "o3":                {"name": "o3",                 "provider": "OpenAI",         "input": 2.00,  "output": 8.00,  "context_window": 200_000,   "notes": "OpenAI reasoning model"},
    "o4-mini":           {"name": "o4-mini",            "provider": "OpenAI",         "input": 1.10,  "output": 4.40,  "context_window": 200_000,   "notes": "Fast OpenAI reasoning"},
    "llama-3.3-70b":     {"name": "Llama 3.3 70B",      "provider": "Meta (via API)", "input": 0.23,  "output": 0.40,  "context_window": 128_000,   "notes": "Via Groq / Together"},
    "llama-3.1-405b":    {"name": "Llama 3.1 405B",     "provider": "Meta (via API)", "input": 0.80,  "output": 0.80,  "context_window": 128_000,   "notes": "Largest open Llama"},
    "command-r-plus":    {"name": "Command R+",         "provider": "Cohere",         "input": 2.50,  "output": 10.00, "context_window": 128_000,   "notes": "Cohere flagship RAG model"},
    "command-r":         {"name": "Command R",          "provider": "Cohere",         "input": 0.15,  "output": 0.60,  "context_window": 128_000,   "notes": "Cohere efficient model"},
}

PROVIDER_COLORS = {
    "Anthropic":      "#e07b39",
    "Google":         "#4285F4",
    "Mistral":        "#FF7000",
    "xAI":            "#1DA1F2",
    "OpenAI":         "#10A37F",
    "Meta (via API)": "#0668E1",
    "Cohere":         "#39C5BB",
    "Groq / Mistral": "#FF7000",
    "Groq / Meta":    "#0668E1",
}

ALL_PROVIDERS = sorted({v["provider"] for v in MODEL_PRICING.values()})

RAG_INFRA = {
    "vector_db_monthly":        70,
    "embedding_per_1m_tokens":  0.02,
    "avg_chunks_retrieved":     5,
    "avg_chunk_tokens":         300,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def token_cost(tokens: int, price_per_million: float) -> float:
    return tokens * price_per_million / 1_000_000


def calc_cost(input_tok: int, output_tok: int, model: str):
    p = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    ic = token_cost(input_tok, p["input"])
    oc = token_cost(output_tok, p["output"])
    return ic, oc, ic + oc


def fmt_cost(v: float) -> str:
    if v == 0:
        return "$0.00"
    if v < 0.0001:
        return f"${v:.6f}"
    if v < 0.01:
        return f"${v:.4f}"
    return f"${v:.4f}"


def key_status(model_key: str) -> tuple[bool, str]:
    """Return (key_present, key_value_or_empty) for the model's required env var."""
    info = LIVE_MODELS[model_key]
    val = os.getenv(info["key_env"], "")
    ok = bool(val) and not val.startswith("sk-ant-...") and not val.endswith("...")
    return ok, val


def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def get_openai_compat_client(model_key: str) -> OpenAI:
    info = LIVE_MODELS[model_key]
    return OpenAI(
        api_key=os.getenv(info["key_env"], "missing"),
        base_url=info["base_url"],
    )


# ── Sidebar: active model selector (shared across Tab 1 & 2) ─────────────────

def render_model_selector() -> str:
    """Render the sidebar model selector and return the selected model key."""
    st.sidebar.markdown("## 🤖 Active Model")
    st.sidebar.caption("Used in Tab 1 (Code Writer) and Tab 2 (Chatbot)")

    # Build flat list with group separators as display labels
    flat_keys = []
    flat_labels = []
    for group, keys in LIVE_MODEL_GROUPS.items():
        for k in keys:
            flat_keys.append(k)
            flat_labels.append(f"{LIVE_MODELS[k]['label']}")

    default_key = os.getenv("CODE_WRITER_MODEL", "claude-sonnet-4-6")
    default_idx = flat_keys.index(default_key) if default_key in flat_keys else 0

    selected_idx = st.sidebar.selectbox(
        "Choose model",
        options=range(len(flat_keys)),
        index=default_idx,
        format_func=lambda i: flat_labels[i],
        key="active_model_idx",
    )
    selected_key = flat_keys[selected_idx]
    info = LIVE_MODELS[selected_key]

    # Show key status
    ok, _ = key_status(selected_key)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Provider details**")
    st.sidebar.write(f"🏢 Provider: **{info['provider']}**")
    st.sidebar.write(f"🔑 Key needed: `{info['key_env']}`")

    if ok:
        st.sidebar.success(f"✅ {info['key_env']} is set")
    else:
        st.sidebar.error(f"❌ {info['key_env']} missing — set it in `.env`")

    feat_cols = st.sidebar.columns(2)
    feat_cols[0].write("🧠 Thinking" if info["supports_thinking"] else "💭 No thinking")
    feat_cols[1].write("🔧 Tools ✅" if info["supports_tools"] else "🔧 No tools")

    ctx = info["context_window"]
    st.sidebar.write(f"📐 Context: **{ctx:,}** tokens")

    return selected_key


# ── Streaming helpers — unified interface across providers ────────────────────

SYSTEM_PROMPT = (
    "You are an expert software engineer. Write clean, well-structured code. "
    "Use the available tools when helpful to gather context. Always explain your code."
)

# MCP tool schemas (same format works for both Anthropic and OpenAI-compat)
MOCK_MCP_TOOLS_ANTHROPIC = [
    {
        "name": "read_file",
        "description": "Read file contents from the project (MCP filesystem tool)",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "search_docs",
        "description": "Search documentation database for examples (MCP docs tool)",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "list_directory",
        "description": "List files in a directory (MCP filesystem tool)",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
]

# OpenAI function-calling format for the same tools
MOCK_MCP_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in MOCK_MCP_TOOLS_ANTHROPIC
]

MOCK_TOOL_RESPONSES = {
    "read_file":      "# Example file content\nclass MyClass:\n    pass\n",
    "search_docs":    "Found 3 relevant examples in documentation.",
    "list_directory": "['main.py', 'utils.py', 'tests/']",
}


def stream_anthropic(model_key: str, task: str):
    """Stream from Anthropic with adaptive thinking + MCP tools."""
    client = get_anthropic_client()
    info = LIVE_MODELS[model_key]
    messages = [{"role": "user", "content": task}]

    pre = client.messages.count_tokens(
        model=model_key,
        system=SYSTEM_PROMPT,
        tools=MOCK_MCP_TOOLS_ANTHROPIC if info["supports_tools"] else [],
        messages=messages,
    )
    yield {"type": "pre_count", "tokens": pre.input_tokens}

    kwargs = dict(
        model=model_key,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    if info["supports_tools"]:
        kwargs["tools"] = MOCK_MCP_TOOLS_ANTHROPIC
    if info["supports_thinking"]:
        kwargs["thinking"] = {"type": "adaptive"}

    thinking_tokens, tool_tokens = 0, 0
    thinking_text, tool_calls_made = "", []

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    yield {"type": "status", "text": "🧠 Reasoning…"}
                elif event.content_block.type == "tool_use":
                    yield {"type": "status", "text": f"🔧 MCP: {event.content_block.name}"}
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    thinking_text += event.delta.thinking
                elif event.delta.type == "text_delta":
                    yield {"type": "text_chunk", "text": event.delta.text}
        final = stream.get_final_message()

    for block in final.content:
        if block.type == "tool_use":
            tool_calls_made.append(block.name)
            tool_tokens += len(json.dumps(block.input).split()) * 2
            tool_tokens += len(MOCK_TOOL_RESPONSES.get(block.name, "").split()) * 2
        if block.type == "thinking":
            thinking_tokens += len(getattr(block, "thinking", "").split()) * 2

    in_tok, out_tok = final.usage.input_tokens, final.usage.output_tokens
    in_price = MODEL_PRICING.get(model_key, {}).get("input", 0)
    out_price = MODEL_PRICING.get(model_key, {}).get("output", 0)

    yield {
        "type": "done",
        "model_key": model_key,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "thinking_tokens": thinking_tokens,
        "tool_tokens": tool_tokens,
        "input_cost": token_cost(in_tok, in_price),
        "output_cost": token_cost(out_tok, out_price),
        "tool_cost": token_cost(tool_tokens, in_price),
        "total_cost": token_cost(in_tok, in_price) + token_cost(out_tok, out_price) + token_cost(tool_tokens, in_price),
        "tools_called": tool_calls_made,
    }


def stream_openai_compat(model_key: str, task: str):
    """Stream from OpenAI-compatible endpoint (Gemini, Groq/Mistral, Groq/Llama)."""
    client = get_openai_compat_client(model_key)
    info = LIVE_MODELS[model_key]
    api_model = info["api_model"]

    kwargs = dict(
        model=api_model,
        max_tokens=4096,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
    )
    if info["supports_tools"]:
        kwargs["tools"] = MOCK_MCP_TOOLS_OPENAI
        kwargs["tool_choice"] = "auto"

    full_text = ""
    in_tok, out_tok, tool_tokens = 0, 0, 0
    tool_calls_made = []

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            full_text += delta.content
            yield {"type": "text_chunk", "text": delta.content}
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    name = tc.function.name
                    if name not in tool_calls_made:
                        tool_calls_made.append(name)
                        yield {"type": "status", "text": f"🔧 Tool call: {name}"}
                        tool_tokens += len(MOCK_TOOL_RESPONSES.get(name, "").split()) * 2
        # capture usage from the last chunk if present
        if hasattr(chunk, "usage") and chunk.usage:
            in_tok = chunk.usage.prompt_tokens or 0
            out_tok = chunk.usage.completion_tokens or 0

    # Fallback token estimate if provider didn't return usage in stream
    if in_tok == 0:
        in_tok = len(task.split()) * 2
    if out_tok == 0:
        out_tok = len(full_text.split()) * 2

    in_price = MODEL_PRICING.get(model_key, {}).get("input", 0)
    out_price = MODEL_PRICING.get(model_key, {}).get("output", 0)

    yield {
        "type": "done",
        "model_key": model_key,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "thinking_tokens": 0,
        "tool_tokens": tool_tokens,
        "input_cost": token_cost(in_tok, in_price),
        "output_cost": token_cost(out_tok, out_price),
        "tool_cost": token_cost(tool_tokens, in_price),
        "total_cost": token_cost(in_tok, in_price) + token_cost(out_tok, out_price) + token_cost(tool_tokens, in_price),
        "tools_called": tool_calls_made,
    }


def run_code_writer(model_key: str, task: str):
    info = LIVE_MODELS[model_key]
    if info["sdk"] == "anthropic":
        yield from stream_anthropic(model_key, task)
    else:
        yield from stream_openai_compat(model_key, task)


def chat_stream_anthropic(model_key: str, messages: list) -> tuple[str, int, int]:
    client = get_anthropic_client()
    full = ""
    placeholder = st.empty()
    with client.messages.stream(model=model_key, max_tokens=2048, messages=messages) as stream:
        for text in stream.text_stream:
            full += text
            placeholder.markdown(full + "▌")
        final = stream.get_final_message()
    placeholder.markdown(full)
    return full, final.usage.input_tokens, final.usage.output_tokens


def chat_stream_openai_compat(model_key: str, messages: list) -> tuple[str, int, int]:
    client = get_openai_compat_client(model_key)
    info = LIVE_MODELS[model_key]
    api_model = info["api_model"]
    oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    full = ""
    in_tok, out_tok = 0, 0
    placeholder = st.empty()

    stream = client.chat.completions.create(model=api_model, max_tokens=2048, stream=True, messages=oai_messages)
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            full += delta.content
            placeholder.markdown(full + "▌")
        if hasattr(chunk, "usage") and chunk.usage:
            in_tok = chunk.usage.prompt_tokens or 0
            out_tok = chunk.usage.completion_tokens or 0

    placeholder.markdown(full)
    if in_tok == 0:
        in_tok = sum(len(m["content"].split()) * 2 for m in messages)
    if out_tok == 0:
        out_tok = len(full.split()) * 2
    return full, in_tok, out_tok


def count_tokens_anthropic(model_key: str, messages: list) -> int:
    client = get_anthropic_client()
    try:
        cr = client.messages.count_tokens(model=model_key, messages=messages)
        return cr.input_tokens
    except Exception:
        return sum(len(m["content"].split()) * 2 for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Code Writer
# ─────────────────────────────────────────────────────────────────────────────

def tab_code_writer(model_key: str):
    info = LIVE_MODELS[model_key]
    ok, _ = key_status(model_key)

    st.header("💻 Code Writer")

    # Model badge
    badge_col, _ = st.columns([2, 3])
    with badge_col:
        color = PROVIDER_COLORS.get(info["provider"], "#888")
        st.markdown(
            f'<span style="background:{color};color:white;padding:4px 10px;border-radius:12px;font-size:0.85em;font-weight:600">'
            f'  {info["label"]}  ·  {info["provider"]}</span>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Model ID: `{info.get('api_model', model_key)}` &nbsp;|&nbsp; "
        f"Context: **{info['context_window']:,}** tokens &nbsp;|&nbsp; "
        f"Thinking: {'✅' if info['supports_thinking'] else '❌'} &nbsp;|&nbsp; "
        f"MCP Tools: {'✅' if info['supports_tools'] else '❌'}"
    )

    if not ok:
        st.error(
            f"❌ `{info['key_env']}` is not set. Add it to your `.env` file to use this model.",
            icon="🔑",
        )
        return

    if "cw_cumulative_cost" not in st.session_state:
        st.session_state.cw_cumulative_cost = 0.0
        st.session_state.cw_iterations = 0
        st.session_state.cw_history = []

    task = st.text_area(
        "Describe your coding task",
        placeholder="e.g. Write a Python function that reads a CSV, groups by a column and plots a bar chart",
        height=120,
    )

    col_btn, col_reset = st.columns([1, 5])
    with col_btn:
        run = st.button("✨ Generate Code", type="primary", disabled=not task.strip())
    with col_reset:
        if st.button("🔄 Reset History"):
            st.session_state.cw_cumulative_cost = 0.0
            st.session_state.cw_iterations = 0
            st.session_state.cw_history = []
            st.rerun()

    if run and task.strip():
        st.session_state.cw_iterations += 1
        status_ph = st.empty()
        code_ph = st.empty()
        final_data = None
        collected = ""

        for chunk in run_code_writer(model_key, task):
            if chunk["type"] == "status":
                status_ph.info(chunk["text"])
            elif chunk["type"] == "text_chunk":
                collected += chunk["text"]
                code_ph.code(collected, language="python")
            elif chunk["type"] == "done":
                final_data = chunk

        status_ph.empty()

        if final_data:
            st.session_state.cw_cumulative_cost += final_data["total_cost"]
            st.session_state.cw_history.append(final_data)

            st.markdown("### 📊 Token Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🧠 Reasoning", f"{final_data['thinking_tokens']:,}",
                      help="Only populated for Claude with adaptive thinking")
            c2.metric("📥 Input",   f"{final_data['input_tokens']:,}")
            c3.metric("📤 Output",  f"{final_data['output_tokens']:,}")
            c4.metric("🔧 Tool Use",f"{final_data['tool_tokens']:,}")

            st.markdown("### 💰 Cost Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Input Cost",    fmt_cost(final_data["input_cost"]))
            c2.metric("Output Cost",   fmt_cost(final_data["output_cost"]))
            c3.metric("Tool Cost",     fmt_cost(final_data["tool_cost"]))
            c4.metric("This Request",  fmt_cost(final_data["total_cost"]))

            st.metric(
                f"🏦 Cumulative Cost ({st.session_state.cw_iterations} iterations)",
                fmt_cost(st.session_state.cw_cumulative_cost),
                delta=fmt_cost(final_data["total_cost"]),
            )

            if final_data["tools_called"]:
                st.info(f"🔧 Tools called: {', '.join(final_data['tools_called'])}")

            labels = ["Input", "Output", "Tool Use"]
            values = [final_data["input_cost"], final_data["output_cost"], max(final_data["tool_cost"], 1e-9)]
            fig = px.pie(names=labels, values=values, title="Cost Distribution")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

            if len(st.session_state.cw_history) > 1:
                st.markdown("### 📈 Cumulative Cost Over Iterations")
                cum, running = [], 0.0
                for i, h in enumerate(st.session_state.cw_history, 1):
                    running += h["total_cost"]
                    cum.append({"Iteration": i, "Cumulative Cost ($)": running})
                fig2 = px.line(pd.DataFrame(cum), x="Iteration", y="Cumulative Cost ($)", markers=True)
                st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Chatbot + Context Window Tracker
# ─────────────────────────────────────────────────────────────────────────────

def tab_chatbot(model_key: str):
    info = LIVE_MODELS[model_key]
    ok, _ = key_status(model_key)
    ctx_window = info["context_window"]

    st.header("💬 Chatbot + Context Window Tracker")

    badge_col, _ = st.columns([2, 3])
    with badge_col:
        color = PROVIDER_COLORS.get(info["provider"], "#888")
        st.markdown(
            f'<span style="background:{color};color:white;padding:4px 10px;border-radius:12px;font-size:0.85em;font-weight:600">'
            f'  {info["label"]}  ·  {info["provider"]}</span>',
            unsafe_allow_html=True,
        )
    st.caption(f"Context window: **{ctx_window:,}** tokens &nbsp;|&nbsp; Key: `{info['key_env']}`")

    if not ok:
        st.error(f"❌ `{info['key_env']}` is not set. Add it to your `.env` file.", icon="🔑")
        return

    if "chat_messages" not in st.session_state or st.session_state.get("chat_model_key") != model_key:
        st.session_state.chat_messages = []
        st.session_state.chat_total_input_tokens = 0
        st.session_state.chat_total_output_tokens = 0
        st.session_state.chat_total_cost = 0.0
        st.session_state.chat_per_turn = []
        st.session_state.chat_model_key = model_key

    # Token count for context bar — exact for Anthropic, estimated for others
    total_conv_tokens = 0
    if st.session_state.chat_messages:
        if info["sdk"] == "anthropic":
            total_conv_tokens = count_tokens_anthropic(model_key, st.session_state.chat_messages)
        else:
            total_conv_tokens = sum(
                len(m["content"].split()) * 2 for m in st.session_state.chat_messages
            )

    pct = min(total_conv_tokens / ctx_window, 1.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Context Window")
    count_label = "exact (Anthropic API)" if info["sdk"] == "anthropic" else "estimated"
    st.sidebar.caption(f"Token count: {count_label}")
    st.sidebar.progress(pct)
    st.sidebar.write(f"**{total_conv_tokens:,}** / **{ctx_window:,}** ({pct*100:.1f}%)")
    if pct >= 0.9:
        st.sidebar.error("🚨 Context window nearly full!")
    elif pct >= 0.7:
        st.sidebar.warning("⚠️ Context 70%+ full — costs rising.")

    st.sidebar.markdown("### 💰 Conversation Cost")
    st.sidebar.metric("Total Cost",          fmt_cost(st.session_state.chat_total_cost))
    st.sidebar.metric("Total Input Tokens",  f"{st.session_state.chat_total_input_tokens:,}")
    st.sidebar.metric("Total Output Tokens", f"{st.session_state.chat_total_output_tokens:,}")

    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state.chat_messages = []
        st.session_state.chat_total_input_tokens = 0
        st.session_state.chat_total_output_tokens = 0
        st.session_state.chat_total_cost = 0.0
        st.session_state.chat_per_turn = []
        st.rerun()

    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                turn_idx = i // 2
                if turn_idx < len(st.session_state.chat_per_turn):
                    t = st.session_state.chat_per_turn[turn_idx]
                    with st.expander("Token details", expanded=False):
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("Input",  f"{t['input']:,}")
                        cc2.metric("Output", f"{t['output']:,}")
                        cc3.metric("Cost",   fmt_cost(t["cost"]))

    if prompt := st.chat_input("Type a message…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if info["sdk"] == "anthropic":
                full, in_tok, out_tok = chat_stream_anthropic(model_key, st.session_state.chat_messages)
            else:
                full, in_tok, out_tok = chat_stream_openai_compat(model_key, st.session_state.chat_messages)

            _, _, cost = calc_cost(in_tok, out_tok, model_key)
            st.session_state.chat_messages.append({"role": "assistant", "content": full})
            st.session_state.chat_total_input_tokens  += in_tok
            st.session_state.chat_total_output_tokens += out_tok
            st.session_state.chat_total_cost          += cost
            st.session_state.chat_per_turn.append({"input": in_tok, "output": out_tok, "cost": cost})

            with st.expander("Token details", expanded=True):
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("This input",  f"{in_tok:,}")
                cc2.metric("This output", f"{out_tok:,}")
                cc3.metric("This cost",   fmt_cost(cost))

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Multi-Provider Enterprise Cost Calculator
# ─────────────────────────────────────────────────────────────────────────────

def tab_enterprise():
    st.header("🏢 Multi-Provider Cost Calculator")
    st.success("✅ **No API keys required** — pure math on published pricing.", icon="✅")

    col1, col2, col3 = st.columns(3)
    with col1:
        qps = st.number_input("Queries per second (QPS)", 0.01, 10000.0, 1.0, 0.5)
        avg_input = st.number_input("Avg input tokens / query", 100, 200_000, 500, 100)
    with col2:
        avg_output = st.number_input("Avg output tokens / query", 50, 8000, 300, 50)
        peak_mult = st.slider("Peak traffic multiplier", 1.0, 20.0, 2.0, 0.5)
    with col3:
        selected_providers = st.multiselect("Filter providers", options=ALL_PROVIDERS, default=ALL_PROVIDERS)

    filtered = {k: v for k, v in MODEL_PRICING.items() if v["provider"] in selected_providers}
    queries_per_month = qps * 86400 * 30

    rows = []
    for key, info in filtered.items():
        q = token_cost(avg_input, info["input"]) + token_cost(avg_output, info["output"])
        rows.append({
            "Model": info["name"], "Provider": info["provider"],
            "Input $/1M": f"${info['input']:.3f}", "Output $/1M": f"${info['output']:.3f}",
            "$ / query": fmt_cost(q),
            "$ / hour":  f"${q * qps * 3600:,.4f}",
            "$ / day":   f"${q * qps * 86400:,.2f}",
            "$ / month": f"${q * queries_per_month:,.2f}",
            "Context":   f"{info['context_window']:,}",
            "Notes":     info["notes"],
            "_q": q,
        })

    df = pd.DataFrame(rows).sort_values("_q")
    st.subheader(f"📊 Cost Comparison — {avg_input} in / {avg_output} out @ {qps} QPS")
    st.dataframe(df.drop(columns=["_q"]).set_index("Model"), use_container_width=True)

    df["Monthly Cost ($)"] = df["_q"] * queries_per_month
    fig = px.bar(
        df.sort_values("Monthly Cost ($)"), x="Model", y="Monthly Cost ($)",
        color="Provider", color_discrete_map=PROVIDER_COLORS,
        title=f"Monthly Cost at {qps} QPS ({avg_input} in / {avg_output} out tokens)", text_auto=".2f",
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        df, x="Context", y="Monthly Cost ($)", color="Provider", text="Model",
        color_discrete_map=PROVIDER_COLORS, title="Context Window vs Monthly Cost", log_x=True,
    )
    fig2.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Optimization Suggestions")
    tips = [
        {"Tip": "Anthropic Batch API (async workloads)",      "Savings": "50% off all Anthropic models",        "Risk": "Latency only"},
        {"Tip": "Prompt caching (repeated system prompts)",   "Savings": "Up to 90% on cached input tokens",    "Risk": "None"},
        {"Tip": "Route simple queries to cheapest model",     "Savings": f"Up to {(1 - df['_q'].min() / df['_q'].max()) * 100:.0f}% vs most expensive", "Risk": "Quality — test first"},
        {"Tip": "Trim input tokens 20% via prompt engineering","Savings": "~20% input cost",                    "Risk": "Low"},
    ]
    st.dataframe(pd.DataFrame(tips), use_container_width=True)

    if len(df) >= 3:
        st.markdown("---")
        st.subheader("🏆 Quick Picks")
        s = df.sort_values("_q")
        c1, c2, c3 = st.columns(3)
        c1.success(f"**Cheapest**\n\n{s.iloc[0]['Model']}\n\n${s.iloc[0]['_q'] * queries_per_month:,.2f}/mo")
        mid = len(s) // 2
        c2.info(f"**Mid-range**\n\n{s.iloc[mid]['Model']}\n\n${s.iloc[mid]['_q'] * queries_per_month:,.2f}/mo")
        c3.warning(f"**Most Expensive**\n\n{s.iloc[-1]['Model']}\n\n${s.iloc[-1]['_q'] * queries_per_month:,.2f}/mo")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RAG vs Plain LLM
# ─────────────────────────────────────────────────────────────────────────────

def tab_rag():
    st.header("🔍 RAG vs Plain LLM Cost")
    st.success("✅ **No API keys required** — pure cost math across all providers.", icon="✅")

    col1, col2 = st.columns(2)
    with col1:
        qps = st.number_input("QPS", 0.01, 10000.0, 1.0, 0.1, key="rag_qps")
        avg_context_kb = st.slider("Full context size (KB)", 1, 500, 20)
        model_key = st.selectbox(
            "LLM Model", options=list(MODEL_PRICING.keys()),
            format_func=lambda k: f"{MODEL_PRICING[k]['name']} ({MODEL_PRICING[k]['provider']})",
            key="rag_model", index=list(MODEL_PRICING.keys()).index("claude-sonnet-4-6"),
        )
    with col2:
        avg_q_tokens = st.number_input("Avg question tokens", 50, 500, 100, key="rag_q")
        avg_a_tokens = st.number_input("Avg answer tokens", 100, 4000, 400, key="rag_a")
        rag_infra    = st.number_input("RAG infra cost ($/mo)", 0, 5000, 70, key="rag_infra")

    avg_context_tokens = (avg_context_kb * 1024) // 4
    retrieved_tokens   = RAG_INFRA["avg_chunks_retrieved"] * RAG_INFRA["avg_chunk_tokens"]
    queries_per_month  = qps * 86400 * 30
    minfo = MODEL_PRICING[model_key]

    plain_in  = avg_context_tokens + avg_q_tokens
    plain_q   = token_cost(plain_in, minfo["input"]) + token_cost(avg_a_tokens, minfo["output"])
    plain_monthly = plain_q * queries_per_month

    rag_embed_q   = token_cost(avg_q_tokens, RAG_INFRA["embedding_per_1m_tokens"])
    rag_llm_in    = retrieved_tokens + avg_q_tokens
    rag_llm_q     = token_cost(rag_llm_in, minfo["input"]) + token_cost(avg_a_tokens, minfo["output"])
    rag_q         = rag_embed_q + rag_llm_q
    rag_embed_mo  = rag_embed_q * queries_per_month
    rag_llm_mo    = rag_llm_q  * queries_per_month
    rag_total_mo  = rag_llm_mo + rag_embed_mo + rag_infra

    saved = plain_monthly - rag_total_mo
    denom = plain_monthly - rag_llm_mo - rag_embed_mo
    breakeven = rag_infra / denom if denom > 0 else float("inf")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Plain LLM / month", f"${plain_monthly:,.2f}")
    c2.metric("RAG / month",       f"${rag_total_mo:,.2f}")
    c3.metric("Savings with RAG",  f"${abs(saved):,.2f}/mo",
              delta="cheaper ✅" if saved > 0 else "more expensive ❌")

    if saved > 0 and breakeven != float("inf"):
        st.info(f"📅 Break-even: **{breakeven:.1f} months** at this traffic level.")
    elif saved <= 0:
        st.warning("⚠️ RAG is not cheaper here — context is small enough that sending it each time beats RAG infra cost.")

    bar_data = pd.DataFrame({
        "Approach":  ["Plain LLM", "RAG"],
        "LLM Input": [token_cost(plain_in,  minfo["input"]) * queries_per_month,
                      token_cost(rag_llm_in, minfo["input"]) * queries_per_month],
        "LLM Output":[token_cost(avg_a_tokens, minfo["output"]) * queries_per_month,
                      token_cost(avg_a_tokens, minfo["output"]) * queries_per_month],
        "Embedding": [0, rag_embed_mo],
        "Infra":     [0, float(rag_infra)],
    })
    fig = go.Figure()
    for col, color in zip(["LLM Input","LLM Output","Embedding","Infra"],
                          ["#4C78A8","#72B7B2","#F58518","#E45756"]):
        fig.add_trace(go.Bar(name=col, x=bar_data["Approach"], y=bar_data[col], marker_color=color))
    fig.update_layout(barmode="stack", title="Monthly Cost Breakdown", yaxis_title="Cost ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔄 RAG Savings Across All Models")
    cross = []
    for k, info in MODEL_PRICING.items():
        pm  = (token_cost(plain_in,    info["input"]) + token_cost(avg_a_tokens, info["output"])) * queries_per_month
        rm  = (token_cost(rag_llm_in,  info["input"]) + token_cost(avg_a_tokens, info["output"])) * queries_per_month + rag_embed_mo + rag_infra
        s   = pm - rm
        cross.append({"Model": info["name"], "Provider": info["provider"],
                      "Plain LLM/mo": f"${pm:,.2f}", "RAG/mo": f"${rm:,.2f}",
                      "Savings": f"${abs(s):,.2f}" if s > 0 else f"-${abs(s):,.2f}",
                      "Worth it?": "✅ Yes" if s > 0 else "❌ No", "_s": s})
    df_c = pd.DataFrame(cross).sort_values("_s", ascending=False)
    st.dataframe(df_c.drop(columns=["_s"]).set_index("Model"), use_container_width=True)

    fig3 = px.bar(df_c, x="Model", y="_s", color="Provider", color_discrete_map=PROVIDER_COLORS,
                  title="Monthly Savings from RAG by Model", labels={"_s": "Monthly Savings ($)"})
    fig3.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig3.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Pricing Reference
# ─────────────────────────────────────────────────────────────────────────────

def tab_reference():
    st.header("📖 Model Pricing Reference")
    st.caption("20 models across 7 providers — published pricing as of June 2025. No API keys needed.")

    provider_filter = st.multiselect("Filter by provider", options=ALL_PROVIDERS, default=ALL_PROVIDERS)
    rows = []
    for k, v in MODEL_PRICING.items():
        if v["provider"] not in provider_filter:
            continue
        rows.append({
            "Model": v["name"], "Provider": v["provider"],
            "Input $/1M": v["input"], "Output $/1M": v["output"],
            "Context (tokens)": v["context_window"], "Notes": v["notes"],
        })
    df = pd.DataFrame(rows).sort_values(["Provider", "Input $/1M"])
    st.dataframe(df.set_index("Model"), use_container_width=True)

    fig = px.scatter(df, x="Input $/1M", y="Output $/1M", color="Provider", text="Model",
                     color_discrete_map=PROVIDER_COLORS, title="Input vs Output Pricing (per 1M tokens)",
                     size=[1]*len(df))
    fig.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(df.sort_values("Input $/1M"), x="Model", y="Input $/1M", color="Provider",
                  color_discrete_map=PROVIDER_COLORS,
                  title="Input Price per 1M Tokens (cheapest → most expensive)")
    fig2.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Token Cost Explorer",
        page_icon="🪙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🪙 AI Token Cost Explorer")
    st.markdown(
        "Live API calls across **Anthropic · Google · Groq** — "
        "cost comparison across **20 models** from 7 providers. "
        "Tabs 3–5 need **no API keys**."
    )

    # Sidebar model selector — drives Tabs 1 & 2
    active_model = render_model_selector()

    # Dashboard info panel
    with st.expander("ℹ️ How each tab works — models & token counting", expanded=False):
        info = LIVE_MODELS[active_model]
        dash_data = [
            {"Tab": "💻 Tab 1 — Code Writer",       "Active Model": info["label"],
             "Provider": info["provider"],            "API Key": info["key_env"],
             "Token Counting": "Anthropic count_tokens() API (exact)" if info["sdk"] == "anthropic" else "Usage from stream response",
             "Thinking": "✅ Adaptive" if info["supports_thinking"] else "❌ N/A",
             "MCP Tools": "✅ Simulated" if info["supports_tools"] else "❌ N/A"},
            {"Tab": "💬 Tab 2 — Chatbot",            "Active Model": info["label"],
             "Provider": info["provider"],            "API Key": info["key_env"],
             "Token Counting": "Anthropic count_tokens() API (exact)" if info["sdk"] == "anthropic" else "Word-count estimate",
             "Thinking": "N/A",                      "MCP Tools": "N/A"},
            {"Tab": "🏢 Tab 3 — Enterprise Calc",    "Active Model": "—  (no live calls)",
             "Provider": "All 7 providers",           "API Key": "None required",
             "Token Counting": "Pure math: tokens × price / 1M",
             "Thinking": "N/A",                      "MCP Tools": "N/A"},
            {"Tab": "🔍 Tab 4 — RAG vs LLM",         "Active Model": "—  (no live calls)",
             "Provider": "All 7 providers",           "API Key": "None required",
             "Token Counting": "Pure math: tokens × price / 1M",
             "Thinking": "N/A",                      "MCP Tools": "N/A"},
            {"Tab": "📖 Tab 5 — Pricing Ref",        "Active Model": "—  (no live calls)",
             "Provider": "All 7 providers",           "API Key": "None required",
             "Token Counting": "Static config dict",
             "Thinking": "N/A",                      "MCP Tools": "N/A"},
        ]
        st.dataframe(pd.DataFrame(dash_data).set_index("Tab"), use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💻 Code Writer (MCP)",
        "💬 Chatbot + Context",
        "🏢 Enterprise Calculator",
        "🔍 RAG vs Plain LLM",
        "📖 Pricing Reference",
    ])

    with tab1:
        tab_code_writer(active_model)
    with tab2:
        tab_chatbot(active_model)
    with tab3:
        tab_enterprise()
    with tab4:
        tab_rag()
    with tab5:
        tab_reference()


if __name__ == "__main__":
    main()
