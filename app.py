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

    # ── Live session stats for both tabs ─────────────────────────────────────
    st.sidebar.markdown("---")

    def _ctx_bar(label: str, used: int, window: int, cost: float, turns: int, sdk: str):
        pct = min(used / window, 1.0) if window else 0
        st.sidebar.markdown(f"### {label}")
        st.sidebar.caption(f"Token count: {'exact' if sdk == 'anthropic' else 'estimated'}")
        st.sidebar.progress(pct)
        st.sidebar.write(f"**{used:,}** / **{window:,}** ({pct*100:.1f}%)")
        if pct >= 0.9:
            st.sidebar.error("🚨 Nearly full!")
        elif pct >= 0.7:
            st.sidebar.warning("⚠️ 70%+ full")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Cost", fmt_cost(cost))
        c2.metric("Turns", turns)

    # Tab 1 — Code Writer stats
    cw_msgs   = st.session_state.get("cw_messages", [])
    cw_model  = st.session_state.get("cw_model_key", selected_key)
    cw_sdk    = LIVE_MODELS.get(cw_model, {}).get("sdk", "anthropic")
    cw_window = LIVE_MODELS.get(cw_model, {}).get("context_window", 1)
    if cw_msgs:
        cw_used = (count_tokens_anthropic(cw_model, cw_msgs)
                   if cw_sdk == "anthropic"
                   else sum(len(m["content"].split()) * 2 for m in cw_msgs))
    else:
        cw_used = 0
    _ctx_bar("💻 Code Writer Context", cw_used, cw_window,
             st.session_state.get("cw_cumulative_cost", 0.0),
             len(st.session_state.get("cw_turn_data", [])),
             cw_sdk)

    if st.sidebar.button("🗑️ Clear Code Writer", key="sb_clear_cw"):
        st.session_state.cw_messages = []
        st.session_state.cw_turn_data = []
        st.session_state.cw_cumulative_cost = 0.0
        st.rerun()

    st.sidebar.markdown("---")

    # Tab 2 — Chatbot stats
    chat_msgs   = st.session_state.get("chat_messages", [])
    chat_model  = st.session_state.get("chat_model_key", selected_key)
    chat_sdk    = LIVE_MODELS.get(chat_model, {}).get("sdk", "anthropic")
    chat_window = LIVE_MODELS.get(chat_model, {}).get("context_window", 1)
    if chat_msgs:
        chat_used = (count_tokens_anthropic(chat_model, chat_msgs)
                     if chat_sdk == "anthropic"
                     else sum(len(m["content"].split()) * 2 for m in chat_msgs))
    else:
        chat_used = 0
    _ctx_bar("💬 Chatbot Context", chat_used, chat_window,
             st.session_state.get("chat_total_cost", 0.0),
             len(st.session_state.get("chat_per_turn", [])),
             chat_sdk)

    if st.sidebar.button("🗑️ Clear Chatbot", key="sb_clear_chat"):
        st.session_state.chat_messages = []
        st.session_state.chat_total_input_tokens = 0
        st.session_state.chat_total_output_tokens = 0
        st.session_state.chat_total_cost = 0.0
        st.session_state.chat_per_turn = []
        st.rerun()

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


def stream_anthropic(model_key: str, messages: list):
    """Stream from Anthropic with adaptive thinking + MCP tools."""
    client = get_anthropic_client()
    info = LIVE_MODELS[model_key]

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
        messages=messages,   # full conversation history
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


def stream_openai_compat(model_key: str, messages: list):
    """Stream from OpenAI-compatible endpoint (Gemini, Groq/Mistral, Groq/Llama)."""
    client = get_openai_compat_client(model_key)
    info = LIVE_MODELS[model_key]
    api_model = info["api_model"]

    kwargs = dict(
        model=api_model,
        max_tokens=4096,
        stream=True,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
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


def run_code_writer(model_key: str, messages: list):
    info = LIVE_MODELS[model_key]
    if info["sdk"] == "anthropic":
        yield from stream_anthropic(model_key, messages)
    else:
        yield from stream_openai_compat(model_key, messages)


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
# Receipt renderer
# ─────────────────────────────────────────────────────────────────────────────

TOOL_EXPLANATIONS = {
    "read_file":      ("📄 Read File",      "Claude wanted to read an existing source file for context before writing code."),
    "search_docs":    ("🔍 Search Docs",    "Claude searched for documentation examples relevant to your task."),
    "list_directory": ("📁 List Directory", "Claude listed project files to understand the codebase structure."),
}

def render_receipt(d: dict, iteration: int, cumulative: float, model_info: dict):
    """Render a billing receipt for a single code writer call."""
    import datetime
    model_key = d["model_key"]
    pricing = MODEL_PRICING.get(model_key, {"input": 0, "output": 0})
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown("---")
    st.markdown("### 🧾 Billing Receipt")

    # ── Receipt card ──────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:20px;font-family:monospace;background:#fafafa">
<div style="display:flex;justify-content:space-between;border-bottom:1px dashed #ccc;padding-bottom:8px;margin-bottom:12px">
  <span style="font-size:1.1em;font-weight:700">AI Token Cost Explorer</span>
  <span style="color:#888;font-size:0.85em">{timestamp}</span>
</div>
<div style="color:#555;margin-bottom:12px;font-size:0.85em">
  Model: <b>{model_info['label']}</b> &nbsp;·&nbsp; Provider: <b>{model_info['provider']}</b><br>
  Rate: ${pricing['input']:.2f} / 1M input &nbsp;·&nbsp; ${pricing['output']:.2f} / 1M output
</div>

<table style="width:100%;border-collapse:collapse;font-size:0.9em">
  <tr style="border-bottom:1px solid #eee">
    <th style="text-align:left;padding:6px 4px;color:#333">Line Item</th>
    <th style="text-align:right;padding:6px 4px;color:#333">Tokens</th>
    <th style="text-align:right;padding:6px 4px;color:#333">Rate $/1M</th>
    <th style="text-align:right;padding:6px 4px;color:#333">Cost</th>
  </tr>
  <tr>
    <td style="padding:6px 4px">📥 Input tokens<br><span style="color:#888;font-size:0.8em">Your prompt + system + tool schemas</span></td>
    <td style="text-align:right;padding:6px 4px">{d['input_tokens']:,}</td>
    <td style="text-align:right;padding:6px 4px">${pricing['input']:.2f}</td>
    <td style="text-align:right;padding:6px 4px">{fmt_cost(d['input_cost'])}</td>
  </tr>
  <tr style="background:#f5f5f5">
    <td style="padding:6px 4px">📤 Output tokens<br><span style="color:#888;font-size:0.8em">Generated code + explanation</span></td>
    <td style="text-align:right;padding:6px 4px">{d['output_tokens']:,}</td>
    <td style="text-align:right;padding:6px 4px">${pricing['output']:.2f}</td>
    <td style="text-align:right;padding:6px 4px">{fmt_cost(d['output_cost'])}</td>
  </tr>
  <tr>
    <td style="padding:6px 4px">🧠 Reasoning tokens<br><span style="color:#888;font-size:0.8em">Claude's internal thinking (Claude only)</span></td>
    <td style="text-align:right;padding:6px 4px">{d['thinking_tokens']:,}</td>
    <td style="text-align:right;padding:6px 4px">${pricing['input']:.2f}</td>
    <td style="text-align:right;padding:6px 4px">{fmt_cost(token_cost(d['thinking_tokens'], pricing['input']))}</td>
  </tr>
  <tr style="background:#f5f5f5">
    <td style="padding:6px 4px">🔧 Tool use tokens<br><span style="color:#888;font-size:0.8em">MCP call JSON + mock responses</span></td>
    <td style="text-align:right;padding:6px 4px">{d['tool_tokens']:,}</td>
    <td style="text-align:right;padding:6px 4px">${pricing['input']:.2f}</td>
    <td style="text-align:right;padding:6px 4px">{fmt_cost(d['tool_cost'])}</td>
  </tr>
  <tr style="border-top:2px solid #333;font-weight:700">
    <td style="padding:8px 4px">TOTAL — This Request</td>
    <td style="text-align:right;padding:8px 4px">{d['input_tokens'] + d['output_tokens'] + d['thinking_tokens'] + d['tool_tokens']:,}</td>
    <td></td>
    <td style="text-align:right;padding:8px 4px">{fmt_cost(d['total_cost'])}</td>
  </tr>
  <tr style="color:#e07b39;font-weight:700">
    <td style="padding:4px 4px">🏦 Cumulative ({iteration} iteration{"s" if iteration > 1 else ""})</td>
    <td></td>
    <td></td>
    <td style="text-align:right;padding:4px 4px">{fmt_cost(cumulative)}</td>
  </tr>
</table>
</div>
""", unsafe_allow_html=True)

    # ── Tool call explanation ─────────────────────────────────────────────────
    if d["tools_called"]:
        st.markdown("#### 🔧 Why Claude Used Tools")
        st.caption(
            "Tools let Claude gather context before writing — like a developer "
            "searching docs or reading files first. Each call adds token overhead."
        )
        unique_tools = list(dict.fromkeys(d["tools_called"]))  # preserve order, dedupe
        for tool_name in unique_tools:
            count = d["tools_called"].count(tool_name)
            emoji_label, reason = TOOL_EXPLANATIONS.get(
                tool_name, ("🔧 " + tool_name, "Claude decided this tool was useful.")
            )
            times = f" (called {count}×)" if count > 1 else ""
            with st.expander(f"{emoji_label}{times}", expanded=True):
                st.write(f"**Why:** {reason}")
                st.write(f"**Token cost:** ~{50 * count} tokens "
                         f"({fmt_cost(token_cost(50 * count, pricing['input']))})"
                         f" — the call JSON + mock response payload")
                st.caption("In a real MCP setup this would call an actual filesystem or database.")

    # ── Charts ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        labels = ["📥 Input", "📤 Output", "🧠 Reasoning", "🔧 Tools"]
        values = [d["input_cost"], d["output_cost"],
                  token_cost(d["thinking_tokens"], pricing["input"]),
                  max(d["tool_cost"], 1e-9)]
        fig = px.pie(names=labels, values=values, title="Cost Distribution")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        tok_labels = ["📥 Input", "📤 Output", "🧠 Reasoning", "🔧 Tools"]
        tok_values = [d["input_tokens"], d["output_tokens"],
                      d["thinking_tokens"], d["tool_tokens"]]
        fig2 = px.bar(x=tok_labels, y=tok_values, title="Token Count by Category",
                      labels={"x": "", "y": "Tokens"},
                      color=tok_labels,
                      color_discrete_sequence=["#4C78A8","#72B7B2","#F58518","#E45756"])
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


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
        st.error(f"❌ `{info['key_env']}` is not set. Add it to your `.env` file.", icon="🔑")
        return

    # Session state init
    if "cw_messages" not in st.session_state:
        st.session_state.cw_messages = []
        st.session_state.cw_turn_data = []
        st.session_state.cw_cumulative_cost = 0.0
        st.session_state.cw_model_key = model_key

    # Model switched — ask what to do instead of silently resetting
    prev_model = st.session_state.get("cw_model_key", model_key)
    if prev_model != model_key and st.session_state.cw_messages:
        prev_label = LIVE_MODELS.get(prev_model, {}).get("label", prev_model)
        st.warning(
            f"⚠️ You switched from **{prev_label}** to **{info['label']}**. "
            f"What would you like to do with the existing conversation?",
            icon="🔄",
        )
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Clear & start fresh", type="primary", use_container_width=True):
            st.session_state.cw_messages = []
            st.session_state.cw_turn_data = []
            st.session_state.cw_cumulative_cost = 0.0
            st.session_state.cw_model_key = model_key
            st.rerun()
        if c2.button("📋 Keep history, switch model", use_container_width=True):
            # Keep messages, just update the model key and cost tracking
            st.session_state.cw_model_key = model_key
            st.rerun()
        return   # don't render the chat until user decides

    st.session_state.cw_model_key = model_key

    # ── Render conversation history ───────────────────────────────────────────
    for i, msg in enumerate(st.session_state.cw_messages):
        role_label = "🧑 You" if msg["role"] == "user" else f"🤖 {info['label']}"
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                turn_idx = i // 2
                if turn_idx < len(st.session_state.cw_turn_data):
                    t = st.session_state.cw_turn_data[turn_idx]
                    with st.expander("🧾 Receipt for this turn", expanded=False):
                        render_receipt(t, turn_idx + 1,
                                       sum(x["total_cost"] for x in st.session_state.cw_turn_data[:turn_idx+1]),
                                       info)

    # ── Input ─────────────────────────────────────────────────────────────────
    placeholder_text = (
        "Describe your coding task…" if not st.session_state.cw_messages
        else "Follow up — e.g. 'make it async', 'add error handling', 'write tests'…"
    )

    if prompt := st.chat_input(placeholder_text):
        st.session_state.cw_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_ph = st.empty()
            code_ph = st.empty()
            collected = ""
            final_data = None

            for chunk in run_code_writer(model_key, st.session_state.cw_messages):
                if chunk["type"] == "status":
                    status_ph.info(chunk["text"])
                elif chunk["type"] == "text_chunk":
                    collected += chunk["text"]
                    code_ph.markdown(collected + "▌")
                elif chunk["type"] == "done":
                    final_data = chunk

            status_ph.empty()
            code_ph.markdown(collected)

            if final_data:
                st.session_state.cw_messages.append({"role": "assistant", "content": collected})
                st.session_state.cw_cumulative_cost += final_data["total_cost"]
                st.session_state.cw_turn_data.append(final_data)

                with st.expander("🧾 Receipt for this turn", expanded=True):
                    render_receipt(final_data, len(st.session_state.cw_turn_data),
                                   st.session_state.cw_cumulative_cost, info)

        st.rerun()

    # ── Cumulative cost chart ─────────────────────────────────────────────────
    if len(st.session_state.cw_turn_data) > 1:
        st.markdown("### 📈 Cumulative Cost Over Turns")
        cum, running = [], 0.0
        for i, h in enumerate(st.session_state.cw_turn_data, 1):
            running += h["total_cost"]
            cum.append({"Turn": i, "Cumulative Cost ($)": running})
        fig3 = px.line(pd.DataFrame(cum), x="Turn", y="Cumulative Cost ($)", markers=True)
        st.plotly_chart(fig3, use_container_width=True)


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

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        st.session_state.chat_total_input_tokens = 0
        st.session_state.chat_total_output_tokens = 0
        st.session_state.chat_total_cost = 0.0
        st.session_state.chat_per_turn = []
        st.session_state.chat_model_key = model_key

    prev_chat_model = st.session_state.get("chat_model_key", model_key)
    if prev_chat_model != model_key and st.session_state.chat_messages:
        prev_label = LIVE_MODELS.get(prev_chat_model, {}).get("label", prev_chat_model)
        st.warning(
            f"⚠️ You switched from **{prev_label}** to **{info['label']}**. "
            f"What would you like to do with the existing conversation?",
            icon="🔄",
        )
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Clear & start fresh", type="primary",
                     use_container_width=True, key="chat_clear_switch"):
            st.session_state.chat_messages = []
            st.session_state.chat_total_input_tokens = 0
            st.session_state.chat_total_output_tokens = 0
            st.session_state.chat_total_cost = 0.0
            st.session_state.chat_per_turn = []
            st.session_state.chat_model_key = model_key
            st.rerun()
        if c2.button("📋 Keep history, switch model",
                     use_container_width=True, key="chat_keep_switch"):
            st.session_state.chat_model_key = model_key
            st.rerun()
        return

    st.session_state.chat_model_key = model_key

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
# TAB 6 — Session Cost Receipt
# ─────────────────────────────────────────────────────────────────────────────

def tab_session_cost():
    import datetime
    st.header("🧾 Session Cost Receipt")
    st.caption("Everything spent in this browser session — across all models, both tabs.")

    # ── Gather data from session state ────────────────────────────────────────
    cw_turns    = st.session_state.get("cw_turn_data", [])
    cw_model    = st.session_state.get("cw_model_key", "—")
    cw_cost     = st.session_state.get("cw_cumulative_cost", 0.0)

    chat_turns  = st.session_state.get("chat_per_turn", [])
    chat_model  = st.session_state.get("chat_model_key", "—")
    chat_cost   = st.session_state.get("chat_total_cost", 0.0)
    chat_in     = st.session_state.get("chat_total_input_tokens", 0)
    chat_out    = st.session_state.get("chat_total_output_tokens", 0)

    total_cost  = cw_cost + chat_cost

    cw_in    = sum(t.get("input_tokens", 0)    for t in cw_turns)
    cw_out   = sum(t.get("output_tokens", 0)   for t in cw_turns)
    cw_think = sum(t.get("thinking_tokens", 0) for t in cw_turns)
    cw_tool  = sum(t.get("tool_tokens", 0)     for t in cw_turns)

    # ── Top summary metrics ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Session Cost", fmt_cost(total_cost))
    c2.metric("💻 Code Writer Cost",   fmt_cost(cw_cost),   help=f"{len(cw_turns)} turns")
    c3.metric("💬 Chatbot Cost",       fmt_cost(chat_cost), help=f"{len(chat_turns)} turns")
    c4.metric("🔁 Total Turns",        len(cw_turns) + len(chat_turns))

    if total_cost == 0:
        st.info("No API calls made yet this session. Use Tab 1 or Tab 2 to start chatting.", icon="💡")
        return

    st.markdown("---")

    # ── Itemised receipt ──────────────────────────────────────────────────────
    st.markdown("### 📋 Itemised Receipt")

    rows = []

    # Code Writer line items
    cw_model_label = LIVE_MODELS.get(cw_model, {}).get("label", cw_model)
    cw_pricing     = MODEL_PRICING.get(cw_model, {"input": 0, "output": 0})
    for i, t in enumerate(cw_turns, 1):
        rows.append({
            "Tab":      "💻 Code Writer",
            "Turn":     i,
            "Model":    cw_model_label,
            "Input tok":    t.get("input_tokens", 0),
            "Output tok":   t.get("output_tokens", 0),
            "Reasoning tok":t.get("thinking_tokens", 0),
            "Tool tok":     t.get("tool_tokens", 0),
            "Total tok":    (t.get("input_tokens", 0) + t.get("output_tokens", 0)
                             + t.get("thinking_tokens", 0) + t.get("tool_tokens", 0)),
            "Cost":         t.get("total_cost", 0.0),
        })

    # Chatbot line items
    chat_model_label = LIVE_MODELS.get(chat_model, {}).get("label", chat_model)
    for i, t in enumerate(chat_turns, 1):
        rows.append({
            "Tab":      "💬 Chatbot",
            "Turn":     i,
            "Model":    chat_model_label,
            "Input tok":    t.get("input", 0),
            "Output tok":   t.get("output", 0),
            "Reasoning tok":0,
            "Tool tok":     0,
            "Total tok":    t.get("input", 0) + t.get("output", 0),
            "Cost":         t.get("cost", 0.0),
        })

    if rows:
        df = pd.DataFrame(rows)
        df["Cost ($)"] = df["Cost"].apply(fmt_cost)
        st.dataframe(
            df[["Tab","Turn","Model","Input tok","Output tok","Reasoning tok","Tool tok","Total tok","Cost ($)"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # ── Summary by model ─────────────────────────────────────────────────────
    st.markdown("### 🤖 Cost by Model")
    model_summary = {}
    for r in rows:
        m = r["Model"]
        if m not in model_summary:
            model_summary[m] = {"turns": 0, "tokens": 0, "cost": 0.0}
        model_summary[m]["turns"]  += 1
        model_summary[m]["tokens"] += r["Total tok"]
        model_summary[m]["cost"]   += r["Cost"]

    summary_rows = [
        {"Model": m, "Turns": v["turns"], "Total Tokens": f"{v['tokens']:,}",
         "Total Cost": fmt_cost(v["cost"]),
         "Avg Cost/Turn": fmt_cost(v["cost"] / v["turns"] if v["turns"] else 0)}
        for m, v in model_summary.items()
    ]
    st.dataframe(pd.DataFrame(summary_rows).set_index("Model"), use_container_width=True)

    # ── Cost breakdown charts ─────────────────────────────────────────────────
    st.markdown("### 📊 Breakdown Charts")
    col_a, col_b = st.columns(2)

    with col_a:
        if len(model_summary) > 0:
            fig = px.pie(
                names=list(model_summary.keys()),
                values=[v["cost"] for v in model_summary.values()],
                title="Cost Split by Model",
                color=list(model_summary.keys()),
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        tab_summary = {"💻 Code Writer": cw_cost, "💬 Chatbot": chat_cost}
        fig2 = px.bar(
            x=list(tab_summary.keys()),
            y=list(tab_summary.values()),
            title="Cost by Tab",
            labels={"x": "Tab", "y": "Cost ($)"},
            color=list(tab_summary.keys()),
            color_discrete_sequence=["#e07b39", "#4285F4"],
            text_auto=".4f",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Token category breakdown ──────────────────────────────────────────────
    if cw_turns:
        st.markdown("### 🔬 Code Writer Token Categories (all turns)")
        total_toks = cw_in + cw_out + cw_think + cw_tool
        cat_df = pd.DataFrame({
            "Category": ["📥 Input", "📤 Output", "🧠 Reasoning", "🔧 Tool Use"],
            "Tokens":   [cw_in, cw_out, cw_think, cw_tool],
            "Cost ($)": [
                fmt_cost(token_cost(cw_in,    cw_pricing["input"])),
                fmt_cost(token_cost(cw_out,   cw_pricing["output"])),
                fmt_cost(token_cost(cw_think, cw_pricing["input"])),
                fmt_cost(token_cost(cw_tool,  cw_pricing["input"])),
            ],
            "% of tokens": [f"{t/total_toks*100:.1f}%" if total_toks else "0%"
                            for t in [cw_in, cw_out, cw_think, cw_tool]],
        })
        st.dataframe(cat_df.set_index("Category"), use_container_width=True)

    # ── Cost over time ────────────────────────────────────────────────────────
    if len(rows) > 1:
        st.markdown("### 📈 Cumulative Cost Over All Turns")
        running, timeline = 0.0, []
        for r in rows:
            running += r["Cost"]
            timeline.append({
                "Turn": f"{r['Tab']} T{r['Turn']}",
                "Cumulative Cost ($)": running,
                "Tab": r["Tab"],
            })
        fig3 = px.line(pd.DataFrame(timeline), x="Turn", y="Cumulative Cost ($)",
                       color="Tab", markers=True,
                       color_discrete_map={"💻 Code Writer": "#e07b39", "💬 Chatbot": "#4285F4"})
        fig3.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Session footer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Session started: this browser tab opened &nbsp;·&nbsp; "
        f"Data resets if you refresh the page &nbsp;·&nbsp; "
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if st.button("🗑️ Clear All Session Data", type="secondary"):
        for key in ["cw_messages","cw_turn_data","cw_cumulative_cost","cw_model_key",
                    "chat_messages","chat_total_input_tokens","chat_total_output_tokens",
                    "chat_total_cost","chat_per_turn","chat_model_key"]:
            st.session_state.pop(key, None)
        st.rerun()


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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💻 Code Writer (MCP)",
        "💬 Chatbot + Context",
        "🏢 Enterprise Calculator",
        "🔍 RAG vs Plain LLM",
        "📖 Pricing Reference",
        "🧾 Session Cost",
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
    with tab6:
        tab_session_cost()


if __name__ == "__main__":
    main()
