"""
AI Token Cost Explorer — Streamlit App
• Tab 1 & 2: Live Anthropic API calls (needs ANTHROPIC_API_KEY)
• Tab 3 & 4: Pure cost math — all providers, no extra API keys required
"""

import json
import anthropic
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Model pricing config (USD per 1M tokens, June 2025) ──────────────────────
# No API keys needed for tabs 3 & 4 — purely mathematical comparisons.
MODEL_PRICING = {
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-opus-4-8": {
        "name": "Claude Opus 4.8",
        "provider": "Anthropic",
        "input": 5.00,
        "output": 25.00,
        "context_window": 1_000_000,
        "notes": "Most capable Anthropic model",
    },
    "claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6",
        "provider": "Anthropic",
        "input": 3.00,
        "output": 15.00,
        "context_window": 1_000_000,
        "notes": "Best speed/intelligence balance",
    },
    "claude-haiku-4-5": {
        "name": "Claude Haiku 4.5",
        "provider": "Anthropic",
        "input": 1.00,
        "output": 5.00,
        "context_window": 200_000,
        "notes": "Fastest & cheapest Anthropic",
    },
    # ── Google ───────────────────────────────────────────────────────────────
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "provider": "Google",
        "input": 0.15,
        "output": 0.60,
        "context_window": 1_000_000,
        "notes": "Best Google price/perf; thinking optional",
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "provider": "Google",
        "input": 1.25,
        "output": 10.00,
        "context_window": 1_000_000,
        "notes": "Top Google model (<200K ctx tier)",
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash",
        "provider": "Google",
        "input": 0.075,
        "output": 0.30,
        "context_window": 1_000_000,
        "notes": "Very cheap, high volume",
    },
    # ── Mistral ──────────────────────────────────────────────────────────────
    "mistral-large-2": {
        "name": "Mistral Large 2",
        "provider": "Mistral",
        "input": 2.00,
        "output": 6.00,
        "context_window": 128_000,
        "notes": "Flagship Mistral model",
    },
    "mistral-small-3.1": {
        "name": "Mistral Small 3.1",
        "provider": "Mistral",
        "input": 0.10,
        "output": 0.30,
        "context_window": 128_000,
        "notes": "Low-cost, multilingual",
    },
    "mistral-nemo": {
        "name": "Mistral Nemo",
        "provider": "Mistral",
        "input": 0.15,
        "output": 0.15,
        "context_window": 128_000,
        "notes": "12B model, Apache 2 license",
    },
    "codestral": {
        "name": "Codestral",
        "provider": "Mistral",
        "input": 0.30,
        "output": 0.90,
        "context_window": 256_000,
        "notes": "Code specialist",
    },
    # ── xAI / Grok ───────────────────────────────────────────────────────────
    "grok-3": {
        "name": "Grok 3",
        "provider": "xAI",
        "input": 3.00,
        "output": 15.00,
        "context_window": 131_072,
        "notes": "xAI flagship",
    },
    "grok-3-mini": {
        "name": "Grok 3 Mini",
        "provider": "xAI",
        "input": 0.30,
        "output": 0.50,
        "context_window": 131_072,
        "notes": "Fast & cheap xAI model",
    },
    "grok-2": {
        "name": "Grok 2",
        "provider": "xAI",
        "input": 2.00,
        "output": 10.00,
        "context_window": 131_072,
        "notes": "Previous xAI flagship",
    },
    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4o": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "input": 2.50,
        "output": 10.00,
        "context_window": 128_000,
        "notes": "OpenAI flagship",
    },
    "gpt-4o-mini": {
        "name": "GPT-4o mini",
        "provider": "OpenAI",
        "input": 0.15,
        "output": 0.60,
        "context_window": 128_000,
        "notes": "Cheap & fast OpenAI",
    },
    "o3": {
        "name": "o3",
        "provider": "OpenAI",
        "input": 2.00,
        "output": 8.00,
        "context_window": 200_000,
        "notes": "OpenAI reasoning model",
    },
    "o4-mini": {
        "name": "o4-mini",
        "provider": "OpenAI",
        "input": 1.10,
        "output": 4.40,
        "context_window": 200_000,
        "notes": "Fast OpenAI reasoning model",
    },
    # ── Meta / Llama (via API providers) ─────────────────────────────────────
    "llama-3.3-70b": {
        "name": "Llama 3.3 70B",
        "provider": "Meta (via API)",
        "input": 0.23,
        "output": 0.40,
        "context_window": 128_000,
        "notes": "Via Groq / Together / Fireworks",
    },
    "llama-3.1-405b": {
        "name": "Llama 3.1 405B",
        "provider": "Meta (via API)",
        "input": 0.80,
        "output": 0.80,
        "context_window": 128_000,
        "notes": "Largest open Llama, via API",
    },
    # ── Cohere ───────────────────────────────────────────────────────────────
    "command-r-plus": {
        "name": "Command R+",
        "provider": "Cohere",
        "input": 2.50,
        "output": 10.00,
        "context_window": 128_000,
        "notes": "Cohere flagship RAG model",
    },
    "command-r": {
        "name": "Command R",
        "provider": "Cohere",
        "input": 0.15,
        "output": 0.60,
        "context_window": 128_000,
        "notes": "Cohere efficient model",
    },
}

PROVIDER_COLORS = {
    "Anthropic": "#e07b39",
    "Google": "#4285F4",
    "Mistral": "#FF7000",
    "xAI": "#1DA1F2",
    "OpenAI": "#10A37F",
    "Meta (via API)": "#0668E1",
    "Cohere": "#39C5BB",
}

ALL_PROVIDERS = sorted({v["provider"] for v in MODEL_PRICING.values()})

RAG_INFRA = {
    "vector_db_monthly": 70,
    "embedding_per_1m_tokens": 0.02,
    "avg_chunks_retrieved": 5,
    "avg_chunk_tokens": 300,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def token_cost(tokens: int, price_per_million: float) -> float:
    return tokens * price_per_million / 1_000_000


def calc_cost(input_tok: int, output_tok: int, model: str):
    p = MODEL_PRICING[model]
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


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def provider_color(provider: str) -> str:
    return PROVIDER_COLORS.get(provider, "#888888")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Code Writer (MCP-powered)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_MCP_TOOLS = [
    {
        "name": "read_file",
        "description": "Read file contents from the project (MCP filesystem tool)",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search documentation database for examples (MCP docs tool)",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files in a directory (MCP filesystem tool)",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]

MOCK_TOOL_RESPONSES = {
    "read_file": "# Example file content\nclass MyClass:\n    pass\n",
    "search_docs": "Found 3 relevant examples in documentation.",
    "list_directory": "['main.py', 'utils.py', 'tests/']",
}


def run_code_writer(task: str):
    client = get_client()
    system = (
        "You are an expert software engineer. Write clean, well-structured code. "
        "Use the available MCP tools when helpful. Always explain your code."
    )
    messages = [{"role": "user", "content": task}]

    # Count tokens before the call
    count_resp = client.messages.count_tokens(
        model="claude-sonnet-4-6",
        system=system,
        tools=MOCK_MCP_TOOLS,
        messages=messages,
    )

    thinking_tokens = 0
    tool_tokens = 0
    code_output = ""
    thinking_text = ""
    tool_calls_made = []

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system,
        tools=MOCK_MCP_TOOLS,
        messages=messages,
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    yield {"type": "status", "text": "🧠 Claude is reasoning…"}
                elif event.content_block.type == "tool_use":
                    yield {"type": "status", "text": f"🔧 MCP call: {event.content_block.name}"}
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    thinking_text += event.delta.thinking
                elif event.delta.type == "text_delta":
                    code_output += event.delta.text
                    yield {"type": "text_chunk", "text": event.delta.text}
        final = stream.get_final_message()

    for block in final.content:
        if block.type == "tool_use":
            tool_calls_made.append(block.name)
            tool_input_json = json.dumps(block.input)
            tool_result = MOCK_TOOL_RESPONSES.get(block.name, "OK")
            tool_tokens += len(tool_input_json.split()) * 2 + len(tool_result.split()) * 2
        if block.type == "thinking":
            thinking_tokens += len(getattr(block, "thinking", "").split()) * 2

    usage = final.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens

    input_cost = token_cost(input_tokens, MODEL_PRICING["claude-sonnet-4-6"]["input"])
    output_cost = token_cost(output_tokens, MODEL_PRICING["claude-sonnet-4-6"]["output"])
    tool_cost = token_cost(tool_tokens, MODEL_PRICING["claude-sonnet-4-6"]["input"])

    yield {
        "type": "done",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "tool_tokens": tool_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "tool_cost": tool_cost,
        "total_cost": input_cost + output_cost + tool_cost,
        "code": code_output,
        "thinking": thinking_text,
        "tools_called": tool_calls_made,
    }


def tab_code_writer():
    st.header("💻 Code Writer")
    st.caption(
        "Requires **ANTHROPIC_API_KEY**. Claude Sonnet 4.6 writes code with "
        "adaptive thinking + MCP tool simulation."
    )
    st.info("🔑 Only Anthropic API key needed here — comparison tabs work without any keys.", icon="ℹ️")

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
        collected_code = ""

        for chunk in run_code_writer(task):
            if chunk["type"] == "status":
                status_ph.info(chunk["text"])
            elif chunk["type"] == "text_chunk":
                collected_code += chunk["text"]
                code_ph.code(collected_code, language="python")
            elif chunk["type"] == "done":
                final_data = chunk

        status_ph.empty()

        if final_data:
            st.session_state.cw_cumulative_cost += final_data["total_cost"]
            st.session_state.cw_history.append(final_data)

            st.markdown("### 📊 Token Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🧠 Reasoning", f"{final_data['thinking_tokens']:,}")
            c2.metric("📥 Input", f"{final_data['input_tokens']:,}")
            c3.metric("📤 Output", f"{final_data['output_tokens']:,}")
            c4.metric("🔧 Tool Use", f"{final_data['tool_tokens']:,}")

            st.markdown("### 💰 Cost Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Input Cost", fmt_cost(final_data["input_cost"]))
            c2.metric("Output Cost", fmt_cost(final_data["output_cost"]))
            c3.metric("Tool Cost", fmt_cost(final_data["tool_cost"]))
            c4.metric("This Request", fmt_cost(final_data["total_cost"]))

            st.metric(
                f"🏦 Cumulative Cost (Iteration {st.session_state.cw_iterations})",
                fmt_cost(st.session_state.cw_cumulative_cost),
                delta=fmt_cost(final_data["total_cost"]),
            )

            if final_data["tools_called"]:
                st.info(f"🔧 MCP tools invoked: {', '.join(final_data['tools_called'])}")

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

def tab_chatbot():
    st.header("💬 Chatbot + Context Window Tracker")
    st.caption("Requires **ANTHROPIC_API_KEY**. Live chat with real-time context window monitoring.")
    st.info("🔑 Only Anthropic API key needed here — comparison tabs work without any keys.", icon="ℹ️")

    model = st.sidebar.selectbox(
        "Chat Model",
        options=["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-8"],
        format_func=lambda k: MODEL_PRICING[k]["name"],
        key="chat_model",
    )
    ctx_window = MODEL_PRICING[model]["context_window"]

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        st.session_state.chat_total_input_tokens = 0
        st.session_state.chat_total_output_tokens = 0
        st.session_state.chat_total_cost = 0.0
        st.session_state.chat_per_turn = []

    client = get_client()

    total_conv_tokens = 0
    if st.session_state.chat_messages:
        cr = client.messages.count_tokens(model=model, messages=st.session_state.chat_messages)
        total_conv_tokens = cr.input_tokens

    pct = min(total_conv_tokens / ctx_window, 1.0)

    st.sidebar.markdown("### 📊 Context Window")
    st.sidebar.progress(pct)
    st.sidebar.write(f"**{total_conv_tokens:,}** / **{ctx_window:,}** ({pct*100:.1f}%)")

    if pct >= 0.9:
        st.sidebar.error("🚨 Context window nearly full!")
    elif pct >= 0.7:
        st.sidebar.warning("⚠️ Getting expensive! Context 70%+ full.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Conversation Cost")
    st.sidebar.metric("Total Cost", fmt_cost(st.session_state.chat_total_cost))
    st.sidebar.metric("Total Input Tokens", f"{st.session_state.chat_total_input_tokens:,}")
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
                        cc1.metric("Input", f"{t['input']:,}")
                        cc2.metric("Output", f"{t['output']:,}")
                        cc3.metric("Cost", fmt_cost(t["cost"]))

    if prompt := st.chat_input("Type a message…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            ph = st.empty()
            full_response = ""
            with client.messages.stream(
                model=model, max_tokens=2048, messages=st.session_state.chat_messages
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    ph.markdown(full_response + "▌")
                final = stream.get_final_message()
            ph.markdown(full_response)

            in_tok = final.usage.input_tokens
            out_tok = final.usage.output_tokens
            _, _, cost = calc_cost(in_tok, out_tok, model)

            st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
            st.session_state.chat_total_input_tokens += in_tok
            st.session_state.chat_total_output_tokens += out_tok
            st.session_state.chat_total_cost += cost
            st.session_state.chat_per_turn.append({"input": in_tok, "output": out_tok, "cost": cost})

            with st.expander("Token details", expanded=True):
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("This input", f"{in_tok:,}")
                cc2.metric("This output", f"{out_tok:,}")
                cc3.metric("This cost", fmt_cost(cost))

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Multi-Provider Cost Calculator
# ─────────────────────────────────────────────────────────────────────────────

def tab_enterprise():
    st.header("🏢 Multi-Provider Cost Calculator")
    st.success(
        "✅ **No API keys required** — all calculations are purely mathematical using published pricing.",
        icon="✅",
    )

    # ── Provider / model filters ───────────────────────────────────────────
    st.subheader("🔧 Configure Your Workload")
    col1, col2, col3 = st.columns(3)
    with col1:
        qps = st.number_input("Queries per second (QPS)", 0.01, 10000.0, 1.0, step=0.5)
        avg_input = st.number_input("Avg input tokens / query", 100, 200_000, 500, step=100)
    with col2:
        avg_output = st.number_input("Avg output tokens / query", 50, 8000, 300, step=50)
        peak_mult = st.slider("Peak traffic multiplier", 1.0, 20.0, 2.0, 0.5)
    with col3:
        selected_providers = st.multiselect(
            "Filter providers",
            options=ALL_PROVIDERS,
            default=ALL_PROVIDERS,
        )

    st.markdown("---")

    # Filter models
    filtered = {
        k: v for k, v in MODEL_PRICING.items() if v["provider"] in selected_providers
    }

    queries_per_month = qps * 86400 * 30

    rows = []
    for key, info in filtered.items():
        q_cost = (
            token_cost(avg_input, info["input"]) + token_cost(avg_output, info["output"])
        )
        rows.append(
            {
                "Model": info["name"],
                "Provider": info["provider"],
                "Input $/1M": f"${info['input']:.3f}",
                "Output $/1M": f"${info['output']:.3f}",
                "$ / query": fmt_cost(q_cost),
                "$ / hour": f"${q_cost * qps * 3600:,.4f}",
                "$ / day": f"${q_cost * qps * 86400:,.2f}",
                "$ / month": f"${q_cost * queries_per_month:,.2f}",
                "Context Window": f"{info['context_window']:,}",
                "Notes": info["notes"],
                "_q": q_cost,
            }
        )

    df = pd.DataFrame(rows).sort_values("_q")
    df_display = df.drop(columns=["_q"]).set_index("Model")

    st.subheader(f"📊 Cost Comparison — {avg_input} input / {avg_output} output tokens @ {qps} QPS")
    st.dataframe(df_display, use_container_width=True)

    # ── Bar chart: monthly cost ───────────────────────────────────────────
    df["Monthly Cost ($)"] = df["_q"] * queries_per_month
    fig = px.bar(
        df.sort_values("Monthly Cost ($)"),
        x="Model",
        y="Monthly Cost ($)",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        title=f"Monthly Cost at {qps} QPS ({avg_input} in / {avg_output} out tokens)",
        text_auto=".2f",
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: price vs context window ─────────────────────────────────
    st.subheader("💡 Price vs Context Window")
    fig2 = px.scatter(
        df,
        x="Context Window",
        y="Monthly Cost ($)",
        color="Provider",
        text="Model",
        size=[1] * len(df),
        color_discrete_map=PROVIDER_COLORS,
        title="Context Window vs Monthly Cost",
        log_x=True,
    )
    fig2.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Peak traffic ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"⚡ Peak Traffic ({peak_mult}× baseline = {qps * peak_mult:.1f} QPS)")

    cheapest_key = df.sort_values("_q").iloc[0]
    most_capable = df[df["Provider"] == "Anthropic"].sort_values("_q", ascending=False).iloc[0] if "Anthropic" in selected_providers and not df[df["Provider"] == "Anthropic"].empty else df.sort_values("_q", ascending=False).iloc[0]

    peak_qps = qps * peak_mult
    peak_monthly_cheap = cheapest_key["_q"] * peak_qps * 86400 * 30
    peak_monthly_capable = most_capable["_q"] * peak_qps * 86400 * 30

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak QPS", f"{peak_qps:.1f}")
    c2.metric(f"Cheapest ({cheapest_key['Model']})", f"${peak_monthly_cheap:,.2f}/mo")
    c3.metric(f"Highest Tier", f"${peak_monthly_capable:,.2f}/mo")

    # ── Optimization tips ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Optimization Suggestions")

    if "claude-sonnet-4-6" in filtered and "claude-haiku-4-5" in filtered:
        sonnet_m = token_cost(avg_input, 3.00) + token_cost(avg_output, 15.00)
        haiku_m = token_cost(avg_input, 1.00) + token_cost(avg_output, 5.00)
        savings_routing = (sonnet_m - haiku_m) * queries_per_month * 0.6
        tips = [
            {"Tip": "Route 60% simple queries to Claude Haiku 4.5",
             "Estimated Savings": f"${savings_routing:,.2f}/mo", "Risk": "Low"},
        ]
    else:
        tips = []

    tips += [
        {"Tip": "Enable Anthropic prompt caching (repeated context)",
         "Estimated Savings": "Up to 90% on cached tokens", "Risk": "None"},
        {"Tip": "Use Anthropic Batch API for async workloads",
         "Estimated Savings": "50% off all Anthropic models", "Risk": "Latency only"},
        {"Tip": "Trim input tokens 20% via prompt engineering",
         "Estimated Savings": "~20% input cost", "Risk": "Low"},
        {"Tip": f"Switch to cheapest model ({df.sort_values('_q').iloc[0]['Model']})",
         "Estimated Savings": f"Up to {(1 - df.sort_values('_q').iloc[0]['_q'] / max(df['_q'])) * 100:.0f}% vs most expensive",
         "Risk": "Quality regression — test first"},
    ]

    st.dataframe(pd.DataFrame(tips), use_container_width=True)

    # ── Side-by-side: cheapest, mid, most capable ─────────────────────────
    if len(df) >= 3:
        st.markdown("---")
        st.subheader("🏆 Quick Picks")
        sorted_df = df.sort_values("_q")
        c1, c2, c3 = st.columns(3)
        c1.success(f"**Cheapest**\n\n{sorted_df.iloc[0]['Model']}\n\n${sorted_df.iloc[0]['_q'] * queries_per_month:,.2f}/mo")
        mid = len(sorted_df) // 2
        c2.info(f"**Mid-range**\n\n{sorted_df.iloc[mid]['Model']}\n\n${sorted_df.iloc[mid]['_q'] * queries_per_month:,.2f}/mo")
        c3.warning(f"**Most Expensive**\n\n{sorted_df.iloc[-1]['Model']}\n\n${sorted_df.iloc[-1]['_q'] * queries_per_month:,.2f}/mo")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RAG vs Plain LLM Cost
# ─────────────────────────────────────────────────────────────────────────────

def tab_rag():
    st.header("🔍 RAG vs Plain LLM Cost")
    st.success("✅ **No API keys required** — pure cost math across all providers.", icon="✅")

    col1, col2 = st.columns(2)
    with col1:
        qps = st.number_input("QPS", 0.01, 10000.0, 1.0, 0.1, key="rag_qps")
        avg_context_kb = st.slider("Full context size (KB)", 1, 500, 20)
        model_key = st.selectbox(
            "LLM Model",
            options=list(MODEL_PRICING.keys()),
            format_func=lambda k: f"{MODEL_PRICING[k]['name']} ({MODEL_PRICING[k]['provider']})",
            key="rag_model",
            index=list(MODEL_PRICING.keys()).index("claude-sonnet-4-6"),
        )
    with col2:
        avg_q_tokens = st.number_input("Avg question tokens", 50, 500, 100, key="rag_q")
        avg_a_tokens = st.number_input("Avg answer tokens", 100, 4000, 400, key="rag_a")
        rag_infra = st.number_input("RAG infra cost ($/mo)", 0, 5000, 70, key="rag_infra")

    chars_per_token = 4
    avg_context_tokens = (avg_context_kb * 1024) // chars_per_token
    retrieved_tokens = RAG_INFRA["avg_chunks_retrieved"] * RAG_INFRA["avg_chunk_tokens"]
    queries_per_month = qps * 86400 * 30
    model_info = MODEL_PRICING[model_key]

    # Plain LLM
    plain_in = avg_context_tokens + avg_q_tokens
    plain_out = avg_a_tokens
    plain_q = token_cost(plain_in, model_info["input"]) + token_cost(plain_out, model_info["output"])
    plain_monthly = plain_q * queries_per_month

    # RAG
    rag_embed_q = token_cost(avg_q_tokens, RAG_INFRA["embedding_per_1m_tokens"])
    rag_llm_in = retrieved_tokens + avg_q_tokens
    rag_llm_out = avg_a_tokens
    rag_llm_q = token_cost(rag_llm_in, model_info["input"]) + token_cost(rag_llm_out, model_info["output"])
    rag_q = rag_embed_q + rag_llm_q
    rag_llm_monthly = rag_llm_q * queries_per_month
    rag_embed_monthly = rag_embed_q * queries_per_month
    rag_total_monthly = rag_llm_monthly + rag_embed_monthly + rag_infra
    monthly_savings = plain_monthly - rag_total_monthly

    breakeven = (
        rag_infra / (plain_monthly - rag_llm_monthly - rag_embed_monthly)
        if (plain_monthly - rag_llm_monthly - rag_embed_monthly) > 0
        else float("inf")
    )

    st.markdown("---")
    st.subheader(f"📊 Results — {model_info['name']} @ {qps} QPS")

    c1, c2, c3 = st.columns(3)
    c1.metric("Plain LLM / month", f"${plain_monthly:,.2f}")
    c2.metric("RAG / month", f"${rag_total_monthly:,.2f}")
    saved = plain_monthly - rag_total_monthly
    c3.metric(
        "Savings with RAG",
        f"${abs(saved):,.2f}/mo",
        delta="cheaper ✅" if saved > 0 else "more expensive ❌",
    )

    if saved > 0 and breakeven != float("inf"):
        st.info(f"📅 Break-even: **{breakeven:.1f} months** to recover RAG infra at this traffic level.")
    elif saved <= 0:
        st.warning(
            "⚠️ RAG is **not** cheaper at this context size / QPS. "
            "The full context is small enough that sending it each time beats RAG infra overhead."
        )

    # Stacked bar
    bar_data = pd.DataFrame({
        "Approach": ["Plain LLM", "RAG"],
        "LLM Input": [
            token_cost(plain_in, model_info["input"]) * queries_per_month,
            token_cost(rag_llm_in, model_info["input"]) * queries_per_month,
        ],
        "LLM Output": [
            token_cost(plain_out, model_info["output"]) * queries_per_month,
            token_cost(rag_llm_out, model_info["output"]) * queries_per_month,
        ],
        "Embedding": [0, rag_embed_monthly],
        "Infra": [0, float(rag_infra)],
    })

    fig = go.Figure()
    for col, color in zip(
        ["LLM Input", "LLM Output", "Embedding", "Infra"],
        ["#4C78A8", "#72B7B2", "#F58518", "#E45756"],
    ):
        fig.add_trace(go.Bar(name=col, x=bar_data["Approach"], y=bar_data[col], marker_color=color))
    fig.update_layout(barmode="stack", title="Monthly Cost Breakdown", yaxis_title="Cost ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Cross-model comparison
    st.markdown("---")
    st.subheader("🔄 RAG Savings Across All Models")
    cross_rows = []
    for k, info in MODEL_PRICING.items():
        p_monthly = (token_cost(plain_in, info["input"]) + token_cost(plain_out, info["output"])) * queries_per_month
        r_llm = (token_cost(rag_llm_in, info["input"]) + token_cost(rag_llm_out, info["output"])) * queries_per_month
        r_total = r_llm + rag_embed_monthly + rag_infra
        s = p_monthly - r_total
        cross_rows.append({
            "Model": info["name"],
            "Provider": info["provider"],
            "Plain LLM/mo": f"${p_monthly:,.2f}",
            "RAG/mo": f"${r_total:,.2f}",
            "Savings": f"${abs(s):,.2f}" if s > 0 else f"-${abs(s):,.2f}",
            "Worth it?": "✅ Yes" if s > 0 else "❌ No",
            "_savings": s,
        })

    df_cross = pd.DataFrame(cross_rows).sort_values("_savings", ascending=False)
    st.dataframe(df_cross.drop(columns=["_savings"]).set_index("Model"), use_container_width=True)

    fig3 = px.bar(
        df_cross,
        x="Model",
        y="_savings",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        title="Monthly Savings from RAG by Model",
        labels={"_savings": "Monthly Savings ($)"},
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig3.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig3, use_container_width=True)

    # Detail
    st.subheader("🔢 Per-Query Detail")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["Input tokens", "Output tokens", "LLM cost", "Extra (embed/infra amortized)", "Total / query"],
            "Plain LLM": [f"{plain_in:,}", f"{plain_out:,}", fmt_cost(plain_q), "$0", fmt_cost(plain_q)],
            "RAG": [f"{rag_llm_in:,}", f"{rag_llm_out:,}", fmt_cost(rag_llm_q), fmt_cost(rag_embed_q), fmt_cost(rag_q)],
        }).set_index("Metric"),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Model Pricing Reference
# ─────────────────────────────────────────────────────────────────────────────

def tab_reference():
    st.header("📖 Model Pricing Reference")
    st.caption(
        "All 20 models across 7 providers — published pricing as of June 2025. "
        "No API keys needed."
    )

    provider_filter = st.multiselect(
        "Filter by provider", options=ALL_PROVIDERS, default=ALL_PROVIDERS
    )

    rows = []
    for k, v in MODEL_PRICING.items():
        if v["provider"] not in provider_filter:
            continue
        rows.append({
            "Model": v["name"],
            "Provider": v["provider"],
            "Input $/1M": v["input"],
            "Output $/1M": v["output"],
            "Context (tokens)": v["context_window"],
            "Notes": v["notes"],
        })

    df = pd.DataFrame(rows).sort_values(["Provider", "Input $/1M"])

    # Color by provider for the table
    st.dataframe(df.set_index("Model"), use_container_width=True)

    # Pricing scatter
    fig = px.scatter(
        df,
        x="Input $/1M",
        y="Output $/1M",
        color="Provider",
        text="Model",
        color_discrete_map=PROVIDER_COLORS,
        title="Input vs Output Pricing (per 1M tokens)",
        size=[1] * len(df),
    )
    fig.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig, use_container_width=True)

    # Input price bar
    fig2 = px.bar(
        df.sort_values("Input $/1M"),
        x="Model",
        y="Input $/1M",
        color="Provider",
        color_discrete_map=PROVIDER_COLORS,
        title="Input Price per 1M Tokens (sorted cheapest → most expensive)",
    )
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
        "Compare LLM costs across **20 models** from Anthropic, Google, Mistral, xAI, OpenAI, Meta, and Cohere. "
        "Tabs 3–5 require **no API keys** — pure math on published pricing."
    )

    # ── Dashboard: model & token-counting overview ───────────────────────────
    with st.expander("ℹ️ How each tab works — models used & token counting method", expanded=True):
        dash_data = [
            {
                "Tab": "💻 Tab 1 — Code Writer",
                "API Key Needed": "✅ ANTHROPIC_API_KEY",
                "Reasoning Model": "Claude Sonnet 4.6\n(`claude-sonnet-4-6`)",
                "Token Counting Method": "`client.messages.count_tokens()` called before each stream\n(Anthropic SDK — exact pre-flight count)",
                "Token Categories Tracked": "🧠 Reasoning · 📥 Input · 📤 Output · 🔧 Tool Use",
                "Notes": "Adaptive thinking enabled; MCP tools are simulated (mock responses)",
            },
            {
                "Tab": "💬 Tab 2 — Chatbot",
                "API Key Needed": "✅ ANTHROPIC_API_KEY",
                "Reasoning Model": "User-selectable:\nOpus 4.8 · Sonnet 4.6 · Haiku 4.5",
                "Token Counting Method": "`client.messages.count_tokens()` on full conversation history each render\n(live — updates as history grows)",
                "Token Categories Tracked": "📥 Input · 📤 Output (per-turn + running total)",
                "Notes": "Context window % bar turns yellow at 70%, red at 90%",
            },
            {
                "Tab": "🏢 Tab 3 — Enterprise Calculator",
                "API Key Needed": "❌ None",
                "Reasoning Model": "No live model call",
                "Token Counting Method": "Pure math: `tokens × price_per_million / 1,000,000`\nusing MODEL_PRICING config dict",
                "Token Categories Tracked": "📥 Input · 📤 Output (configurable counts)",
                "Notes": "Compares all 20 models; filter by provider",
            },
            {
                "Tab": "🔍 Tab 4 — RAG vs Plain LLM",
                "API Key Needed": "❌ None",
                "Reasoning Model": "No live model call",
                "Token Counting Method": "Pure math using MODEL_PRICING config dict\n+ embedding cost ($0.02/1M) + infra overhead",
                "Token Categories Tracked": "📥 LLM Input · 📤 LLM Output · 🔎 Embedding · 🏗️ Infra",
                "Notes": "Cross-model RAG break-even comparison for all 20 models",
            },
            {
                "Tab": "📖 Tab 5 — Pricing Reference",
                "API Key Needed": "❌ None",
                "Reasoning Model": "No live model call",
                "Token Counting Method": "Static config — MODEL_PRICING dict (June 2025 published rates)",
                "Token Categories Tracked": "N/A — reference only",
                "Notes": "Filterable by provider; input/output scatter + bar chart",
            },
        ]

        df_dash = pd.DataFrame(dash_data).set_index("Tab")
        st.dataframe(df_dash, use_container_width=True, height=230)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("**Live token counting API**")
        c1.code("client.messages.count_tokens(\n    model=...,\n    messages=...\n)")
        c2.markdown("**Exact cost formula**")
        c2.code("tokens × price_per_million\n────────────────────────\n      1,000,000")
        c3.markdown("**Tabs 1 & 2 — need key**")
        c3.info("Set `ANTHROPIC_API_KEY`\nin your shell before\nrunning the app.")
        c4.markdown("**Tabs 3, 4, 5 — no key**")
        c4.success("Works offline.\n20 models pre-loaded\nfrom published pricing.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💻 Code Writer (MCP)",
        "💬 Chatbot + Context",
        "🏢 Enterprise Calculator",
        "🔍 RAG vs Plain LLM",
        "📖 Pricing Reference",
    ])

    with tab1:
        tab_code_writer()
    with tab2:
        tab_chatbot()
    with tab3:
        tab_enterprise()
    with tab4:
        tab_rag()
    with tab5:
        tab_reference()


if __name__ == "__main__":
    main()
