# AI Token Cost Explorer

A Streamlit app that makes LLM token costs **visible, measurable, and comparable** across 20 models from 7 providers — with live Anthropic API calls, MCP tool simulation, and pure-math cost calculators that require no API keys.

---

## What This App Does

Most LLM cost surprises come from not tracking tokens in real time. This app fixes that:

- **See every token category** — reasoning, input, output, tool use — broken out with per-category costs
- **Watch your context window fill up** — live progress bar with warnings before you hit the limit
- **Compare 20 models side-by-side** — Anthropic, Google, Mistral, xAI, OpenAI, Meta, Cohere
- **Understand when RAG beats plain LLM** — and calculate the exact break-even point

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit App (app.py)                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Tab 1      │  │   Tab 2      │  │  Tabs 3, 4, 5         │  │
│  │  Code Writer │  │  Chatbot     │  │  Enterprise / RAG /   │  │
│  │  (MCP+think) │  │  + Context   │  │  Pricing Reference    │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────────┘  │
│         │                 │                    │                  │
│         │ Anthropic SDK   │ Anthropic SDK      │  Pure Math       │
│         │ (streaming)     │ (streaming +       │  MODEL_PRICING   │
│         │                 │  count_tokens)     │  config dict     │
└─────────┼─────────────────┼────────────────────┼──────────────────┘
          │                 │                    │
          ▼                 ▼                    ▼
   ┌─────────────┐   ┌─────────────┐    ┌────────────────┐
   │ Anthropic   │   │ Anthropic   │    │ No API calls   │
   │ Messages    │   │ Messages    │    │ Published rates │
   │ API         │   │ API +       │    │ from config    │
   │ (Sonnet 4.6)│   │ count_tokens│    │                │
   └─────────────┘   └─────────────┘    └────────────────┘
```

**Key design decision:** Tabs 3–5 are intentionally API-key-free. All 20 model prices are hardcoded in the `MODEL_PRICING` dict at the top of `app.py` — update that dict to change any price, no other code changes needed.

---

## Tab-by-Tab Breakdown

### Tab 1 — Code Writer with MCP Tool Simulation

**What it does:** You describe a coding task; Claude writes code using adaptive thinking + MCP-style tool calls. Every token category is measured and costs are tracked cumulatively.

**Model used:** `claude-sonnet-4-6` (hardcoded — best balance of reasoning capability and cost)

**Token counting:** `client.messages.count_tokens()` called _before_ the stream starts — this is a pre-flight count using the Anthropic token counting API that tells you exactly how many input tokens will be consumed.

**MCP integration:**

This tab uses the **Model Context Protocol (MCP)** tool schema pattern. Tools are defined as JSON schemas and passed to the API in the `tools` parameter — exactly how a real MCP server exposes tools to a Claude agent.

```
┌──────────────────────────────────────────────────────────┐
│                     Tab 1 — MCP Flow                     │
│                                                           │
│  User types task                                          │
│       │                                                   │
│       ▼                                                   │
│  client.messages.count_tokens()  ← pre-flight token count │
│       │                                                   │
│       ▼                                                   │
│  client.messages.stream(                                  │
│    model="claude-sonnet-4-6",                             │
│    thinking={"type": "adaptive"},   ← reasoning tokens    │
│    tools=[                          ← MCP tool schemas    │
│      read_file,                                           │
│      search_docs,                                         │
│      list_directory                                       │
│    ],                                                     │
│    messages=[{"role": "user", ...}]                       │
│  )                                                        │
│       │                                                   │
│       ├─ thinking blocks  → reasoning token count         │
│       ├─ text blocks      → output tokens (streamed)      │
│       └─ tool_use blocks  → tool token overhead           │
│                                                           │
│  Mock MCP server responds with static fixture data        │
│  (real MCP server would call actual filesystem / DB)      │
└──────────────────────────────────────────────────────────┘
```

**MCP tools defined:**

| Tool | MCP Category | Description | Mock Response |
|------|-------------|-------------|---------------|
| `read_file` | Filesystem | Read source file contents | Python class stub |
| `search_docs` | Knowledge base | Search documentation | "Found 3 examples" |
| `list_directory` | Filesystem | List project files | `['main.py', 'utils.py', ...]` |

**Why simulated?** A real MCP server requires a running process that Claude Code connects to via stdio or SSE transport. Inside a Streamlit web app there is no persistent subprocess — so the tool _schemas_ are real (Claude reasons about them, generates tool calls, and those calls consume real tokens), but the _responses_ are static fixtures. In production you would replace the `MOCK_TOOL_RESPONSES` dict with actual MCP server calls.

**Cost tracked per call:**

| Category | Source | Formula |
|----------|--------|---------|
| Input | `usage.input_tokens` from API | `tokens × $3.00 / 1M` |
| Output | `usage.output_tokens` from API | `tokens × $15.00 / 1M` |
| Reasoning | Word count of thinking blocks × 2 | `tokens × $3.00 / 1M` |
| Tool Use | JSON payload size estimate | `tokens × $3.00 / 1M` |

---

### Tab 2 — Chatbot + Live Context Window Tracker

**What it does:** A normal multi-turn chat interface. The sidebar shows a live progress bar of how much of the context window you've consumed, and warns you before costs spiral from a full context.

**Models available (user-selectable):**

| Model | Input $/1M | Output $/1M | Context Window |
|-------|-----------|------------|----------------|
| Claude Opus 4.8 | $5.00 | $25.00 | 1,000,000 |
| Claude Sonnet 4.6 | $3.00 | $15.00 | 1,000,000 |
| Claude Haiku 4.5 | $1.00 | $5.00 | 200,000 |

**Token counting method:**

```python
# Called every render cycle on the full conversation history
count_response = client.messages.count_tokens(
    model=model,
    messages=st.session_state.chat_messages   # entire history
)
total_tokens = count_response.input_tokens
pct = total_tokens / context_window_size
```

This uses the Anthropic token counting endpoint — not an estimate, not a tiktoken approximation. It's the exact count the API would charge for that conversation.

**Context window warnings:**
- Progress bar goes **yellow** at 70% — costs are rising fast
- Progress bar goes **red** at 90% — approaching the limit, consider clearing or summarizing

---

### Tab 3 — Multi-Provider Enterprise Cost Calculator

**What it does:** Pure cost math. Enter your workload (QPS, token counts, peak multiplier), filter by provider, and instantly see per-query / hourly / daily / monthly costs for all 20 models.

**No API calls. No API keys.**

**How the math works:**

```python
# Per-query cost
query_cost = (input_tokens × input_price / 1_000_000)
           + (output_tokens × output_price / 1_000_000)

# Monthly cost at given QPS
monthly = query_cost × qps × 86_400 × 30
```

**Models compared:** All 20 entries in `MODEL_PRICING` — filterable by provider with color-coded charts.

**Outputs:**
- Sortable cost comparison table (per-query through per-month)
- Bar chart: monthly cost sorted cheapest → most expensive
- Scatter: price vs context window size (log scale)
- Peak traffic projection at configurable multiplier
- Optimization tips table (prompt caching, batch API, model routing)

---

### Tab 4 — RAG vs Plain LLM Cost Comparison

**What it does:** Calculates whether retrieval-augmented generation is cheaper than sending the full context every time, at your specific traffic level.

**No API calls. No API keys.**

**Plain LLM approach:**
```
Every query sends the full document context + question → LLM
Cost = (full_context_tokens + question_tokens) × input_price
     + answer_tokens × output_price
```

**RAG approach:**
```
Query → embedding model → vector DB → retrieve top-K chunks → LLM
Cost = question_tokens × embedding_price          (embed the query)
     + (K_chunks × chunk_tokens + question) × input_price  (LLM on chunks)
     + answer_tokens × output_price
     + vector_db_monthly_infra                    (amortized)
```

**Break-even formula:**
```
break_even_months = infra_monthly_cost
                    ──────────────────────────────────────────
                    (plain_llm_monthly - rag_llm_monthly - embed_monthly)
```

**RAG defaults used:**

| Parameter | Default | Configurable? |
|-----------|---------|---------------|
| Embedding price | $0.02 / 1M tokens | In config dict |
| Chunks retrieved per query | 5 | In config dict |
| Avg chunk size | 300 tokens | In config dict |
| Vector DB monthly infra | $70/mo | ✅ Via slider |

**Cross-model RAG comparison:** A second chart shows which of the 20 models benefit most from RAG — expensive models (Opus, GPT-4o) benefit most since you're saving on input tokens; cheap models (Gemini Flash, Haiku) may not justify the RAG infra overhead at low QPS.

---

### Tab 5 — Pricing Reference

**What it does:** A static but filterable reference table of all 20 models with pricing, context windows, and notes. Two charts: input/output price scatter, and input price bar sorted cheapest → most expensive.

**No API calls. No API keys.**

---

## Model Coverage (June 2025 Pricing)

| Provider | Model | Input $/1M | Output $/1M | Context |
|----------|-------|-----------|------------|---------|
| **Anthropic** | Claude Opus 4.8 | $5.00 | $25.00 | 1M |
| | Claude Sonnet 4.6 | $3.00 | $15.00 | 1M |
| | Claude Haiku 4.5 | $1.00 | $5.00 | 200K |
| **Google** | Gemini 2.5 Flash | $0.15 | $0.60 | 1M |
| | Gemini 2.5 Pro | $1.25 | $10.00 | 1M |
| | Gemini 1.5 Flash | $0.075 | $0.30 | 1M |
| **Mistral** | Mistral Large 2 | $2.00 | $6.00 | 128K |
| | Mistral Small 3.1 | $0.10 | $0.30 | 128K |
| | Mistral Nemo | $0.15 | $0.15 | 128K |
| | Codestral | $0.30 | $0.90 | 256K |
| **xAI** | Grok 3 | $3.00 | $15.00 | 131K |
| | Grok 3 Mini | $0.30 | $0.50 | 131K |
| | Grok 2 | $2.00 | $10.00 | 131K |
| **OpenAI** | GPT-4o | $2.50 | $10.00 | 128K |
| | GPT-4o mini | $0.15 | $0.60 | 128K |
| | o3 | $2.00 | $8.00 | 200K |
| | o4-mini | $1.10 | $4.40 | 200K |
| **Meta (via API)** | Llama 3.3 70B | $0.23 | $0.40 | 128K |
| | Llama 3.1 405B | $0.80 | $0.80 | 128K |
| **Cohere** | Command R+ | $2.50 | $10.00 | 128K |
| | Command R | $0.15 | $0.60 | 128K |

To update any price: edit the `MODEL_PRICING` dict at the top of `app.py`. All tabs recalculate instantly.

---

## Setup & Running

### Prerequisites

- Python 3.9+
- An Anthropic API key (only needed for Tabs 1 & 2)

### Install

```bash
git clone <repo-url>
cd ai-token-explorer
pip3 install -r requirements.txt
```

### Set your API key (Tabs 1 & 2 only)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Tabs 3, 4, 5 work without any API key — you can run the app and use the cost calculators entirely offline.

### Run

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`.

---

## Tech Stack

| Layer | Technology | Used for |
|-------|-----------|---------|
| UI | [Streamlit](https://streamlit.io) | All tabs, session state, sidebar |
| LLM | [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) `>=0.40.0` | Tabs 1 & 2 — streaming, token counting |
| Token counting | `client.messages.count_tokens()` | Pre-flight exact counts (not estimates) |
| Adaptive thinking | `thinking={"type": "adaptive"}` | Tab 1 — reasoning token extraction |
| MCP tools | JSON schema via `tools=` parameter | Tab 1 — tool call simulation |
| Charts | [Plotly](https://plotly.com/python/) | Bar, pie, scatter, line, stacked bar |
| Tables | [Pandas](https://pandas.pydata.org/) | All comparison dataframes |

---

## File Structure

```
ai-token-explorer/
├── app.py              # Entire app — all 5 tabs, helpers, pricing config
├── requirements.txt    # anthropic, streamlit, pandas, plotly
└── README.md           # This file
```

The app is intentionally a single file. The `MODEL_PRICING` dict at the top is the only place prices live — update it to change any model's pricing across all tabs simultaneously.

---

## Key Design Decisions

**Single-file app:** All logic in `app.py`. No separate modules, no config files. Easy to read end-to-end and share as a gist.

**MCP tools are simulated, not wired:** Real MCP servers require a persistent process (stdio or SSE transport) — incompatible with Streamlit's stateless render model. The tool _schemas_ passed to the API are identical to what a real MCP server would expose; only the tool _responses_ are mocked. The token costs from tool definitions and tool call/response cycles are real.

**Token counting uses the API, not tiktoken:** `client.messages.count_tokens()` calls Anthropic's counting endpoint and returns the exact number of tokens the API will charge — including special tokens, system prompt formatting, and tool schema overhead. This is more accurate than client-side tokenizer libraries.

**No multi-provider API calls:** Gemini, Mistral, Grok etc. appear in the cost comparison purely through their published pricing in `MODEL_PRICING`. Adding live calls to those providers would require their respective SDKs and API keys — the app is designed so you can compare costs without signing up for every service.

**Prices are June 2025 snapshots:** LLM pricing changes frequently. The `MODEL_PRICING` dict is the single source of truth — update it when providers change their rates.
