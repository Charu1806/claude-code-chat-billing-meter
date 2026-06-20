# AI Token Cost Explorer

A Streamlit app that makes LLM token costs **visible, measurable, and comparable** across 20 models from 7 providers — with live multi-provider API calls, MCP tool simulation, and pure-math cost calculators that require no API keys.

---

## Quick Start

```bash
git clone https://github.com/Charu1806/claude-code-chat-billing-meter.git
cd claude-code-chat-billing-meter
cp .env.example .env      # then open .env and add your keys
./launch.sh               # installs deps and starts the app
```

App opens at **http://localhost:8501**

---

## API Keys

| Provider | Models | Where to get |
|----------|--------|-------------|
| `ANTHROPIC_API_KEY` | Claude Opus 4.8, Sonnet 4.6, Haiku 4.5 | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| `GOOGLE_API_KEY` | Gemini 2.5 Flash, Gemini 2.5 Pro | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| `GROQ_API_KEY` | Mistral Saba, Llama 3.3 70B, Llama 3.1 8B | [console.groq.com/keys](https://console.groq.com/keys) |

You only need keys for the providers you want to use. **Tabs 3, 4, 5 work with zero keys.**

---

## What This App Does

Most LLM cost surprises come from not tracking tokens in real time. This app fixes that:

- **Sidebar model selector** — switch between 8 live models across Anthropic, Google, Gemini, Groq. Selected model drives Tabs 1 & 2.
- **See every token category** — reasoning, input, output, tool use — broken out with per-category costs
- **Live context window bar** — progress bar with warnings before you hit the limit
- **Compare 20 models side-by-side** — Anthropic, Google, Mistral, xAI, OpenAI, Meta, Cohere
- **RAG break-even calculator** — exact month when RAG pays off vs. full-context LLM

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sidebar: Model Selector                      │
│   Anthropic (Claude) · Google (Gemini) · Groq (Mistral/Llama)   │
│               ↓ drives Tabs 1 & 2                               │
├──────────────────┬──────────────────┬──────────────────────────┤
│   Tab 1          │   Tab 2          │  Tabs 3, 4, 5            │
│   Code Writer    │   Chatbot        │  Enterprise / RAG /      │
│   (MCP + think)  │   + Context bar  │  Pricing Reference       │
└────────┬─────────┴────────┬─────────┴──────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
  Provider Router     Provider Router      Pure Math
  (based on model)    (based on model)     MODEL_PRICING dict
         │                  │
    ┌────┴──────────────────┴───────────┐
    │  Anthropic SDK  │  OpenAI-compat  │
    │  (Claude)       │  client         │
    └─────────────────┴─────────────────┘
              │               │
    ┌─────────┘    ┌──────────┴──────────┐
    │              │                     │
    ▼              ▼                     ▼
Anthropic       Google                 Groq
API             Generative             API
(exact token    Language API           (Mistral,
 count_tokens)  (OpenAI-compat)         Llama)
```

**Key design:** All non-Anthropic providers use the `openai` Python package pointed at each provider's OpenAI-compatible endpoint — one SDK, different `base_url`. No separate `google-generativeai` or `mistralai` packages needed.

---

## Tab-by-Tab Breakdown

### Tab 1 — Code Writer with MCP Tool Simulation

**What it does:** Describe a coding task; the active model writes code using simulated MCP tool calls. Every token category is tracked and costs are cumulative across iterations.

**MCP integration:**

Tools are defined as JSON schemas in MCP format (`name`, `description`, `input_schema`) and passed to the API. Claude reasons about them and generates real tool call requests — those definitions and call/response JSON consume real tokens that are measured.

```
User types task
    │
    ├─ Anthropic model → count_tokens() pre-flight → stream() with thinking + tools
    └─ Other models    → stream() with OpenAI-format tools
         │
         ├─ thinking blocks  → reasoning tokens (Claude only)
         ├─ text blocks      → output tokens (streamed live)
         └─ tool_use blocks  → tool overhead tokens
```

**MCP tools defined (simulated):**

| Tool | Description | Why simulated? |
|------|-------------|---------------|
| `read_file` | Read source file contents | No persistent process in Streamlit |
| `search_docs` | Search documentation | No persistent process in Streamlit |
| `list_directory` | List project files | No persistent process in Streamlit |

In a real app (CLI agent, desktop app, backend service), you'd replace the mock responses with a live MCP server over stdio or SSE.

**Token counting:**
- **Claude models:** `client.messages.count_tokens()` — exact pre-flight count via Anthropic API
- **Other models:** Token usage returned in the stream response

### Tab 2 — Chatbot + Live Context Window Tracker

Multi-turn chat with the active model. Sidebar shows:
- Context window progress bar (yellow at 70%, red at 90%)
- Running token total + cost
- Per-turn breakdown

Context window % is exact for Claude (via `count_tokens()`) and estimated for other providers (word count × 2).

### Tab 3 — Multi-Provider Enterprise Cost Calculator *(no keys)*

Configure QPS, token counts, and peak multiplier. Compare all 20 models:
- Cost per query / hour / day / month
- Bar chart sorted cheapest → most expensive
- Price vs context window scatter
- Optimization tips (prompt caching, batch API, model routing)

### Tab 4 — RAG vs Plain LLM *(no keys)*

Calculates whether RAG is cheaper than sending full context every time:
- Plain LLM: `full_context_tokens × price` per query
- RAG: `embedding_cost + retrieved_chunks × price + infra/month`
- Break-even formula: `infra_monthly / (plain_monthly - rag_llm_monthly)`
- Cross-model chart: which models benefit most from RAG

### Tab 5 — Pricing Reference *(no keys)*

Filterable table + scatter + bar chart of all 20 models.

---

## Live Model Selector

| Model Key | Label | Provider | SDK Route |
|-----------|-------|----------|-----------|
| `claude-opus-4-8` | Claude Opus 4.8 | Anthropic | `anthropic` SDK |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 | Anthropic | `anthropic` SDK |
| `claude-haiku-4-5` | Claude Haiku 4.5 | Anthropic | `anthropic` SDK |
| `gemini-2.5-flash` | Gemini 2.5 Flash | Google | `openai` → `generativelanguage.googleapis.com` |
| `gemini-2.5-pro` | Gemini 2.5 Pro | Google | `openai` → `generativelanguage.googleapis.com` |
| `mistral-saba` | Mistral Saba | Groq / Mistral | `openai` → `api.groq.com` |
| `llama-3.3-70b` | Llama 3.3 70B | Groq / Meta | `openai` → `api.groq.com` |
| `llama-3.1-8b` | Llama 3.1 8B | Groq / Meta | `openai` → `api.groq.com` |

---

## Full Pricing Reference (June 2025)

| Provider | Model | Input $/1M | Output $/1M | Context |
|----------|-------|-----------|------------|---------|
| Anthropic | Claude Opus 4.8 | $5.00 | $25.00 | 1M |
| Anthropic | Claude Sonnet 4.6 | $3.00 | $15.00 | 1M |
| Anthropic | Claude Haiku 4.5 | $1.00 | $5.00 | 200K |
| Google | Gemini 2.5 Flash | $0.15 | $0.60 | 1M |
| Google | Gemini 2.5 Pro | $1.25 | $10.00 | 1M |
| Google | Gemini 1.5 Flash | $0.075 | $0.30 | 1M |
| Mistral | Mistral Large 2 | $2.00 | $6.00 | 128K |
| Mistral | Mistral Small 3.1 | $0.10 | $0.30 | 128K |
| Mistral | Mistral Nemo | $0.15 | $0.15 | 128K |
| Mistral | Codestral | $0.30 | $0.90 | 256K |
| xAI | Grok 3 | $3.00 | $15.00 | 131K |
| xAI | Grok 3 Mini | $0.30 | $0.50 | 131K |
| xAI | Grok 2 | $2.00 | $10.00 | 131K |
| OpenAI | GPT-4o | $2.50 | $10.00 | 128K |
| OpenAI | GPT-4o mini | $0.15 | $0.60 | 128K |
| OpenAI | o3 | $2.00 | $8.00 | 200K |
| OpenAI | o4-mini | $1.10 | $4.40 | 200K |
| Meta (via API) | Llama 3.3 70B | $0.23 | $0.40 | 128K |
| Meta (via API) | Llama 3.1 405B | $0.80 | $0.80 | 128K |
| Cohere | Command R+ | $2.50 | $10.00 | 128K |
| Cohere | Command R | $0.15 | $0.60 | 128K |

---

## File Structure

```
claude-code-chat-billing-meter/
├── app.py              # Entire app — all 5 tabs, provider router, pricing config
├── .env.example        # Template — copy to .env and add your keys
├── requirements.txt    # anthropic, openai, streamlit, pandas, plotly, python-dotenv
├── launch.sh           # One-command setup: checks keys, installs deps, starts app
└── README.md           # This file
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Anthropic models | `anthropic` SDK — streaming + `count_tokens()` |
| Google + Groq models | `openai` SDK — pointed at each provider's OpenAI-compat endpoint |
| Charts | Plotly |
| Tables | Pandas |
| Config | `python-dotenv` |
