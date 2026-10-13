# research-radar

**An LLM agent that keeps you on top of the literature.** It watches arXiv and PubMed for the topics you care about, uses an LLM to score each new paper against a plain-English brief, summarizes the ones that matter, and writes a Markdown digest. Everything it screens is kept, so you can later ask questions across the whole history and get cited answers.

Built with **LangGraph** (agent orchestration), **LangChain** (provider-agnostic chat models), and **LlamaIndex** (BM25 retrieval for Q&A). It works with Anthropic Claude, OpenAI, Azure OpenAI, AWS Bedrock, or a local model through Ollama.

## Sample output

A real run on the `ai-oil-gas-ops` topic, fully local with **llama3.2 (3B) through Ollama on an RTX 4080**: 23 new arXiv papers screened, scored, and summarized in **25 seconds**, with no API key. Full digest: [`examples/2026-10-12_ai-oil-gas-ops.md`](examples/2026-10-12_ai-oil-gas-ops.md).

> ## 5. [Towards Better Shale Gas Production Forecasting Using Transfer Learning](https://arxiv.org/abs/2106.11051)
> Omar S. Alolayan, Samuel J. Raymond, Justin B. Montgomery et al. · arXiv cs.LG · 2021-06-21 · **relevance 8/10** · `AI` `production forecasting` `transfer learning`
>
> **TL;DR:** Using transfer learning, deep neural networks can improve shale gas production forecasts by up to 47% compared to traditional methods.
>
> - The proposed method reduces forecasting error between 11-47% compared to the Arps decline curve model

The scores separate signal from noise well: decline-curve and production-forecasting papers scored 6–8, and adjacent "forecasting" papers on wind power, GDP, crypto and cloud cover scored 0.

```
$ research-radar ask "Which papers compare machine learning against Arps decline curves, and how much better was it?"
Only paper [1] compares machine learning against Arps decline curves. According to the paper, the results
reduce the forecasting error between 11% and 47% compared to the widely used Arps decline curve model.
[1] Towards Better Shale Gas Production Forecasting Using Transfer Learning — https://arxiv.org/abs/2106.11051
```

**Lessons from running it on a small local model.** These are all handled in the code now:
- 3B models tend to echo a JSON *schema* back instead of filling it in, so the prompt shows a filled-in example, and Ollama's constrained JSON mode is turned on for scoring and summaries.
- One paper the model can't score is logged and retried on the next run instead of failing the whole digest.
- Small models occasionally mis-number citations when retrieval returns off-topic papers. `ask` therefore searches only papers scored ≥ 4 by default; for production use, a larger model (Claude, GPT) is recommended.

## Why

Reading every new paper in a fast-moving field doesn't scale, and keyword alerts are noisy. A brief like *"physics + ML hybrids validated on real industrial data; toy PDE benchmarks are low priority"* captures what you actually want, and an LLM can apply it consistently across hundreds of abstracts. research-radar turns that into a repeatable, auditable pipeline: every score comes with a rationale, and summaries are constrained to facts stated in the abstract.

## How it works

```mermaid
flowchart LR
    A[fetch<br/>arXiv · PubMed] --> B[dedupe<br/>SQLite store]
    B --> C[assess<br/>LLM scores 0-10<br/>+ rationale + tags]
    C -->|any ≥ threshold| D[summarize<br/>TL;DR · findings · why it matters]
    C -->|none| E
    D --> E[compose<br/>Markdown digest]
    E --> F[(store)]
    F -.-> G[ask<br/>LlamaIndex BM25 + LLM<br/>cited answers]
```

| Node | What it does | Design notes |
|---|---|---|
| `fetch` | Queries each configured source | A failing source is logged and skipped, not fatal. Respects arXiv (3 s) and NCBI (3 req/s) rate limits |
| `dedupe` | Drops papers already seen for this topic | SQLite, so repeated runs (e.g. a weekly cron) only surface new work |
| `assess` | LLM scores relevance against the topic brief | Parallel calls; JSON validated by Pydantic, with one self-correcting retry on malformed output |
| `summarize` | Structured summary for papers above the threshold | Prompted to use only facts in the abstract |
| `compose` | Renders the digest and persists all scores | Low-scoring papers are stored too, so `ask` can search everything |
| `ask` | Retrieval-augmented Q&A over stored papers | BM25 (no embedding model or vector DB needed); answers cite `[1]`, `[2]` |

**Provider-agnostic structured output.** Rather than depending on one vendor's function-calling API, the agent asks for JSON, extracts it robustly (code fences, surrounding chatter), validates it against a Pydantic schema, and feeds validation errors back to the model once. That lets the same pipeline run on frontier APIs and on small local models.

## Quick start

```bash
git clone https://github.com/faazeli/research-radar && cd research-radar
uv sync                                   # or: pip install -e .
cp config.example.yaml config.yaml        # edit topics, briefs, provider

export ANTHROPIC_API_KEY=...              # or OPENAI_API_KEY, or run Ollama locally
uv run research-radar sources             # dry run: check your queries return papers (no LLM calls)
uv run research-radar run                 # screen, summarize, write digests/
uv run research-radar ask "Which papers validate PINNs on field data?"
```

Run fully locally with Ollama:

```bash
ollama pull llama3.2
uv run research-radar run --provider ollama --model llama3.2
```

## Configuration

```yaml
llm:
  provider: anthropic          # anthropic | openai | ollama | azure_openai | bedrock_converse
topics:
  - name: ai-oil-gas-ops
    title: AI for Oil & Gas Production Operations
    brief: >
      AI and automation for upstream production: artificial lift, production forecasting,
      predictive maintenance, anomaly detection on well sensor data. Field-validated results matter most.
    min_relevance: 6           # papers below this are stored but left out of the digest
    max_in_digest: 10
    sources:
      - name: arxiv
        query: '(abs:"artificial lift" OR abs:"production forecasting") AND abs:learning'
```

See [`config.example.yaml`](config.example.yaml) for three worked topics (arXiv and PubMed).

## Tests

```bash
uv run pytest
```

The tests run the **entire LangGraph pipeline offline**: real arXiv and PubMed API responses are stored as fixtures, and a scripted fake chat model stands in for the LLM. They cover parsing, the relevance threshold, de-duplication across runs, recovery from malformed model output, a failing source, and BM25 retrieval for `ask`.

## Extending

- **New source:** add a `fetch_<name>(query, max_results) -> list[Paper]` to `sources.py` and register it in `FETCHERS`.
- **New step:** add a node in `graph.py`, e.g. a critic that double-checks summaries against the abstract.
- **Delivery:** pipe the digest to Slack or email from a scheduled job.

## License

MIT
