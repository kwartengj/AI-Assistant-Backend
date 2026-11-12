# Multi-database retrieval strategies

An application can search several vector databases at once (`SetOfDbs`), for example its own
sources plus other assistants' sources. `ragApplication.retrieve(query, options)` searches them all,
lets a **strategy** decide which chunks the model sees, and returns a **trace** of what happened.

```js
const { chunks, trace } = await ragApplication.retrieve('How is open data licensed?', {
    strategy: 'weighted_relevance', // see below
    k: 6,                           // chunks to return (per database for top_k_per_db)
});
// pass `chunks` as the context to query() / queryStream()
```

The strategy runs per call, so concurrent requests can use different strategies. `similaritySearch`
and `getContext` keep their original behaviour.

## Pipeline

1. **Embed** the query once (unit length).
2. **Search every database in parallel** for `candidatesPerDb` candidates (default `max(20, 2k)`).
   Strategies that compare chunks with each other or with topics also get each chunk's **stored
   vector**, so nothing is embedded again. A database that fails is reported in the trace instead of
   failing the request.
3. **Filter**: drop candidates at or below `minScore` (default: the application's relevance cut-off)
   and passages repeated across databases.
4. **Select** with the strategy. The output is ordered best first; callers may trim the end to fit a
   token budget, so strategies that balance databases interleave them to keep the balance at any length.
5. **Trace**: per database (candidates, above cut-off, top score, mean of top 3, whole-database
   similarity, weight, quota, selected, time) and per selected chunk (database, rank in its database,
   score, strategy score), plus timings and model calls.

Writes (`insertChunks`, `deleteKeys`, `reset`) and `getVectorCount` only touch the **first (primary)**
database; connected databases are read-only.

## Scores

MongoDB Atlas reports cosine matches as `s = (1 + cos) / 2` (0–1). Similarities computed here
(whole-database similarity, MMR redundancy) are put on the same scale before they are combined with `s`.

**Database profile.** For database *d* with chunk vectors *v*, `μ_d = mean(v / |v|)`. Then
`q · μ_d` is exactly the average cosine between the query and every chunk in *d*. Profiles are
computed by streaming the vectors once and cached until the database's chunk count changes.

## Strategies

| id | Selection | Model calls |
|---|---|---|
| `top_k` | All candidates ranked by `s`, top *k*. Baseline; exposes score bias between databases. | 0 |
| `top_k_n` | Equal quota per database (`k/n`, largest remainder); unused slots go to the others; round-robin order. | 0 |
| `top_k_per_db` | Top *k* from every database (*n·k*), round-robin order. "Maximum context" condition. | 0 |
| `weighted_relevance` | Evidence `e_d = m · mean(top-3 s) + (1 − m) · (1 + q·μ_d)/2`, weights `w = softmax(e / T)`, quotas `k·w` (largest remainder, capped by available candidates), smooth weighted round-robin order. | 0 |
| `topic_classification` | Topic labels are embedded once. Distributions over labels `p(x) = softmax(cos(x, labels) / T)` for the query, each candidate and each database (`μ_d`). Re-rank by `0.6·s + 0.25·cos(p(q), p(chunk)) + 0.15·cos(p(q), p(d))`, top *k*. Without topics: `weighted_relevance`. | 0 |
| `llm_summarization` | Selects with `summarizationBase` (default `weighted_relevance`), then compresses chunks from the same document (2+ chunks) into one passage focused on the query, in parallel, with a timeout. Compressed passages keep their source (citations still work) and list `compressedFrom`. | 1 per multi-chunk document |
| `rrf` | Reciprocal rank fusion of the global ranking by `s` and each chunk's rank inside its own database: `1/(60 + r_global) + 1/(60 + r_db)`. Rank-based, so it doesn't assume scores are comparable across databases. | 0 |
| `mmr` | Maximal marginal relevance: greedily pick `argmax λ·s − (1 − λ)·max sim(chunk, picked)`. Reduces near-duplicates. | 0 |

Older names still work: `default`/`top_k_chunks` → `top_k`, `topKNChunks`/`top_k_n_chunks` → `top_k_n`,
`topKChunksPerSource`/`top_k_chunks_per_source` → `top_k_per_db`, `weightedRelevance`,
`topicClassification`, `llmSummarization`.

## Options

| option | default | |
|---|---|---|
| `strategy` | `SetOfDbs.setStrategy()` or `top_k` | |
| `k` | `setNumberK()` or the search result count | 1–50 |
| `candidatesPerDb` | `max(20, 2k)` | ≥ k, ≤ 200 |
| `minScore` | `setRelevanceThreshold()` or the relevance cut-off | |
| `temperature` (*T*) | 0.1 | lower = sharper weights / topic distributions |
| `evidenceMix` (*m*) | 0.6 | `weighted_relevance`: best chunks vs whole database |
| `mmrLambda` (*λ*) | 0.7 | 1 = relevance only |
| `topics` | `[]` | labels for `topic_classification` |
| `summarizationBase` | `weighted_relevance` | |
| `llmTimeoutMs` | 60000 | per compression call |
| `verboseTrace` | false | include every candidate in the trace |

## Changes from the earlier SetOfDbs strategies

- Strategies run per request instead of being set on a shared `SetOfDbs`.
- `weightedRelevance` used a Euclidean *distance* as a similarity (less similar databases got more
  weight) and embedded each database's full text, which embedding models truncate (~512 tokens).
  It now uses search evidence plus the exact whole-database profile.
- `topicClassification` and `llmSummarization` called the model for every chunk in every database on
  each query. Topic matching now uses embeddings only; summarisation compresses only the selected chunks.
- `reset()`/`deleteKeys()` no longer touch connected databases.
