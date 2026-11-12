/**
 * Selection strategies for retrieving from several vector databases.
 *
 * Each strategy is a pure function: it receives every database's candidates (already above the
 * score cut-off and sorted best first) and returns the chosen chunks in priority order. Callers may
 * trim the end of the list to fit a token budget, so strategies that balance databases interleave
 * them (see weightedInterleave) to keep that balance at any length.
 *
 * None of these make model calls. llm_summarization (in rag-application.ts) runs a base strategy
 * from here and then compresses the result with the model.
 */
import { Candidate, DbCandidates, ScoredCandidate, StrategyContext, StrategyId } from './types.js';
import { allocate, dot, meanTop, softmax, unit } from './vector-math.js';

type StrategyFn = (pool: DbCandidates[], ctx: StrategyContext) => ScoredCandidate[];

const RRF_K = 60;
const TOPIC_MIX = { vector: 0.6, chunkTopic: 0.25, dbTopic: 0.15 };

/* ---------- helpers ---------- */

const byScore = (a: Candidate, b: Candidate) => b.score - a.score;
const flat = (pool: DbCandidates[]) => pool.flatMap((d) => d.candidates);

function setDbInfo(ctx: StrategyContext, dbName: string, info: Record<string, unknown>) {
    ctx.dbInfo.set(dbName, { ...(ctx.dbInfo.get(dbName) || {}), ...info });
}

/**
 * Merges per-database lists so that any prefix of the result keeps each database close to its quota
 * share (smooth weighted round robin). Equal quotas give a plain round robin.
 */
export function weightedInterleave(lists: Candidate[][], quotas: number[]): Candidate[] {
    const total = quotas.reduce((s, q) => s + q, 0);
    const taken = quotas.map(() => 0);
    const out: Candidate[] = [];
    for (let step = 1; step <= total; step++) {
        let best = -1;
        let bestDeficit = -Infinity;
        quotas.forEach((q, i) => {
            if (taken[i] >= q) return;
            const deficit = (q * step) / total - taken[i];
            // Ties go to the database whose next chunk scores higher
            const tie = best >= 0 && Math.abs(deficit - bestDeficit) < 1e-9 && lists[i][taken[i]].score > lists[best][taken[best]].score;
            if (deficit > bestDeficit + 1e-9 || tie) {
                best = i;
                bestDeficit = deficit;
            }
        });
        if (best < 0) break;
        out.push(lists[best][taken[best]]);
        taken[best] += 1;
    }
    return out;
}

/**
 * MongoDB Atlas reports cosine matches as (1 + cos) / 2, a 0-1 score. Similarities this module computes
 * itself are put on the same scale before they are combined with search scores.
 */
const toScore = (cos: number) => (1 + cos) / 2;

/** Average similarity between the query and every chunk in the database, on the search-score scale. */
function profileSimilarity(ctx: StrategyContext, dbName: string): number | null {
    const p = ctx.profiles.get(dbName);
    return p && p.meanUnit ? toScore(dot(unit(ctx.queryVector), p.meanUnit)) : null;
}

/* ---------- strategies ---------- */

/** Baseline: every candidate ranked by its raw vector score, top k overall. */
const topK: StrategyFn = (pool, { opts }) => flat(pool).sort(byScore).slice(0, opts.k);

/** Equal share per database (ceil(k/n)); slots a database can't fill go to the others. */
const topKN: StrategyFn = (pool, ctx) => {
    const lists = pool.map((d) => d.candidates);
    const quotas = allocate(ctx.opts.k, lists.map(() => 1), lists.map((l) => l.length));
    pool.forEach((d, i) => setDbInfo(ctx, d.dbName, { quota: quotas[i] }));
    return weightedInterleave(lists, quotas);
};

/** Top k from every database (n*k in total), interleaved. The "maximum context" condition. */
const topKPerDb: StrategyFn = (pool, ctx) => {
    const lists = pool.map((d) => d.candidates);
    const quotas = lists.map((l) => Math.min(ctx.opts.k, l.length));
    pool.forEach((d, i) => setDbInfo(ctx, d.dbName, { quota: quotas[i] }));
    return weightedInterleave(lists, quotas);
};

/**
 * Each database is weighted by evidence for this query:
 *   evidence = mix * mean(top 3 candidate scores) + (1 - mix) * average cosine to the whole database
 * weights = softmax(evidence / temperature); k is split by weight (largest remainder).
 */
const weightedRelevance: StrategyFn = (pool, ctx) => {
    const { opts } = ctx;
    const lists = pool.map((d) => d.candidates);
    const evidence = pool.map((d) => {
        const top = meanTop(d.candidates.map((c) => c.score)) ?? 0;
        const prof = profileSimilarity(ctx, d.dbName);
        return prof === null ? top : opts.evidenceMix * top + (1 - opts.evidenceMix) * prof;
    });
    // Databases with no candidates above the cut-off get no weight
    const usable = lists.map((_, i) => i).filter((i) => lists[i].length > 0);
    const usableWeights = softmax(usable.map((i) => evidence[i]), opts.temperature);
    const weights = lists.map(() => 0);
    usable.forEach((i, j) => { weights[i] = usableWeights[j]; });
    const quotas = allocate(opts.k, weights, lists.map((l) => l.length));
    pool.forEach((d, i) => setDbInfo(ctx, d.dbName, { weight: round(weights[i]), quota: quotas[i], evidence: round(evidence[i]) }));
    return weightedInterleave(lists, quotas).map((c) => ({ ...c, strategyScore: round(evidence[pool.findIndex((d) => d.dbName === c.dbName)]) }));
};

/**
 * Zero-shot topic classification with embeddings (no model calls). The query, each candidate and
 * each database get a distribution over the configured topic labels; chunks are re-ranked by
 *   0.6 * vector score + 0.25 * topic match(query, chunk) + 0.15 * topic match(query, database)
 * where topic match is the cosine between the two distributions.
 */
const topicClassification: StrategyFn = (pool, ctx) => {
    const { opts, topicVectors } = ctx;
    if (!topicVectors.length) {
        ctx.notes.push('topic_classification: no topics configured, used weighted_relevance instead');
        return weightedRelevance(pool, ctx);
    }
    const labels = topicVectors.map((t) => t.vector);
    const dist = (v: number[]) => softmax(labels.map((l) => dot(v, l)), opts.temperature);
    const distCos = (a: number[], b: number[]) => {
        const m = Math.sqrt(dot(a, a) * dot(b, b));
        return m > 0 ? dot(a, b) / m : 0;
    };

    const qDist = dist(unit(ctx.queryVector));
    const topQuery = qDist
        .map((p, i) => ({ label: topicVectors[i].label, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3)
        .map((t) => `${t.label} ${(t.p * 100).toFixed(0)}%`);
    ctx.notes.push(`topic_classification: query topics ${topQuery.join(', ')}`);

    const dbTopic = new Map<string, number>();
    pool.forEach((d) => {
        const p = ctx.profiles.get(d.dbName);
        const match = p && p.meanUnit ? distCos(qDist, dist(p.meanUnit)) : 0;
        dbTopic.set(d.dbName, match);
        setDbInfo(ctx, d.dbName, { weight: round(match) });
    });

    let missing = 0;
    const scored = flat(pool).map((c) => {
        let chunkTopic = 0;
        if (c.vector) chunkTopic = distCos(qDist, dist(c.vector));
        else missing++;
        const s = TOPIC_MIX.vector * c.score + TOPIC_MIX.chunkTopic * chunkTopic + TOPIC_MIX.dbTopic * (dbTopic.get(c.dbName) ?? 0);
        return { ...c, strategyScore: round(s) };
    });
    if (missing) ctx.notes.push(`topic_classification: ${missing} candidates had no stored vector`);
    return scored.sort((a, b) => b.strategyScore - a.strategyScore).slice(0, opts.k);
};

/**
 * Reciprocal rank fusion of two rankings: the global ranking by score, and each chunk's rank inside
 * its own database. Rank-based, so it does not assume scores from different databases are comparable.
 */
const rrf: StrategyFn = (pool, { opts }) => {
    const global = flat(pool).sort(byScore);
    return global
        .map((c, i) => ({ ...c, strategyScore: round(1 / (RRF_K + i + 1) + 1 / (RRF_K + c.dbRank), 6) }))
        .sort((a, b) => b.strategyScore - a.strategyScore)
        .slice(0, opts.k);
};

/**
 * Maximal marginal relevance: picks chunks that are relevant but not near-duplicates of chunks
 * already picked. score = lambda * relevance - (1 - lambda) * max similarity to the picked set.
 */
const mmr: StrategyFn = (pool, ctx) => {
    const { opts } = ctx;
    const all = flat(pool).sort(byScore);
    if (all.some((c) => !c.vector)) {
        ctx.notes.push('mmr: stored vectors unavailable, used top_k instead');
        return all.slice(0, opts.k);
    }
    const picked: ScoredCandidate[] = [];
    const rest = [...all];
    while (picked.length < opts.k && rest.length) {
        let bestIdx = 0;
        let bestVal = -Infinity;
        rest.forEach((c, i) => {
            const redundancy = picked.length ? Math.max(...picked.map((p) => toScore(dot(c.vector!, p.vector!)))) : 0;
            const val = opts.mmrLambda * c.score - (1 - opts.mmrLambda) * redundancy;
            if (val > bestVal) {
                bestVal = val;
                bestIdx = i;
            }
        });
        picked.push({ ...rest[bestIdx], strategyScore: round(bestVal) });
        rest.splice(bestIdx, 1);
    }
    return picked;
};

export const STRATEGY_FUNCTIONS: Record<Exclude<StrategyId, 'llm_summarization'>, StrategyFn> = {
    top_k: topK,
    top_k_n: topKN,
    top_k_per_db: topKPerDb,
    weighted_relevance: weightedRelevance,
    topic_classification: topicClassification,
    rrf,
    mmr,
};

export function runStrategy(id: StrategyId, pool: DbCandidates[], ctx: StrategyContext): ScoredCandidate[] {
    const fn = STRATEGY_FUNCTIONS[id as Exclude<StrategyId, 'llm_summarization'>] ?? topK;
    return fn(pool, ctx);
}

function round(x: number, digits = 4) {
    const f = 10 ** digits;
    return Math.round(x * f) / f;
}
