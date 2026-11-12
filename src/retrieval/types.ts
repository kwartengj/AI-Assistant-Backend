import { ExtractChunkData } from '../global/types.js';

/**
 * Multi-database retrieval: types shared by the candidate search, the strategies and the trace.
 *
 * Terms used throughout:
 *   - database (db): one vector store, e.g. one assistant's MongoDB database. A SetOfDbs holds several.
 *   - candidate:     a chunk returned by a database's vector search, before any strategy runs.
 *   - selection:     the ordered chunks a strategy returns (best first). The caller may still trim
 *                    the end to fit a token budget, so order matters.
 */

export const STRATEGIES = [
    'top_k',
    'top_k_n',
    'top_k_per_db',
    'weighted_relevance',
    'topic_classification',
    'llm_summarization',
    'rrf',
    'mmr',
] as const;
export type StrategyId = (typeof STRATEGIES)[number];

/** Old names (frontend and SetOfDbs) mapped to the current ids. */
export const STRATEGY_ALIASES: Record<string, StrategyId> = {
    default: 'top_k',
    top_k_chunks: 'top_k',
    topKNChunks: 'top_k_n',
    top_k_n_chunks: 'top_k_n',
    topKChunksPerSource: 'top_k_per_db',
    top_k_chunks_per_source: 'top_k_per_db',
    weightedRelevance: 'weighted_relevance',
    topicClassification: 'topic_classification',
    llmSummarization: 'llm_summarization',
};

export function resolveStrategy(name?: string): StrategyId {
    if (!name) return 'top_k';
    if ((STRATEGIES as readonly string[]).includes(name)) return name as StrategyId;
    return STRATEGY_ALIASES[name] ?? 'top_k';
}

export type Candidate = ExtractChunkData & {
    dbName: string;
    /** 1-based rank within its own database's results */
    dbRank: number;
    /** Stored embedding (unit length), present when the database can return it */
    vector?: number[];
};

export type DbCandidates = {
    dbName: string;
    candidates: Candidate[];
    ms: number;
    error?: string;
};

/**
 * A database summarised by the mean of its unit-length chunk vectors.
 * dot(unit(q), meanUnit) is exactly the average cosine similarity between q and every chunk in it.
 */
export type DbProfile = {
    dbName: string;
    count: number;
    meanUnit: number[] | null;
};

export type RetrievalOptions = {
    strategy?: string;
    /** Chunks to return (per database for top_k_per_db). Default 7 */
    k?: number;
    /** Candidates fetched from each database before a strategy runs. Default max(20, 2k) */
    candidatesPerDb?: number;
    /** Candidates scoring at or below this are dropped. Default: the application's relevance cut-off */
    minScore?: number;
    /** MMR trade-off between relevance (1) and diversity (0). Default 0.7 */
    mmrLambda?: number;
    /** Softmax temperature for database weights and topic distributions (lower = sharper). Default 0.1 */
    temperature?: number;
    /** weighted_relevance: share of the database weight from its best candidates vs. its whole-database profile. Default 0.6 */
    evidenceMix?: number;
    /** topic_classification: topic labels to classify the query, chunks and databases against */
    topics?: string[];
    /** llm_summarization: strategy that picks the chunks before they are compressed. Default weighted_relevance */
    summarizationBase?: string;
    /** llm_summarization: per-call timeout in ms. Default 60000 */
    llmTimeoutMs?: number;
    /** Include candidate lists in the trace (larger). Default false */
    verboseTrace?: boolean;
};

export type ResolvedOptions = Required<Omit<RetrievalOptions, 'topics' | 'strategy' | 'summarizationBase'>> & {
    strategy: StrategyId;
    topics: string[];
    summarizationBase: StrategyId;
};

export type DbTrace = {
    dbName: string;
    candidates: number;
    aboveCutoff: number;
    topScore: number | null;
    meanTop3: number | null;
    /** Average cosine (raw, -1..1) between the query and every chunk in the database (needs a profile) */
    profileSimilarity?: number | null;
    weight?: number;
    quota?: number;
    selected: number;
    ms: number;
    error?: string;
};

export type SelectedTrace = {
    id: string;
    dbName: string;
    source: string;
    score: number;
    dbRank: number;
    rank: number;
    /** Strategy-specific score used for ordering, when it differs from the vector score */
    strategyScore?: number;
    compressedFrom?: string[];
};

export type RetrievalTrace = {
    strategy: StrategyId;
    requestedStrategy?: string;
    k: number;
    candidatesPerDb: number;
    minScore: number;
    dbs: DbTrace[];
    selected: SelectedTrace[];
    timings: { embedMs: number; searchMs: number; profileMs: number; strategyMs: number; totalMs: number };
    llmCalls: number;
    notes: string[];
    candidates?: { id: string; dbName: string; score: number; dbRank: number }[];
};

export type StrategyContext = {
    opts: ResolvedOptions;
    queryVector: number[];
    profiles: Map<string, DbProfile>;
    topicVectors: { label: string; vector: number[] }[];
    /** Per-database facts strategies may add to (weights, quotas) */
    dbInfo: Map<string, Partial<DbTrace>>;
    notes: string[];
};

export type ScoredCandidate = Candidate & { strategyScore?: number; compressedFrom?: string[] };

export type RetrievalResult = {
    chunks: ScoredCandidate[];
    trace: RetrievalTrace;
};
