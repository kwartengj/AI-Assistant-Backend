/**
 * author: joseph.kwarteng@open.ac.uk
 * created on: 27-02-2025-18h-35m
 * github: https://github.com/kwartengj
 * copyright 2025
 *
 * Several vector databases searched as one. The first database is the primary one: it receives
 * every write (new chunks, deletions, resets). The others are read-only for this application.
 *
 * Selection strategies live in ../retrieval and run per request through RAGApplication.retrieve(),
 * so concurrent requests can use different strategies. similaritySearch() keeps the original
 * behaviour (top results from every database, merged) for code that doesn't use retrieve().
 */
import createDebugMessages from 'debug';

import { BaseDb } from '../interfaces/base-db.js';
import { ExtractChunkData, InsertChunkData } from '../global/types.js';
import { Candidate, DbCandidates, DbProfile } from '../retrieval/types.js';

type NamedDb = { database: BaseDb; name: string };

export class SetOfDbs implements BaseDb {
    private readonly debug = createDebugMessages('embedjs:vector:SetOfDbs');
    private static readonly PRIMARY = 0;

    private readonly dbs: NamedDb[];
    /** Profiles are cached until a database's chunk count changes */
    private readonly profileCache = new Map<string, DbProfile>();
    private readonly profileJobs = new Map<string, Promise<DbProfile>>();

    /** Defaults for RAGApplication.retrieve() when a request doesn't name them (kept for older callers) */
    private defaultStrategy: string | undefined;
    private relevanceThreshold: number | undefined;
    private numberK: number | undefined;

    constructor(dbs: NamedDb[]) {
        if (!dbs || dbs.length === 0) {
            throw new Error('At least one database must be provided.');
        }
        this.dbs = dbs;
    }

    get names(): string[] {
        return this.dbs.map((db) => db.name);
    }

    async init({ dimensions }: { dimensions: number }) {
        this.debug('Connecting to set of databases...');
        await Promise.all(this.dbs.map((db) => db.database.init({ dimensions })));
        this.debug('Connected to set of databases');
    }

    /* ---------- writes: primary database only ---------- */

    async insertChunks(chunks: InsertChunkData[]): Promise<number> {
        return this.dbs[SetOfDbs.PRIMARY].database.insertChunks(chunks);
    }

    async deleteKeys(uniqueLoaderId: string): Promise<boolean> {
        return this.dbs[SetOfDbs.PRIMARY].database.deleteKeys(uniqueLoaderId);
    }

    async reset(): Promise<void> {
        await this.dbs[SetOfDbs.PRIMARY].database.reset();
    }

    /** Chunks in the primary database (the application's own sources) */
    async getVectorCount(): Promise<number> {
        return this.dbs[SetOfDbs.PRIMARY].database.getVectorCount();
    }

    /* ---------- reads ---------- */

    /** Original behaviour: the top k from every database, merged (the caller sorts and trims). */
    async similaritySearch(query: number[], k: number): Promise<ExtractChunkData[]> {
        const perDb = await this.searchAllDbs(query, this.numberK || k);
        return perDb.flatMap((d) =>
            d.candidates.map(({ vector: _v, dbName, dbRank: _r, ...chunk }) => ({
                ...chunk,
                metadata: { ...chunk.metadata, dbName },
            })),
        );
    }

    /**
     * Searches every database in parallel. A database that fails is reported in `error` instead of
     * failing the whole search.
     */
    async searchAllDbs(query: number[], perDb: number, { withVectors = false } = {}): Promise<DbCandidates[]> {
        return Promise.all(
            this.dbs.map(async (db): Promise<DbCandidates> => {
                const started = Date.now();
                try {
                    const rows: (ExtractChunkData & { vector?: number[] })[] = db.database.searchCandidates
                        ? await db.database.searchCandidates(query, perDb, { withVectors })
                        : await db.database.similaritySearch(query, perDb);
                    const candidates: Candidate[] = rows
                        .sort((a, b) => b.score - a.score)
                        .map((row, i) => ({
                            ...row,
                            metadata: { ...row.metadata, dbName: db.name },
                            dbName: db.name,
                            dbRank: i + 1,
                        }));
                    return { dbName: db.name, candidates, ms: Date.now() - started };
                } catch (error) {
                    this.debug(`Search failed for database ${db.name}:`, error);
                    return { dbName: db.name, candidates: [], ms: Date.now() - started, error: (error as Error).message };
                }
            }),
        );
    }

    /** Whole-database profiles (see DbProfile), computed once and refreshed when a database changes. */
    async getProfiles(): Promise<Map<string, DbProfile>> {
        const out = new Map<string, DbProfile>();
        await Promise.all(
            this.dbs.map(async (db) => {
                if (!db.database.getProfile) return;
                try {
                    const count = await db.database.getVectorCount();
                    const cached = this.profileCache.get(db.name);
                    if (cached && cached.count === count) {
                        out.set(db.name, cached);
                        return;
                    }
                    let job = this.profileJobs.get(db.name);
                    if (!job) {
                        job = db.database
                            .getProfile()
                            .then((p) => ({ dbName: db.name, ...p }))
                            .finally(() => this.profileJobs.delete(db.name));
                        this.profileJobs.set(db.name, job);
                    }
                    const profile = await job;
                    this.profileCache.set(db.name, profile);
                    out.set(db.name, profile);
                } catch (error) {
                    this.debug(`Profile failed for database ${db.name}:`, error);
                }
            }),
        );
        return out;
    }

    async getFullText(): Promise<string> {
        const allTexts = await Promise.all(this.dbs.map((db) => db.database.getFullText()));
        return allTexts.join(' ');
    }

    async getChunks(): Promise<ExtractChunkData[]> {
        const allChunks = await Promise.all(this.dbs.map((db) => db.database.getChunks()));
        return allChunks.flat();
    }

    /* ---------- defaults (older API) ---------- */

    setStrategy(strategy: string): void {
        this.defaultStrategy = strategy;
    }

    getStrategy(): string | undefined {
        return this.defaultStrategy;
    }

    setRelevanceThreshold(relevanceThreshold: number): void {
        this.relevanceThreshold = relevanceThreshold;
    }

    getRelevanceThreshold(): number | undefined {
        return this.relevanceThreshold;
    }

    setNumberK(numberK: number): void {
        this.numberK = numberK;
    }

    getNumberK(): number | undefined {
        return this.numberK;
    }

    /** No longer needed (strategies don't call the model); kept so existing callers still work. */
    ragApp(_ragApp: unknown): void {}
}
