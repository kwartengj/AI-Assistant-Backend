/**
 * author: joseph.kwarteng@open.ac.uk
 * created on: 27-02-2025-18h-35m
 * github: https://github.com/kwartengj
 * copyright 2025
 *
 * Several vector databases searched as one. Searches run on every database in parallel and the
 * results are merged (each chunk is tagged with its database's name in metadata.dbName).
 *
 * The first database is the primary one: writes, deletions, resets and counts only touch it, so
 * the other databases are read-only for this application.
 */
import createDebugMessages from 'debug';
import { BaseDb } from '../interfaces/base-db.js';
import { ExtractChunkData, InsertChunkData } from '../global/types.js';

type NamedDb = { database: BaseDb; name: string };

export class SetOfDbs implements BaseDb {
    private readonly debug = createDebugMessages('embedjs:vector:SetOfDbs');
    private readonly dbs: NamedDb[];

    constructor(dbs: NamedDb[]) {
        if (!dbs || dbs.length === 0) {
            throw new Error('At least one database must be provided.');
        }
        this.dbs = dbs;
    }

    private get primary(): BaseDb {
        return this.dbs[0].database;
    }

    async init({ dimensions }: { dimensions: number }): Promise<void> {
        this.debug('Connecting to set of databases...');
        await Promise.all(this.dbs.map((db) => db.database.init({ dimensions })));
        this.debug('Connected to set of databases');
    }

    async insertChunks(chunks: InsertChunkData[]): Promise<number> {
        return this.primary.insertChunks(chunks);
    }

    /** Top k from every database, merged. The caller filters, sorts and trims. */
    async similaritySearch(query: number[], k: number): Promise<ExtractChunkData[]> {
        const results = await Promise.all(
            this.dbs.map(async (db) => {
                try {
                    const rows = await db.database.similaritySearch(query, k);
                    return rows.map((chunk) => ({ ...chunk, metadata: { ...chunk.metadata, dbName: db.name } }));
                } catch (error) {
                    // One unavailable database shouldn't fail the whole search
                    this.debug(`Search failed for database ${db.name}:`, error);
                    return [];
                }
            }),
        );
        return results.flat();
    }

    async getVectorCount(): Promise<number> {
        return this.primary.getVectorCount();
    }

    async deleteKeys(uniqueLoaderId: string): Promise<boolean> {
        return this.primary.deleteKeys(uniqueLoaderId);
    }

    async reset(): Promise<void> {
        await this.primary.reset();
    }
}
