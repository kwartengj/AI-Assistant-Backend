import { MongoClient } from 'mongodb';
import createDebugMessages from 'debug';

/**
 * One MongoClient (one connection pool) per connection string, shared by every MongoDb vector store,
 * MongoCache and MongoConversations in the process. Without this, each of those opens its own pool,
 * so an app serving many assistants holds three pools per assistant.
 *
 * Clients are reference counted: acquire() in init(), release() in close(). The pool is closed when
 * the last user releases it.
 */
const debug = createDebugMessages('embedjs:mongo-clients');
const pools = new Map<string, { client: Promise<MongoClient>; refs: number }>();

export async function acquireMongoClient(uri: string): Promise<MongoClient> {
    let pool = pools.get(uri);
    if (!pool) {
        const client = new MongoClient(uri).connect();
        pool = { client, refs: 0 };
        pools.set(uri, pool);
        // A failed connection shouldn't be cached; the next acquire tries again
        client.catch(() => {
            if (pools.get(uri) === pool) pools.delete(uri);
        });
        debug('Opened a shared MongoDB client');
    }
    pool.refs += 1;
    try {
        return await pool.client;
    } catch (error) {
        pool.refs -= 1;
        throw error;
    }
}

export async function releaseMongoClient(uri: string): Promise<void> {
    const pool = pools.get(uri);
    if (!pool) return;
    pool.refs -= 1;
    if (pool.refs > 0) return;
    pools.delete(uri);
    debug('Closing the shared MongoDB client (no users left)');
    await (await pool.client).close();
}
