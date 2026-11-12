import createDebugMessages from 'debug';
import { v4 as uuidv4 } from 'uuid';
import { Chunk, Message, ConversationEntry, Sources, QueryStreamEvent } from '../global/types.js';
import { BaseConversations } from './base-conversations.js';

export abstract class BaseModel {
    private static defaultTemperature: number;
    private static conversations: BaseConversations; // Static property for managing conversations

    private readonly baseDebug = createDebugMessages('embedjs:model:BaseModel');
    private readonly _temperature?: number;

    constructor(temperature?: number) {
        this._temperature = temperature;
    }

    public static setDefaultTemperature(temperature?: number) {
        BaseModel.defaultTemperature = temperature;
    }

    public static setConversations(conversations: BaseConversations) {
        BaseModel.conversations = conversations; // Correct setting of the static property
    }

    public get temperature() {
        return this._temperature ?? BaseModel.defaultTemperature;
    }

    public async init(): Promise<void> {}

    public async query(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        conversationId: string = 'default',
        userId?: string,
        instanceId?: string
    ): Promise<any> {
        const conversation = await BaseModel.conversations.getConversation(conversationId); // Use static property

        this.baseDebug(`${conversation.entries.length} history entries found for conversationId '${conversationId}'`);

        const uniqueSources = this.extractUniqueSources(supportingContext);

        const chunksIds = this.extractChunkIds(supportingContext);

        // Extract only the content from each entry in the conversation
        const pastConversations = conversation.entries.map(entry => entry.content);

        const result = await this.runQuery(system, userQuery, supportingContext, pastConversations, userId, instanceId);

        return this.saveExchange(conversationId, userQuery, result, uniqueSources, chunksIds);
    }

    /**
     * Streaming version of `query`. Yields `{ type: 'token', text }` for each piece of the answer as the
     * model generates it. Once the answer is complete, the user query and the full answer are saved to the
     * conversation (exactly as `query` does) and a final `{ type: 'done', entry }` event is yielded.
     *
     * If the consumer stops iterating early (e.g. the client disconnected), nothing is saved.
     */
    public async *queryStream(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        conversationId: string = 'default',
        userId?: string,
        instanceId?: string
    ): AsyncGenerator<QueryStreamEvent, void, undefined> {
        const conversation = await BaseModel.conversations.getConversation(conversationId);
        this.baseDebug(`${conversation.entries.length} history entries found for conversationId '${conversationId}' (streaming)`);

        const uniqueSources = this.extractUniqueSources(supportingContext);
        const chunksIds = this.extractChunkIds(supportingContext);
        const pastConversations = conversation.entries.map(entry => entry.content);

        let result = '';
        for await (const text of this.runQueryStream(system, userQuery, supportingContext, pastConversations, userId, instanceId)) {
            if (!text) continue;
            result += text;
            yield { type: 'token', text };
        }

        const entry = await this.saveExchange(conversationId, userQuery, result, uniqueSources, chunksIds);
        yield { type: 'done', entry };
    }

    /**
     * Adds the user query and the AI answer to the conversation history and returns the AI entry.
     */
    private async saveExchange(
        conversationId: string,
        userQuery: string,
        answer: string,
        sources: Sources[],
        chunkIds: string[],
    ): Promise<ConversationEntry> {
        // Add user query to history
        await BaseModel.conversations.addEntryToConversation(conversationId, {
            _id: uuidv4(),
            timestamp: new Date(),
            content: {
                sender: 'HUMAN',
                message: userQuery
            },
            sources: [],
            chunks: []
        });

        const newEntry: ConversationEntry = {
            _id: uuidv4(),
            timestamp: new Date(),
            content: {
                sender: "AI",
                message: answer
            },
            sources,
            chunks: chunkIds
        }
        // Add AI response to history
        await BaseModel.conversations.addEntryToConversation(conversationId, newEntry);

        return newEntry;
    }

    /*
     * silentConversationQuery
     * Do a query based upon the conversation history, but do not add the query or the response to the conversation.
     * This can be used to ask the LLM about the conversation, e.g. if the topic has changed or newer chunks are required etc.
     */
    public async silentConversationQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        conversationId: string = 'default',
        userId?: string,
        instanceId?: string
    ): Promise<any> {
        const conversation = await BaseModel.conversations.getConversation(conversationId); // Use static property

        this.baseDebug(`${conversation.entries.length} history entries found for conversationId '${conversationId}'`);

        // Extract only the content from each entry in the conversation
        const pastConversations = conversation.entries.map(entry => entry.content);

        const result = await this.runQuery(system, userQuery, supportingContext, pastConversations, userId, instanceId);

        return result;
    }

    /**
     * One-off query with no conversation history and nothing saved (e.g. compressing retrieved
     * passages). The context is passed in the prompt by the caller, not as supporting chunks.
     */
    public async standaloneQuery(system: string, userQuery: string): Promise<string> {
        const result = await this.runQuery(system, userQuery, [], []);
        return typeof result === 'string' ? result : String(result ?? '');
    }

    private extractUniqueSources(supportingContext: Chunk[]): Sources[] {
        const uniqueSources = new Map<string, Sources>();  // Use a Map to track unique sources by URL

        supportingContext.forEach(item => {
            const { metadata } = item;
            if (metadata && metadata.source) {
                // Use the source URL as the key to ensure uniqueness
                if (!uniqueSources.has(metadata.source)) {
                    uniqueSources.set(metadata.source, {
                        source: metadata.source,
                        loaderId: metadata.uniqueLoaderId // Assuming this field always exists
                    });
                }
            }
        });

        // Convert the values of the Map to an array
        return Array.from(uniqueSources.values());
    }

    private extractChunkIds(supportingContext: Chunk[]): string[] {
        // Create a Set to track unique chunk IDs
        const chunkIds = new Set<string>();

        supportingContext.forEach(item => {
            const { metadata } = item;
            if (metadata && metadata.id) {
                // Add the chunk ID to the Set (automatically handles uniqueness)
                chunkIds.add(metadata.id);
            }
        });

        // Convert the Set to an array
        return Array.from(chunkIds);
    }

    protected abstract runQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
        userId?: string,
        instanceId?: string,
    ): Promise<any>;

    /**
     * Yields the answer text piece by piece. Models that support streaming override this; the default
     * runs the normal (non-streaming) query and yields the whole answer as a single piece, so
     * `queryStream` works with every model.
     */
    protected async *runQueryStream(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
        userId?: string,
        instanceId?: string,
    ): AsyncGenerator<string, void, undefined> {
        const result = await this.runQuery(system, userQuery, supportingContext, pastConversations, userId, instanceId);
        yield typeof result === 'string' ? result : String(result ?? '');
    }
}