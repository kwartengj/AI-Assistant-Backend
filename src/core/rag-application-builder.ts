import { BaseDb } from '../interfaces/base-db.js';
import { RAGApplication } from './rag-application.js';
import { BaseCache } from '../interfaces/base-cache.js';
import { BaseEmbeddings } from '../interfaces/base-embeddings.js';
import { BaseModel } from '../interfaces/base-model.js';
import { SIMPLE_MODELS } from '../global/constants.js';
import { OpenAi } from '../models/openai-model.js';
import { LoaderParam } from './dynamic-loader-selector.js';
import { BaseConversations } from '../interfaces/base-conversations.js';

export class RAGApplicationBuilder {
    private temperature: number;
    private queryTemplate: string;
    private cache?: BaseCache;
    private model: BaseModel | null;
    private searchResultCount: number;
    private embeddingModel: BaseEmbeddings;
    private embeddingRelevanceCutOff: number;
    private loaders: LoaderParam[];
    private vectorDb: BaseDb;
    private conversations: BaseConversations;

    constructor() {
        this.loaders = [];
        this.temperature = 0.1;
        this.searchResultCount = 7;

        this.queryTemplate = `You are a friendly, knowledgeable assistant. Answer the user's latest question using the source material provided with it and the conversation so far.

Base your answers on the source material. If it doesn't cover the question, say so plainly, then suggest what the user could ask instead or where they might look. Never invent facts, figures, quotes or references. If the material answers only part of the question, answer that part and say what's missing.

Lead with the answer and match its length to the question: a few sentences for a simple question, more detail with steps or bullet points when that helps. If a question is unclear, ask a short clarifying question.

Refer to the material naturally, for example "the materials say…". Don't mention "context", "data", "provided documents" or "training data", and don't describe how you were given information. Reply in the language the user writes in.`;

        this.setModel(SIMPLE_MODELS.OPENAI_GPT4_O);
        this.embeddingRelevanceCutOff = 0;
    }

    async build() {
        const entity = new RAGApplication(this);
        await entity.init();
        return entity;
    }

    addLoader(loader: LoaderParam) {
        this.loaders.push(loader);
        return this;
    }

    setSearchResultCount(searchResultCount: number) {
        this.searchResultCount = searchResultCount;
        return this;
    }

    setVectorDb(vectorDb: BaseDb) {
        this.vectorDb = vectorDb;
        return this;
    }

    setTemperature(temperature: number) {
        this.temperature = temperature;
        if (this.model) this.setModel(this.model);
        return this;
    }

    setEmbeddingRelevanceCutOff(embeddingRelevanceCutOff: number) {
        this.embeddingRelevanceCutOff = embeddingRelevanceCutOff;
        return this;
    }

    setQueryTemplate(queryTemplate: string) {
        // if (!queryTemplate.includes('{0}'))
        //     throw new Error('queryTemplate must include a placeholder for the query using {0}');

        this.queryTemplate = queryTemplate;
        return this;
    }

    setCache(cache: BaseCache) {
        this.cache = cache;
        return this;
    }

    setEmbeddingModel(embeddingModel: BaseEmbeddings) {
        this.embeddingModel = embeddingModel;
        return this;
    }

    setModel(model: 'NO_MODEL' | SIMPLE_MODELS | BaseModel) {
        if (typeof model === 'object') this.model = model;
        else {
            if (model === SIMPLE_MODELS.OPENAI_GPT4_O) this.model = new OpenAi({ modelName: 'gpt-4o' });
            else if (model === SIMPLE_MODELS['OPENAI_GPT4_TURBO'])
                this.model = new OpenAi({ modelName: 'gpt-4-turbo' });
            else if (model === SIMPLE_MODELS['OPENAI_GPT3.5_TURBO'])
                this.model = new OpenAi({ modelName: 'gpt-3.5-turbo' });
            else this.model = null;
        }

        return this;
    }

    getLoaders() {
        return this.loaders;
    }

    getSearchResultCount() {
        return this.searchResultCount;
    }

    getVectorDb() {
        return this.vectorDb;
    }

    getTemperature() {
        return this.temperature;
    }

    getEmbeddingRelevanceCutOff() {
        return this.embeddingRelevanceCutOff;
    }

    getQueryTemplate() {
        return this.queryTemplate;
    }

    getCache() {
        return this.cache;
    }

    getEmbeddingModel() {
        return this.embeddingModel;
    }

    getModel() {
        return this.model;
    }

    setConversations(conversations: BaseConversations) {
        this.conversations = conversations;
        return this;
    }

    getConversations() {
        return this.conversations;
    }
}
