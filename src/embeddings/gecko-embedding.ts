import { VertexAIEmbeddings } from '@langchain/google-vertexai';
import { BaseEmbeddings } from '../interfaces/base-embeddings.js';

export class GeckoEmbedding implements BaseEmbeddings {
    private model: VertexAIEmbeddings;

    constructor() {
        this.model = new VertexAIEmbeddings({ model: 'textembedding-gecko', maxConcurrency: 3, maxRetries: 5 });
    }

    getDimensions(): number {
        return 768;
    }

    embedDocuments(texts: string[]): Promise<number[][]> {
        return this.model.embedDocuments(texts);
    }

    embedQuery(text: string): Promise<number[]> {
        return this.model.embedQuery(text);
    }
}
