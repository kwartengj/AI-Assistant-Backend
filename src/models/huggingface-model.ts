import createDebugMessages from 'debug';
import { InferenceClient } from '@huggingface/inference';

import { BaseModel } from '../interfaces/base-model.js';
import { Chunk, Message } from '../global/types.js';

export class HuggingFace extends BaseModel {
    private readonly debug = createDebugMessages('embedjs:model:HuggingFace');

    private readonly modelName: string;
    private readonly maxNewTokens: number;
    private readonly endpointUrl?: string;
    private readonly apiKey?: string;
    private client: InferenceClient;

    constructor(params?: { modelName?: string; temperature?: number; maxNewTokens?: number; endpointUrl?: string, apiKey?: string }) {
        super(params?.temperature);

        this.endpointUrl = params?.endpointUrl;
        this.apiKey = params?.apiKey;
        this.maxNewTokens = params?.maxNewTokens ?? 300;
        this.modelName = params?.modelName ?? 'mistralai/Mixtral-8x7B-Instruct-v0.1';
    }

    override async init(): Promise<void> {
        // Called directly, so the library doesn't need @langchain/community just for this model
        this.client = new InferenceClient(this.apiKey);
    }

    override async runQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
    ): Promise<string> {
        const pastMessages = [system];
        pastMessages.push(`Data: ${supportingContext.map((s) => s.pageContent).join('; ')}`);

        pastMessages.push.apply(
            pastMessages,
            pastConversations.map((c) => {
                if (c.sender === 'AI') return `AI: ${c.message}`;
                else if (c.sender === 'SYSTEM') return `SYSTEM: ${c.message}`;
                else return `HUMAN: ${c.message}`;
            }),
        );

        pastMessages.push(`Question: ${userQuery}?`);
        pastMessages.push('Answer: ');

        const finalPrompt = pastMessages.join('\n');
        // this.debug('Final prompt being sent to HF - ', finalPrompt);
        this.debug(`Executing hugging face '${this.modelName}' model with prompt -`, userQuery);
        const parameters = { max_new_tokens: this.maxNewTokens, temperature: this.temperature, return_full_text: false };
        const result = this.endpointUrl
            ? await this.client.endpoint(this.endpointUrl).textGeneration({ inputs: finalPrompt, parameters })
            : await this.client.textGeneration({ model: this.modelName, inputs: finalPrompt, parameters });
        this.debug('Hugging response -', result);
        return result.generated_text;
    }
}
