import createDebugMessages from 'debug';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, AIMessage, SystemMessage } from '@langchain/core/messages';

import { BaseModel } from '../interfaces/base-model.js';
import { Chunk, Message } from '../global/types.js';

export class OpenAi extends BaseModel {
    private readonly debug = createDebugMessages('embedjs:model:OpenAi');
    private readonly modelName: string;
    private readonly baseURL: string;
    private readonly apiKey: string;
    private model: ChatOpenAI;

    constructor({ temperature, modelName, baseURL, apiKey }: { temperature?: number; modelName: string, baseURL?: string, apiKey? : string }) {
        super(temperature);
        this.modelName = modelName;
        if (baseURL !== undefined) {
            this.baseURL = baseURL;
        }
        this.apiKey = apiKey ?? undefined
    }

    override async init(): Promise<void> {
        this.model = new ChatOpenAI({ temperature: this.temperature, model: this.modelName, apiKey : this.apiKey }, this.baseURL ? { baseURL: this.baseURL } : {});
    }

    private buildMessages(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
    ): (AIMessage | SystemMessage | HumanMessage)[] {
        const pastMessages: (AIMessage | SystemMessage | HumanMessage)[] = [new SystemMessage(system)];
        pastMessages.push(
            new SystemMessage(`Supporting context: ${supportingContext.map((s) => s.pageContent).join('; ')}`),
        );

        pastMessages.push.apply(
            pastMessages,
            pastConversations.map((c) => {
                if (c.sender === 'AI') return new AIMessage({ content: c.message });
                else if (c.sender === 'SYSTEM') return new SystemMessage({ content: c.message });
                else return new HumanMessage({ content: c.message });
            }),
        );
        pastMessages.push(new HumanMessage(`${userQuery}?`));
        return pastMessages;
    }

    override async runQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
    ): Promise<string> {
        const pastMessages = this.buildMessages(system, userQuery, supportingContext, pastConversations);

        this.debug('Executing openai model with prompt -', userQuery);
        const result = await this.model.invoke(pastMessages);
        this.debug('OpenAI response -', result);
        return result.content.toString();
    }

    override async *runQueryStream(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
    ): AsyncGenerator<string, void, undefined> {
        const pastMessages = this.buildMessages(system, userQuery, supportingContext, pastConversations);

        this.debug('Streaming openai model with prompt -', userQuery);
        const stream = await this.model.stream(pastMessages);
        for await (const chunk of stream) {
            const text = typeof chunk.content === 'string' ? chunk.content : chunk.content.map((c: any) => c?.text ?? '').join('');
            if (text) yield text;
        }
    }
}