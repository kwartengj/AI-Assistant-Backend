import createDebugMessages from 'debug';
import ModelClient from "@azure-rest/ai-inference";
import { isUnexpected } from "@azure-rest/ai-inference";
import { BaseModel } from '../interfaces/base-model.js';
import { Chunk, Message } from '../global/types.js';
import { AzureKeyCredential } from '@azure/core-auth';
import { traceable } from "langsmith/traceable";

type ChatMessage = { role: string; content: string };

export class AzureAIInferenceModel extends BaseModel {
    private readonly debug = createDebugMessages('embedjs:model:AzureAIInference');
    private readonly modelName: string;
    private readonly maxNewTokens: number;
    private readonly endpointUrl?: string;
    private readonly apiKey?: string;
    private model: any;

    constructor(params?: { modelName?: string; temperature?: number; maxNewTokens?: number; endpointUrl?: string; apiKey?: string }) {
        super(params?.temperature);
        this.modelName = params?.modelName ?? 'Meta-Llama-3-70B-Instruct';
        this.endpointUrl = params?.endpointUrl;
        this.apiKey = params?.apiKey;
        this.maxNewTokens = params?.maxNewTokens ?? 300;
    }

    override async init(): Promise<void> {
        if (!this.endpointUrl) {
            throw new Error('Azure AI endpoint URL is missing.');
        }

        if (!this.apiKey) {
            throw new Error('Azure AI API key is missing.');
        }

        this.model = ModelClient(
            this.endpointUrl,
            new AzureKeyCredential(this.apiKey),
            {
                allowInsecureConnection: true // to allow non https conncetions
            }
        );
    }

    private buildMessages(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
    ): ChatMessage[] {
        const messages: ChatMessage[] = [
            {
                role: "system",
                content: system
            }
        ];
        messages.push({
            role: "system",
            content: `Data: ${supportingContext.map((s) => s.pageContent).join('; ')}`
        });

        messages.push.apply(
            messages,
            pastConversations.map((c) => {
                if (c.sender === 'AI') return {
                    role: "assistant",
                    content: c.message
                };
                else if (c.sender === 'SYSTEM') return {
                    role: "system",
                    content: c.message
                };
                else return {
                    role: "user",
                    content: c.message
                };
            }),
        );

        messages.push({
            role: "user",
            content: userQuery
        });

        return messages;
    }

    override async runQuery(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
        userId?: string,
        instanceId?: string,
    ): Promise<string> {
        const finalPrompt = this.buildMessages(system, userQuery, supportingContext, pastConversations);
        this.debug(`Executing Azure AI Inference '${this.modelName}' model with prompt -`, userQuery);

        const chatModel = traceable(
            async ({ messages }: { messages: ChatMessage[] }) => {
                const response = await this.model.path("chat/completions").post({
                    body: {
                        model: this.modelName,
                        messages: messages,
                        max_tokens: this.maxNewTokens,
                        temperature: this.temperature
                    }
                });
                if (isUnexpected(response)) {
                    throw response.body.error;
                }
                return response.body;
            },
            {
                run_type: "llm",
                name: "ChatAzureAI",
                metadata: { ls_provider: "azure-ai-inference", ls_model_name: "azure_model", ls_model_type: "chat", ls_temperature: this.temperature, ls_instance: instanceId, ls_user: userId },
            }
        );

        const result = (await chatModel({ messages: finalPrompt })).choices[0].message.content
        this.debug('Azure response -', result);
        return result;
    }

    override async *runQueryStream(
        system: string,
        userQuery: string,
        supportingContext: Chunk[],
        pastConversations: Message[],
        userId?: string,
        instanceId?: string,
    ): AsyncGenerator<string, void, undefined> {
        const finalPrompt = this.buildMessages(system, userQuery, supportingContext, pastConversations);
        this.debug(`Streaming Azure AI Inference '${this.modelName}' model with prompt -`, userQuery);

        const client = this.model;
        const modelName = this.modelName;
        const maxTokens = this.maxNewTokens;
        const temperature = this.temperature;

        const streamChat = traceable(
            async function* ({ messages }: { messages: ChatMessage[] }): AsyncGenerator<string, void, undefined> {
                const response = await client.path("chat/completions").post({
                    body: {
                        model: modelName,
                        messages: messages,
                        max_tokens: maxTokens,
                        temperature: temperature,
                        stream: true
                    }
                }).asNodeStream();

                const body: NodeJS.ReadableStream | undefined = response.body;
                if (!body) {
                    throw new Error('Azure AI Inference returned an empty response.');
                }

                if (String(response.status) !== '200') {
                    const text = await readAll(body);
                    let message = text;
                    try {
                        message = JSON.parse(text)?.error?.message ?? text;
                    } catch {
                        // not JSON; use the raw text
                    }
                    throw new Error(`Model request failed (${response.status}): ${String(message).slice(0, 500)}`);
                }

                for await (const data of readServerSentEvents(body)) {
                    if (data === '[DONE]') return;
                    let event: any;
                    try {
                        event = JSON.parse(data);
                    } catch {
                        continue; // ignore keep-alives and partial lines
                    }
                    if (event?.error) {
                        throw new Error(event.error.message ?? 'Model stream failed.');
                    }
                    const text = event?.choices?.[0]?.delta?.content;
                    if (text) yield text;
                }
            },
            {
                run_type: "llm",
                name: "ChatAzureAIStream",
                aggregator: (pieces: string[]) => pieces.join(''),
                metadata: { ls_provider: "azure-ai-inference", ls_model_name: "azure_model", ls_model_type: "chat", ls_temperature: temperature, ls_instance: instanceId, ls_user: userId },
            }
        );

        yield* streamChat({ messages: finalPrompt });
    }
}

/** Reads a whole Node stream into a string (used for error bodies). */
async function readAll(stream: NodeJS.ReadableStream): Promise<string> {
    const decoder = new TextDecoder();
    let text = '';
    for await (const chunk of stream as AsyncIterable<Uint8Array | string>) {
        text += typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true });
    }
    return text + decoder.decode();
}

/** Yields the `data:` payload of each server-sent event in an OpenAI-style streaming response. */
async function* readServerSentEvents(stream: NodeJS.ReadableStream): AsyncGenerator<string, void, undefined> {
    const decoder = new TextDecoder();
    let buffer = '';
    for await (const chunk of stream as AsyncIterable<Uint8Array | string>) {
        buffer += typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true });
        let newline: number;
        while ((newline = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, newline).replace(/\r$/, '');
            buffer = buffer.slice(newline + 1);
            if (line.startsWith('data:')) yield line.slice(5).trim();
        }
    }
    buffer += decoder.decode();
    if (buffer.startsWith('data:')) yield buffer.slice(5).trim();
}
