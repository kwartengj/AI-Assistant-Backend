// base-conversations.ts
import { Conversation, ConversationEntry } from '../global/types.js';

export interface BaseConversations {
    init(): Promise<void>;
    addConversation(conversationId: string): Promise<void>;
    getConversation(conversationId: string): Promise<Conversation>;
    hasConversation(conversationId: string): Promise<boolean>;
    deleteConversation(conversationId: string): Promise<void>;
    addEntryToConversation(conversationId: string, entry: ConversationEntry): Promise<void>;
    /** Optional: append several entries in one write. Callers fall back to addEntryToConversation. */
    addEntriesToConversation?(conversationId: string, entries: ConversationEntry[]): Promise<void>;
    clearConversations(): Promise<void>;
}