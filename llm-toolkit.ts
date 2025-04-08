//MCV Core library for LLM interactions

export type ModelProvider = 'openai' | 'anthropic' | 'google' | 'custom';

export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp?: number;
  id?: string;
}

export interface Conversation {
  id: string;
  messages: Message[];
  metadata?: Record<string, any>;
  tokenCount?: number;
}

export interface CompletionOptions {
  model: string;
  provider: ModelProvider;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
  apiKey?: string;
  endpoint?: string;
  extraParams?: Record<string, any>;
}

export interface LLMToolkitConfig {
  defaultProvider?: ModelProvider;
  defaultModel?: string;
  apiKeys?: Record<ModelProvider, string>;
  endpoints?: Record<ModelProvider, string>;
  tokenLimit?: number;
  defaultSystemPrompt?: string;
  useLocalStorage?: boolean; //** Optional use localStorage */
  storageKeyPrefix?: string;
}

/* New optional localstorage session management */

class StorageManager {
  private enabled: boolean;
  private prefix: string;

  constructor(enabled = false, prefix = 'llmtoolkit:') {
    this.enabled = enabled;
    this.prefix = prefix;
  }

  private safeKey(key: string): string {
    return `${this.prefix}${key}`;
  }

  get(key: string): string | null {
    if (!this.enabled) return null;
    try {
      return localStorage.getItem(this.safeKey(key));
    } catch(e) {
      console.warn(e, "Unable to fetch items from localStorage")
      return null;
    }
  }

  set(key: string, value: string): void {
    if (!this.enabled) return;
    try {
      localStorage.setItem(this.safeKey(key), value);
    } catch(e) {
      console.warn(e, "Unable to save item to localStorage")
    }
  }

  remove(key: string): void {
    if (!this.enabled) return;
    try {
      localStorage.removeItem(this.safeKey(key));
    } catch {}
  }
}

/* End of LocalStorage session management class */

const AVERAGE_CHARS_PER_TOKEN = 4;

export const estimateTokenCount = (text: string): number => {
  return Math.ceil(text.length / AVERAGE_CHARS_PER_TOKEN);
};

export class LLMToolkit {
  private conversations: Map<string, Conversation> = new Map();
  private config: LLMToolkitConfig;
  private tokenWarningThreshold: number;
  private storage: StorageManager;

  
  constructor(config: LLMToolkitConfig = {}) {
    this.config = {
      defaultProvider: 'openai',
      defaultModel: 'gpt-3.5-turbo',
      defaultSystemPrompt: 'You are a helpful assistant.',
      tokenLimit: 18000,
      ...config
    };

    this.storage = new StorageManager(config.useLocalStorage ?? false, config.storageKeyPrefix);
    
    this.tokenWarningThreshold = Math.floor((this.config.tokenLimit || 18000) * 0.80)
  }

  saveSession(sessionId: string, data: string) {
    this.storage.set(`session-${sessionId}`, data);
  }

  loadSession(sessionId: string): string | null {
    return this.storage.get(`session-${sessionId}`);
  }

  private generateId(prefix: string): string {
    try {
        if (crypto?.randomUUID) {
            return `${prefix}-${crypto.randomUUID()}`;
        }
    } catch (e) {
        console.warn("crypto.randomUUID not available or failed.");
    }
    console.warn("Using insecure fallback ID generation. Consider a UUID library.");
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 9);
    return `${prefix}-${timestamp}-${random}`;
  }

  private _addMessage(conversationId: string, role: Message['role'], content: string, metadata?: Record<string, any>): void {
    const conversation = this.getConversation(conversationId);

    const message: Message = {
      role,
      content,
      timestamp: Date.now(),
      id: this.generateId(`msg-${role}`)
    };

    conversation.messages.push(message);

    if (metadata) {
        conversation.metadata = { ...conversation.metadata, ...metadata };
    }

    this.updateConversationTokenCount(conversation);
    this.checkTokenWarning(conversation);
  }

  
  public createConversation(systemPrompt?: string): string {
    const id = this.generateId('conv');

    const initialMessage: Message = {
      role: 'system',
      content: systemPrompt || this.config.defaultSystemPrompt || '',
      timestamp: Date.now(),
      id: `msg-system-${Date.now()}`
    };
    
    const tokenCount = estimateTokenCount(initialMessage.content);
    
    const conversation: Conversation = {
      id,
      messages: [initialMessage],
      tokenCount,
      metadata: {}
    };
    
    this.conversations.set(id, conversation);
    return id;
  }
  
  public addContext(conversationId: string, content: string, metadata?: Record<string, any>): void {
    this._addMessage(conversationId, 'system', content, metadata);
  }
  
  public addUserMessage(conversationId: string, content: string): void {
    this._addMessage(conversationId, 'user', content);
  }
  
  public addAssistantMessage(conversationId: string, content: string): void {
    this._addMessage(conversationId, 'assistant', content);    
  }
  
  public getConversation(conversationId: string): Conversation {
    const conversation = this.conversations.get(conversationId);
    if (!conversation) {
      throw new Error(`Conversation with ID ${conversationId} not found`);
    }
    return conversation;
  }
  
  public getAllConversations(): Conversation[] {
    return Array.from(this.conversations.values());
  }
  
  public deleteConversation(conversationId: string): boolean {
    return this.conversations.delete(conversationId);
  }
  
  public async sendCompletion(
    conversationId: string, 
    userMessage: string, 
    options?: Partial<CompletionOptions>
  ): Promise<string> {
    this.addUserMessage(conversationId, userMessage);
    
    const conversation = this.getConversation(conversationId);
    const completionOptions: CompletionOptions = {
      model: options?.model || this.config.defaultModel || 'gpt-3.5-turbo',
      provider: options?.provider || this.config.defaultProvider || 'openai',
      temperature: options?.temperature || 0.7,
      maxTokens: options?.maxTokens || 1000,
      stream: options?.stream || false,
      apiKey: options?.apiKey || this.getApiKey(options?.provider || this.config.defaultProvider || 'openai'),
      endpoint: options?.endpoint || this.getEndpoint(options?.provider || this.config.defaultProvider || 'openai'),
      extraParams: options?.extraParams || {}
    };
    
    try {
      const response = await this.makeApiRequest(conversation, completionOptions);
      this.addAssistantMessage(conversationId, response);
      return response;
    } catch (error) {
      console.error('Error during LLM completion:', error);
      throw error;
    }
  }
  
  public async streamCompletion(
    conversationId: string,
    userMessage: string,
    onChunk: (chunk: string) => void,
    options?: Partial<CompletionOptions>
  ): Promise<string> {
    this.addUserMessage(conversationId, userMessage);
    
    const conversation = this.getConversation(conversationId);
    const completionOptions: CompletionOptions = {
      model: options?.model || this.config.defaultModel || 'gpt-3.5-turbo',
      provider: options?.provider || this.config.defaultProvider || 'openai',
      temperature: options?.temperature || 0.7,
      maxTokens: options?.maxTokens || 1000,
      stream: true,
      apiKey: options?.apiKey || this.getApiKey(options?.provider || this.config.defaultProvider || 'openai'),
      endpoint: options?.endpoint || this.getEndpoint(options?.provider || this.config.defaultProvider || 'openai'),
      extraParams: options?.extraParams || {}
    };
    
    try {
      const fullResponse = await this.makeStreamingApiRequest(conversation, completionOptions, onChunk);
      this.addAssistantMessage(conversationId, fullResponse);
      return fullResponse;
    } catch (error) {
      console.error('Error during LLM streaming completion:', error);
      throw error;
    }
  }
  
  public async batchCompletions(
    requests: Array<{conversationId: string, userMessage: string, options?: Partial<CompletionOptions>}>
  ): Promise<string[]> {
    const promises = requests.map(request => 
      this.sendCompletion(request.conversationId, request.userMessage, request.options)
    );
    return Promise.all(promises);
  }
  
  private updateConversationTokenCount(conversation: Conversation): void {
    const allText = conversation.messages.map(msg => msg.content).join(' ');
    conversation.tokenCount = estimateTokenCount(allText);
  }
  
  private checkTokenWarning(conversation: Conversation): void {
    if ((conversation.tokenCount || 0) >= this.tokenWarningThreshold) {
      console.warn(`Token count warning: Conversation ${conversation.id} has reached ${conversation.tokenCount} tokens, approaching the limit of ${this.config.tokenLimit}.`);
    }
  }
  
  private getApiKey(provider: ModelProvider): string {
    const apiKey = this.config.apiKeys?.[provider];
    if (!apiKey) {
      throw new Error(`No API key configured for provider: ${provider}`);
    }
    return apiKey;
  }
  
  private getEndpoint(provider: ModelProvider): string {
    const defaultEndpoints: Record<ModelProvider, string> = {
      'openai': 'https://api.openai.com/v1/chat/completions',
      'anthropic': 'https://api.anthropic.com/v1/messages',
      'google': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
      'custom': ''
    };
    
    return this.config.endpoints?.[provider] || defaultEndpoints[provider];
  }
  
  private formatMessagesForProvider(messages: Message[], provider: ModelProvider): any {
    switch (provider) {
      case 'openai':
        return messages.map(msg => ({
          role: msg.role,
          content: msg.content
        }));
      
      case 'anthropic':
        return {
          system: messages.find(msg => msg.role === 'system')?.content || '',
          messages: messages.filter(msg => msg.role !== 'system').map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        };
      
      case 'google':
        return {
          contents: messages.map(msg => ({
            role: msg.role === 'assistant' ? 'model' : msg.role,
            parts: [{ text: msg.content }]
          }))
        };
        
      case 'custom':
        return messages;
        
      default:
        return messages;
    }
  }
  
  private async makeApiRequest(conversation: Conversation, options: CompletionOptions): Promise<string> {
    const formattedMessages = this.formatMessagesForProvider(conversation.messages, options.provider);
    
    const requestBody = this.createRequestBody(formattedMessages, options);
    const headers = this.createRequestHeaders(options);
    
    const response = await fetch(options.endpoint!, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed with status ${response.status}: ${errorText}`);
    }
    
    const data = await response.json();
    return this.extractCompletionFromResponse(data, options.provider);
  }
  
  private async makeStreamingApiRequest(
    conversation: Conversation, 
    options: CompletionOptions, 
    onChunk: (chunk: string) => void
  ): Promise<string> {
    const formattedMessages = this.formatMessagesForProvider(conversation.messages, options.provider);
    
    const requestBody = this.createRequestBody(formattedMessages, options);
    const headers = this.createRequestHeaders(options);
    
    const response = await fetch(options.endpoint!, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed with status ${response.status}: ${errorText}`);
    }
    
    if (!response.body) {
      throw new Error('Response body is null');
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let fullText = '';
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const textChunk = this.processStreamChunk(chunk, options.provider);
        
        if (textChunk) {
          fullText += textChunk;
          onChunk(textChunk);
        }
      }
    } finally {
      reader.releaseLock();
    }
    
    return fullText;
  }
  
  private createRequestBody(formattedMessages: any, options: CompletionOptions): any {
    switch (options.provider) {
      case 'openai':
        return {
          model: options.model,
          messages: formattedMessages,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stream: options.stream,
          ...options.extraParams
        };
      
      case 'anthropic':
        return {
          model: options.model,
          system: formattedMessages.system,
          messages: formattedMessages.messages,
          max_tokens: options.maxTokens,
          temperature: options.temperature,
          stream: options.stream,
          ...options.extraParams
        };
      
      case 'google':
        return {
          model: options.model,
          contents: formattedMessages.contents,
          generationConfig: {
            temperature: options.temperature,
            maxOutputTokens: options.maxTokens,
          },
          ...options.extraParams
        };
        
      case 'custom':
        return {
          model: options.model,
          messages: formattedMessages,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stream: options.stream,
          ...options.extraParams
        };
        
      default:
        return {
          model: options.model,
          messages: formattedMessages,
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stream: options.stream,
          ...options.extraParams
        };
    }
  }
  
  private createRequestHeaders(options: CompletionOptions): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    switch (options.provider) {
      case 'openai':
        headers['Authorization'] = `Bearer ${options.apiKey}`;
        break;
      
      case 'anthropic':
        headers['x-api-key'] = options.apiKey!;
        headers['anthropic-version'] = '2023-06-01';
        break;
      
      case 'google':
        headers['Authorization'] = `Bearer ${options.apiKey}`;
        break;
        
      case 'custom':
        if (options.apiKey) {
          headers['Authorization'] = `Bearer ${options.apiKey}`;
        }
        break;
        
      default:
        if (options.apiKey) {
          headers['Authorization'] = `Bearer ${options.apiKey}`;
        }
    }
    
    return headers;
  }
  
  private extractCompletionFromResponse(response: any, provider: ModelProvider): string {
    switch (provider) {
      case 'openai':
        return response.choices?.[0]?.message?.content || '';
      
      case 'anthropic':
        return response.content?.[0]?.text || '';
      
      case 'google':
        return response.candidates?.[0]?.content?.parts?.[0]?.text || '';
        
      case 'custom':
        return response.content || response.text || response.message?.content || 
               response.choices?.[0]?.message?.content || response.completion || '';
        
      default:
        return JSON.stringify(response);
    }
  }
  
  private processStreamChunk(chunk: string, provider: ModelProvider): string {
    switch (provider) {
      case 'openai':
        try {
          const lines = chunk.split('\n').filter(line => line.trim().startsWith('data: ') && !line.includes('[DONE]'));
          let textChunk = '';
          
          for (const line of lines) {
            const jsonStr = line.replace(/^data: /, '').trim();
            if (!jsonStr) continue;
            
            const data = JSON.parse(jsonStr);
            const content = data.choices?.[0]?.delta?.content || '';
            textChunk += content;
          }
          
          return textChunk;
        } catch (e) {
          return '';
        }
      
      case 'anthropic':
        try {
          const lines = chunk.split('\n').filter(Boolean);
          let textChunk = '';
          
          for (const line of lines) {
            const data = JSON.parse(line);
            if (data.type === 'content_block_delta') {
              textChunk += data.delta?.text || '';
            }
          }
          
          return textChunk;
        } catch (e) {
          return '';
        }
      
      case 'google':
        try {
          const data = JSON.parse(chunk);
          return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
        } catch (e) {
          return '';
        }
        
      case 'custom':
      default:
        try {
          const data = JSON.parse(chunk);
          return data.text || data.content || data.completion || '';
        } catch (e) {
          return chunk;
        }
    }
  }
}