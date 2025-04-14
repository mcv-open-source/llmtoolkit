import { useEffect, useState, useRef, useCallback } from 'react';
import { LLMToolkit, Message, CompletionOptions, Conversation } from './llm-toolkit';

export function useConversation(
  toolkit: LLMToolkit,
  conversationId: string,
  initialSystemPrompt?: string
) {
  const [currentConversationId, setCurrentConversationId] = useState<string>(
    conversationId || toolkit.createConversation(initialSystemPrompt)
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    try {
      const conversation = toolkit.getConversation(currentConversationId);
      setMessages(conversation.messages);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error occurred'));
    }
  }, [currentConversationId, toolkit]);
  
  const sendMessage = useCallback(async (
    userMessage: string,
    options?: Partial<CompletionOptions>
  ) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await toolkit.sendCompletion(currentConversationId, userMessage, options);
      const conversation = toolkit.getConversation(currentConversationId);
      setMessages([...conversation.messages]);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to send message'));
    } finally {
      setIsLoading(false);
    }
  }, [currentConversationId, toolkit]);
  
  const streamMessage = useCallback(async (
    userMessage: string,
    onChunk: (chunk: string) => void,
    options?: Partial<CompletionOptions>
  ) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await toolkit.streamCompletion(currentConversationId, userMessage, onChunk, options);
      const conversation = toolkit.getConversation(currentConversationId);
      setMessages([...conversation.messages]);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to stream message'));
    } finally {
      setIsLoading(false);
    }
  }, [currentConversationId, toolkit]);
  
  const addContext = useCallback((content: string, metadata?: Record<string, any>) => {
    try {
      toolkit.addContext(currentConversationId, content, metadata);
      const conversation = toolkit.getConversation(currentConversationId);
      setMessages([...conversation.messages]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to add context'));
    }
  }, [currentConversationId, toolkit]);
  
  const createNewConversation = useCallback((systemPrompt?: string) => {
    const newConversationId = toolkit.createConversation(systemPrompt);
    setCurrentConversationId(newConversationId);
    return newConversationId;
  }, [toolkit]);
  
  const switchConversation = useCallback((conversationId: string) => {
    setCurrentConversationId(conversationId);
  }, []);
  
  return {
    conversationId: currentConversationId,
    messages,
    isLoading,
    error,
    sendMessage,
    streamMessage,
    addContext,
    createNewConversation,
    switchConversation
  };
}

export function useConversations(toolkit: LLMToolkit, initialSystemPrompt?: string) {
  const [conversationIds, setConversationIds] = useState<string[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    if (conversationIds.length === 0) {
      const newConversationId = toolkit.createConversation(initialSystemPrompt);
      setConversationIds([newConversationId]);
      setCurrentConversationId(newConversationId);
    }
  }, [conversationIds.length, initialSystemPrompt, toolkit]);
  
  const createConversation = useCallback((systemPrompt?: string) => {
    const newConversationId = toolkit.createConversation(systemPrompt);
    setConversationIds(prev => [...prev, newConversationId]);
    setCurrentConversationId(newConversationId);
    return newConversationId;
  }, [toolkit]);
   
  const deleteConversation = useCallback((conversationId: string) => {
    const deleted = toolkit.deleteConversation(conversationId);
    
    if (deleted) {
      setConversationIds(prev => prev.filter(id => id !== conversationId));
      
      if (currentConversationId === conversationId) {
        if (conversationIds.length > 1) {
          const newCurrentId = conversationIds.find(id => id !== conversationId);
          setCurrentConversationId(newCurrentId || null);
        } else {
          const newConversationId = toolkit.createConversation(initialSystemPrompt);
          setConversationIds([newConversationId]);
          setCurrentConversationId(newConversationId);
        }
      }
    }
    
    return deleted;
  }, [conversationIds, currentConversationId, initialSystemPrompt, toolkit]);
  
  const getAllConversations = useCallback((): Conversation[] => {
    return toolkit.getAllConversations();
  }, [toolkit]);
  
  const switchConversation = useCallback((conversationId: string) => {
    if (conversationIds.includes(conversationId)) {
      setCurrentConversationId(conversationId);
      return true;
    }
    return false;
  }, [conversationIds]);
  
  const getCurrentConversation = useCallback((): Conversation | null => {
    if (!currentConversationId) return null;
    try {
      return toolkit.getConversation(currentConversationId);
    } catch {
      return null;
    }
  }, [currentConversationId, toolkit]);
  
  return {
    conversationIds,
    currentConversationId,
    createConversation,
    deleteConversation,
    getAllConversations,
    switchConversation,
    getCurrentConversation,
    isLoading
  };
}

export function useStreamingResponse() {
  const [streamedText, setStreamedText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const fullResponseRef = useRef('');
  
  const handleChunk = useCallback((chunk: string) => {
    fullResponseRef.current += chunk;
    setStreamedText(fullResponseRef.current);
  }, []);
  
  const resetStream = useCallback(() => {
    fullResponseRef.current = '';
    setStreamedText('');
  }, []);
  
  const startStreaming = useCallback(() => {
    resetStream();
    setIsStreaming(true);
  }, [resetStream]);
  
  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
  }, []);
  
  return {
    streamedText,
    isStreaming,
    handleChunk,
    resetStream,
    startStreaming,
    stopStreaming
  };
}