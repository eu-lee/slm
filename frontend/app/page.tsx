"use client";

import { useState, useRef, useEffect, useCallback, FormEvent } from "react";

interface Message {
  id?: number;
  role: "user" | "assistant" | "system";
  content: string;
}

interface Conversation {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load conversation list
  const loadConversations = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/conversations`);
      if (res.ok) setConversations(await res.json());
    } catch (err) {
      console.error("Failed to load conversations:", err);
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // Load messages for a conversation
  async function selectConversation(id: number) {
    setActiveConvId(id);
    try {
      const res = await fetch(`${API_URL}/api/conversations/${id}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(
          data.messages.map((m: Message) => ({
            id: m.id,
            role: m.role,
            content: m.content,
          }))
        );
      }
    } catch (err) {
      console.error("Failed to load conversation:", err);
    }
  }

  // Create new conversation
  async function createConversation() {
    try {
      const res = await fetch(`${API_URL}/api/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Conversation" }),
      });
      if (res.ok) {
        const conv = await res.json();
        setConversations((prev) => [conv, ...prev]);
        setActiveConvId(conv.id);
        setMessages([]);
      }
    } catch (err) {
      console.error("Failed to create conversation:", err);
    }
  }

  // Rename conversation
  async function renameConversation(id: number, title: string) {
    try {
      const res = await fetch(`${API_URL}/api/conversations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (res.ok) {
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, title } : c))
        );
      }
    } catch (err) {
      console.error("Failed to rename conversation:", err);
    }
    setEditingId(null);
  }

  // Delete conversation
  async function deleteConversation(id: number) {
    try {
      await fetch(`${API_URL}/api/conversations/${id}`, { method: "DELETE" });
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConvId === id) {
        setActiveConvId(null);
        setMessages([]);
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isStreaming) return;

    // Auto-create conversation if none selected
    let convId = activeConvId;
    if (!convId) {
      try {
        const res = await fetch(`${API_URL}/api/conversations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: text.slice(0, 50) }),
        });
        if (res.ok) {
          const conv = await res.json();
          convId = conv.id;
          setActiveConvId(conv.id);
          setConversations((prev) => [conv, ...prev]);
        }
      } catch {
        return;
      }
    }

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);

    // Add placeholder for assistant response
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversation_id: convId,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      if (!reader) throw new Error("No response body");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEvent = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7);
          } else if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));
            if (currentEvent === "context_cleared") {
              setMessages((prev) => {
                const updated = [...prev];
                updated.splice(updated.length - 1, 0, {
                  role: "system",
                  content: data.message,
                });
                return updated;
              });
            } else if ("token" in data) {
              setMessages((prev) => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                updated[updated.length - 1] = {
                  ...last,
                  content: last.content + data.token,
                };
                return updated;
              });
            }
            currentEvent = "";
          }
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        updated[updated.length - 1] = {
          ...last,
          content: last.content || "[Error generating response]",
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
      loadConversations(); // Refresh sidebar order
    }
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-64" : "w-0"
        } flex-shrink-0 bg-gray-900 border-r border-gray-800 transition-all duration-200 overflow-hidden`}
      >
        <div className="flex flex-col h-full w-64">
          {/* Sidebar header */}
          <div className="flex items-center justify-between px-3 py-3 border-b border-gray-800">
            <span className="text-sm font-medium text-gray-300">Conversations</span>
            <button
              onClick={createConversation}
              className="text-gray-400 hover:text-white text-lg leading-none px-1"
              title="New conversation"
            >
              +
            </button>
          </div>

          {/* Conversation list */}
          <div className="flex-1 overflow-y-auto">
            {conversations.length === 0 && (
              <div className="px-3 py-4 text-xs text-gray-600">
                No conversations yet
              </div>
            )}
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group flex items-center gap-1 px-3 py-2 cursor-pointer text-sm border-b border-gray-800/50 ${
                  conv.id === activeConvId
                    ? "bg-gray-800 text-white"
                    : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                }`}
                onClick={() => selectConversation(conv.id)}
              >
                {editingId === conv.id ? (
                  <input
                    className="flex-1 bg-gray-700 text-white text-sm px-1 py-0.5 rounded outline-none"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") renameConversation(conv.id, editTitle);
                      if (e.key === "Escape") setEditingId(null);
                    }}
                    onBlur={() => renameConversation(conv.id, editTitle)}
                    onClick={(e) => e.stopPropagation()}
                    autoFocus
                  />
                ) : (
                  <>
                    <span className="flex-1 truncate">{conv.title}</span>
                    <div className="hidden group-hover:flex items-center gap-1">
                      <button
                        className="text-gray-500 hover:text-gray-300 text-xs"
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingId(conv.id);
                          setEditTitle(conv.title);
                        }}
                        title="Rename"
                      >
                        &#9998;
                      </button>
                      <button
                        className="text-gray-500 hover:text-red-400 text-xs"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteConversation(conv.id);
                        }}
                        title="Delete"
                      >
                        &times;
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className="flex items-center px-4 py-3 border-b border-gray-800">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-gray-400 hover:text-white mr-3 text-lg leading-none"
          >
            &#9776;
          </button>
          <h1 className="text-lg font-semibold text-gray-200">SLM Chat</h1>
          <span className="ml-2 text-xs text-gray-500">36M params</span>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
              {activeConvId
                ? "No messages yet"
                : "Create or select a conversation to start chatting"}
            </div>
          )}
          {messages.map((msg, i) =>
            msg.role === "system" ? (
              <div key={i} className="flex justify-center">
                <div className="text-xs text-yellow-500 bg-yellow-500/10 border border-yellow-500/20 rounded-lg px-3 py-1.5">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-800 text-gray-100"
                  }`}
                >
                  {msg.content || (
                    <span className="inline-block w-2 h-4 bg-gray-500 animate-pulse rounded-sm" />
                  )}
                </div>
              </div>
            )
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form
          onSubmit={handleSubmit}
          className="border-t border-gray-800 px-4 py-3"
        >
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a message..."
              disabled={isStreaming}
              className="flex-1 rounded-xl bg-gray-800 border border-gray-700 px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
