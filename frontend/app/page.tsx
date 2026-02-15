"use client";

import { useState, useRef, useEffect, useCallback, FormEvent } from "react";
import { useAuth } from "./auth";
import SlmIcon from "./icons/slm";

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
  const { user, token, isGuest, loginAsGuest, logout, loading } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showSavePopup, setShowSavePopup] = useState(false);
  const [guestDismissed, setGuestDismissed] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const isAuthed = !!token && !isGuest;
  const isGuestMode = !isAuthed;
  const hasMessages = messages.length > 0;

  const authHeaders = useCallback((): Record<string, string> => ({
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  }), [token]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-set guest mode if not logged in (no redirect)
  useEffect(() => {
    if (!loading && !token && !isGuest) {
      loginAsGuest();
    }
  }, [loading, token, isGuest, loginAsGuest]);

  // Clear state when switching to guest mode (e.g. after logout)
  useEffect(() => {
    if (isGuestMode) {
      setConversations([]);
      setActiveConvId(null);
      setMessages([]);
      setGuestDismissed(false);
    }
  }, [isGuestMode]);

  const loadConversations = useCallback(async () => {
    if (!isAuthed) return;
    try {
      const res = await fetch(`${API_URL}/api/conversations`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setConversations(await res.json());
    } catch (err) {
      console.error("Failed to load conversations:", err);
    }
  }, [isAuthed, token]);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  async function selectConversation(id: number) {
    setActiveConvId(id);
    try {
      const res = await fetch(`${API_URL}/api/conversations/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
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

  function newChat() {
    setActiveConvId(null);
    setMessages([]);
  }

  async function renameConversation(id: number, title: string) {
    try {
      const res = await fetch(`${API_URL}/api/conversations/${id}`, {
        method: "PATCH",
        headers: authHeaders(),
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

  async function deleteConversation(id: number) {
    try {
      await fetch(`${API_URL}/api/conversations/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConvId === id) {
        setActiveConvId(null);
        setMessages([]);
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
    }
  }

  async function readStream(res: Response) {
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
  }

  async function sendMessage(text: string) {
    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      let res: Response;

      if (isGuestMode) {
        const history = messages
          .filter((m) => m.role !== "system")
          .map((m) => ({ role: m.role, content: m.content }));
        res = await fetch(`${API_URL}/api/chat/guest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, history }),
        });
      } else {
        let convId = activeConvId;
        if (!convId) {
          const convRes = await fetch(`${API_URL}/api/conversations`, {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify({ title: text.slice(0, 50) }),
          });
          if (!convRes.ok) throw new Error("Failed to create conversation");
          const conv = await convRes.json();
          convId = conv.id;
          setActiveConvId(conv.id);
          setConversations((prev) => [conv, ...prev]);
        }
        res = await fetch(`${API_URL}/api/chat`, {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({ message: text, conversation_id: convId }),
        });
      }

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await readStream(res);
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
      if (isAuthed) loadConversations();
      // Show save popup after first guest reply (with delay)
      if (isGuestMode && !guestDismissed && !showSavePopup) {
        setTimeout(() => setShowSavePopup(true), 3000);
      }
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isStreaming) return;
    await sendMessage(text);
  }

  function handleGuestDismiss() {
    setShowSavePopup(false);
    setGuestDismissed(true);
  }

  if (loading) return null;

  return (
    <div className="flex h-screen bg-[#212121]">
      {/* Save Chat Popup */}
      {showSavePopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[#2f2f2f] rounded-2xl p-6 max-w-sm w-full mx-4 shadow-xl">
            <h2 className="text-lg font-semibold text-gray-100 mb-2">Save your chats?</h2>
            <p className="text-sm text-gray-400 mb-5">
              Guest chats are not saved and will be lost when you close the browser. Log in or create an account to keep your conversation history.
            </p>
            <div className="flex flex-col gap-2">
              <button
                onClick={() => { window.location.href = "/login"; }}
                className="w-full rounded-xl bg-white text-black px-4 py-2.5 text-sm font-medium hover:bg-gray-200 transition-colors"
              >
                Log in or Register
              </button>
              <button
                onClick={handleGuestDismiss}
                className="w-full rounded-xl bg-transparent border border-gray-600 text-gray-300 px-4 py-2.5 text-sm font-medium hover:bg-gray-700 transition-colors"
              >
                Continue without saving
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar (authed only) */}
      {isAuthed && (
        <div
          className={`${
            sidebarOpen ? "w-[260px]" : "w-0"
          } flex-shrink-0 bg-[#171717] transition-all duration-200 overflow-hidden`}
        >
          <div className="flex flex-col h-full w-[260px]">
            <div className="px-3 pt-3 pb-1">
              <button
                onClick={() => { newChat(); }}
                className="flex items-center gap-2 w-full px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:bg-[#2f2f2f] transition-colors"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 20h9" /><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z" />
                </svg>
                New chat
              </button>
            </div>

            <div className="px-3 pt-4 pb-1">
              <span className="text-xs font-medium text-gray-500 px-3">Your chats</span>
            </div>

            <div className="flex-1 overflow-y-auto px-3">
              {conversations.length === 0 && (
                <div className="px-3 py-3 text-xs text-gray-600">
                  No conversations yet
                </div>
              )}
              {conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`group flex items-center gap-1 px-3 py-2 rounded-lg cursor-pointer text-sm mb-0.5 ${
                    conv.id === activeConvId
                      ? "bg-[#2f2f2f] text-white"
                      : "text-gray-400 hover:bg-[#2f2f2f] hover:text-gray-200"
                  }`}
                  onClick={() => selectConversation(conv.id)}
                >
                  {editingId === conv.id ? (
                    <input
                      className="flex-1 bg-[#3f3f3f] text-white text-sm px-2 py-0.5 rounded outline-none"
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
                          className="text-gray-500 hover:text-gray-300 text-xs p-0.5"
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
                          className="text-gray-500 hover:text-red-400 text-xs p-0.5"
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

            {/* User info at bottom */}
            <div className="border-t border-[#2f2f2f] px-4 py-3 flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-xs font-medium text-white flex-shrink-0">
                {user?.username?.slice(0, 2).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm text-gray-200 truncate">{user?.username}</div>
              </div>
              <button
                onClick={logout}
                className="text-xs text-gray-500 hover:text-red-400 transition-colors flex-shrink-0"
              >
                Log out
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Top bar */}
        <header className="relative z-10 flex items-center h-12 px-4">
          {isAuthed && (
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-gray-400 hover:text-white mr-3"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 3v18" />
              </svg>
            </button>
          )}
          {/* Guest: logo/new-chat button to the left of title */}
          {isGuestMode && (
            <button
              onClick={hasMessages ? newChat : undefined}
              className="group mr-2 flex-shrink-0"
              title={hasMessages ? "New chat" : "SLM Chat"}
            >
              {/* Default: logo icon */}
              <SlmIcon size={28} className="group-hover:hidden" />
              {/* Hover: new chat pencil inside squircle */}
              <svg className="hidden group-hover:block text-gray-400" width="28" height="28" viewBox="0 0 640 640" fill="none">
                <rect width="640" height="640" rx="128" fill="none" stroke="currentColor" strokeWidth="32" />
                <g transform="translate(180, 180) scale(11.7)">
                  <path d="M12 20h9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </g>
              </svg>
            </button>
          )}
          <span className="text-base font-medium text-gray-200">SLM Chat</span>
          {/* Guest: auth buttons on the right */}
          {isGuestMode && (
            <div className="ml-auto flex items-center gap-2">
              <a
                href="/login"
                className="rounded-lg border border-[#3f3f3f] px-3 py-1.5 text-sm text-gray-300 hover:bg-[#2f2f2f] transition-colors"
              >
                Log In
              </a>
              <a
                href="/login?register=true"
                className="rounded-lg bg-white px-3 py-1.5 text-sm font-medium text-black hover:bg-gray-200 transition-colors"
              >
                Sign Up for Free
              </a>
            </div>
          )}
        </header>
        {/* Chat area */}
        {!hasMessages ? (
          /* Empty state: centered prompt */
          <div className="flex-1 flex flex-col items-center justify-center px-4 -mt-52">
            <h2 className="text-[28px] font-normal text-gray-100 mb-8">
              What&apos;s on your mind today?
            </h2>
            <form onSubmit={handleSubmit} className="w-full max-w-[680px]">
              <div className="relative bg-[#2f2f2f] rounded-full has-[textarea:not([data-single-line])]:rounded-3xl transition-all">
                <div className="flex items-end gap-2 px-3 py-2.5">
                  <button
                    type="button"
                    className="flex-shrink-0 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-gray-300"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="5" x2="12" y2="19"></line>
                      <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                  </button>
                  <textarea
                    value={input}
                    onChange={(e) => {
                      setInput(e.target.value);
                      // Auto-resize
                      e.target.style.height = 'auto';
                      const newHeight = Math.min(e.target.scrollHeight, 200);
                      e.target.style.height = newHeight + 'px';
                      // Toggle attribute for styling
                      if (newHeight > 24) {
                        e.target.removeAttribute('data-single-line');
                      } else {
                        e.target.setAttribute('data-single-line', '');
                      }
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e);
                      }
                    }}
                    placeholder="Ask anything"
                    disabled={isStreaming}
                    rows={1}
                    data-single-line=""
                    className="flex-1 bg-transparent border-none px-2 py-1 text-base text-gray-100 placeholder-gray-500 focus:outline-none disabled:opacity-50 resize-none overflow-hidden"
                    style={{ minHeight: '24px', maxHeight: '200px' }}
                  />
                  <button
                    type="submit"
                    disabled={isStreaming || !input.trim()}
                    className="flex-shrink-0 w-8 h-8 rounded-full bg-white flex items-center justify-center disabled:opacity-30 disabled:bg-gray-600 transition-colors"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 19V5" /><path d="M5 12l7-7 7 7" />
                    </svg>
                  </button>
                </div>
              </div>
            </form>
          </div>
        ) : (
          /* Active chat */
          <>
            <div className="flex-1 overflow-y-auto">
              <div className="max-w-[680px] mx-auto px-4 py-6 space-y-6">
                {messages.map((msg, i) =>
                  msg.role === "system" ? (
                    <div key={i} className="flex justify-center">
                      <div className="text-xs text-yellow-500 bg-yellow-500/10 border border-yellow-500/20 rounded-lg px-3 py-1.5">
                        {msg.content}
                      </div>
                    </div>
                  ) : (
                    <div key={i} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}>
                      {msg.role === "assistant" && (
                        <SlmIcon size={28} className="flex-shrink-0 mt-0.5" />
                      )}
                      <div
                        className={`text-sm leading-relaxed max-w-[85%] ${
                          msg.role === "user"
                            ? "bg-[#2f2f2f] text-gray-100 rounded-2xl px-4 py-2.5"
                            : "text-gray-100"
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
            </div>

            {/* Bottom input */}
            <div className="px-4 pb-4 pt-2">
              <form onSubmit={handleSubmit} className="max-w-[680px] mx-auto">
                <div className="relative bg-[#2f2f2f] rounded-full has-[textarea:not([data-single-line])]:rounded-3xl transition-all">
                  <div className="flex items-end gap-2 px-3 py-2.5">
                    <button
                      type="button"
                      className="flex-shrink-0 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-gray-300"
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="5" x2="12" y2="19"></line>
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                      </svg>
                    </button>
                    <textarea
                      value={input}
                      onChange={(e) => {
                        setInput(e.target.value);
                        // Auto-resize
                        e.target.style.height = 'auto';
                        const newHeight = Math.min(e.target.scrollHeight, 200);
                        e.target.style.height = newHeight + 'px';
                        // Toggle attribute for styling
                        if (newHeight > 24) {
                          e.target.removeAttribute('data-single-line');
                        } else {
                          e.target.setAttribute('data-single-line', '');
                        }
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSubmit(e);
                        }
                      }}
                      placeholder="Ask anything"
                      disabled={isStreaming}
                      rows={1}
                      data-single-line=""
                      className="flex-1 bg-transparent border-none px-2 py-1 text-base text-gray-100 placeholder-gray-500 focus:outline-none disabled:opacity-50 resize-none overflow-hidden"
                      style={{ minHeight: '24px', maxHeight: '200px' }}
                    />
                    <button
                      type="submit"
                      disabled={isStreaming || !input.trim()}
                      className="flex-shrink-0 w-8 h-8 rounded-full bg-white flex items-center justify-center disabled:opacity-30 disabled:bg-gray-600 transition-colors"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 19V5" /><path d="M5 12l7-7 7 7" />
                      </svg>
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
