"use client";

import { useState, useEffect, FormEvent } from "react";
import { useAuth } from "../auth";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function LoginPage() {
  const { login, loginAsGuest } = useAuth();
  const [isRegister, setIsRegister] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("register") === "true") {
      setIsRegister(true);
    }
  }, []);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const endpoint = isRegister ? "/api/auth/register" : "/api/auth/login";

    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        const data = await res.json();
        setError(data.detail || "Something went wrong");
        return;
      }

      const data = await res.json();
      login(data.token, data.user);
      window.location.href = "/";
    } catch {
      setError("Failed to connect to server");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-[#212121]">
      <div className="w-full max-w-sm mx-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-semibold text-gray-100">EuGPT</h1>
          <p className="text-sm text-gray-500 mt-2">36M parameter language model</p>
        </div>

        <div className="bg-[#2f2f2f] rounded-2xl p-6">
          <h2 className="text-lg font-medium text-gray-200 text-center mb-5">
            {isRegister ? "Create your account" : "Welcome back"}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-3">
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full rounded-xl bg-[#3f3f3f] border-none px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-500"
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full rounded-xl bg-[#3f3f3f] border-none px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-500"
            />

            {error && (
              <div className="text-sm text-red-400 bg-red-400/10 rounded-lg px-3 py-2">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-white text-black px-4 py-3 text-sm font-medium hover:bg-gray-200 disabled:opacity-50 transition-colors"
            >
              {loading ? "..." : isRegister ? "Create Account" : "Log In"}
            </button>
          </form>

          <div className="text-center mt-4">
            <button
              onClick={() => {
                setIsRegister(!isRegister);
                setError("");
              }}
              className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
            >
              {isRegister
                ? "Already have an account? Log in"
                : "Don't have an account? Register"}
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3 my-4">
          <div className="flex-1 border-t border-[#3f3f3f]" />
          <span className="text-xs text-gray-600">or</span>
          <div className="flex-1 border-t border-[#3f3f3f]" />
        </div>

        <button
          onClick={() => {
            loginAsGuest();
            window.location.href = "/";
          }}
          className="w-full rounded-xl bg-transparent border border-[#3f3f3f] text-gray-400 px-4 py-3 text-sm font-medium hover:bg-[#2f2f2f] hover:text-gray-200 transition-colors"
        >
          Continue as Guest
        </button>
      </div>
    </div>
  );
}
