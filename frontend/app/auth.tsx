"use client";

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";

interface User {
  id: number;
  username: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isGuest: boolean;
  login: (token: string, user: User) => void;
  loginAsGuest: () => void;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  isGuest: false,
  login: () => {},
  loginAsGuest: () => {},
  logout: () => {},
  loading: true,
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isGuest, setIsGuest] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedToken = localStorage.getItem("token");
    const savedUser = localStorage.getItem("user");
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser(JSON.parse(savedUser));
    }
    // Guest state is never persisted — it resets on browser close
    const savedGuest = sessionStorage.getItem("guest");
    if (savedGuest) {
      setIsGuest(true);
    }
    setLoading(false);
  }, []);

  const login = useCallback((newToken: string, newUser: User) => {
    setToken(newToken);
    setUser(newUser);
    setIsGuest(false);
    localStorage.setItem("token", newToken);
    localStorage.setItem("user", JSON.stringify(newUser));
    sessionStorage.removeItem("guest");
  }, []);

  const loginAsGuest = useCallback(() => {
    setToken(null);
    setUser(null);
    setIsGuest(true);
    sessionStorage.setItem("guest", "true");
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
    setIsGuest(false);
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    sessionStorage.removeItem("guest");
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, isGuest, login, loginAsGuest, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
