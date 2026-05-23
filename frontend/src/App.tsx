/**
 * DeepTrace React App
 */

import React from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Scan, BarChart2, Clock, Github } from "lucide-react";
import UploadPage from "./pages/Upload";
import AnalyticsPage from "./pages/Analytics";

const queryClient = new QueryClient();

function Sidebar() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-colors ${
      isActive
        ? "bg-indigo-50 text-indigo-700"
        : "text-gray-500 hover:text-gray-800 hover:bg-gray-100"
    }`;

  return (
    <aside className="w-60 flex-shrink-0 bg-white border-r border-gray-100 flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
            <Scan size={16} className="text-white" />
          </div>
          <div>
            <div className="font-bold text-gray-900 text-sm">DeepTrace</div>
            <div className="text-xs text-gray-400">Image Provenance</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        <NavLink to="/" className={linkClass} end>
          <Scan size={18} /> Analyze Image
        </NavLink>
        <NavLink to="/analytics" className={linkClass}>
          <BarChart2 size={18} /> Analytics
        </NavLink>
        <NavLink to="/history" className={linkClass}>
          <Clock size={18} /> History
        </NavLink>
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-gray-100">
        <a
          href="https://github.com/yourusername/deeptrace"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-xs text-gray-400 hover:text-gray-600 transition-colors"
        >
          <Github size={14} /> GitHub
        </a>
        <p className="text-xs text-gray-300 mt-1">v1.0.0</p>
      </div>
    </aside>
  );
}

function HistoryPage() {
  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">History</h1>
      <p className="text-gray-500 mb-6">Your past image analyses.</p>
      <div className="bg-white rounded-2xl border border-gray-100 p-12 text-center text-gray-400">
        <Clock size={40} className="mx-auto mb-3 text-gray-200" />
        <p className="font-medium">No predictions yet</p>
        <p className="text-sm mt-1">Analyzed images will appear here.</p>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex bg-gray-50 min-h-screen">
          <Sidebar />
          <main className="flex-1 p-8 overflow-auto">
            <Routes>
              <Route path="/" element={<UploadPage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/history" element={<HistoryPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
