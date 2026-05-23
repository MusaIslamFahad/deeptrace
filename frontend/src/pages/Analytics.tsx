/**
 * DeepTrace Analytics Page
 * Shows prediction volume by source, confidence trends, and model info.
 */

import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, PieChart, Pie, Cell, ResponsiveContainer,
} from "recharts";
import { getHealth } from "../api/client";
import { SOURCE_COLORS, SOURCE_LABELS } from "../api/client";
import type { SourceType } from "../api/client";

// Mock analytics data — in production, this comes from a /api/v1/analytics endpoint
const MOCK_WEEKLY_DATA = [
  { day: "Mon", stable_diffusion: 120, midjourney: 85, dalle3: 60, flux: 45, real: 190 },
  { day: "Tue", stable_diffusion: 145, midjourney: 92, dalle3: 55, flux: 38, real: 170 },
  { day: "Wed", stable_diffusion: 98,  midjourney: 110, dalle3: 70, flux: 52, real: 200 },
  { day: "Thu", stable_diffusion: 180, midjourney: 78,  dalle3: 48, flux: 60, real: 155 },
  { day: "Fri", stable_diffusion: 210, midjourney: 130, dalle3: 90, flux: 75, real: 220 },
  { day: "Sat", stable_diffusion: 90,  midjourney: 65,  dalle3: 40, flux: 28, real: 140 },
  { day: "Sun", stable_diffusion: 75,  midjourney: 55,  dalle3: 35, flux: 22, real: 120 },
];

const MOCK_CONFIDENCE_TREND = [
  { day: "Mon", avg_confidence: 0.87 },
  { day: "Tue", avg_confidence: 0.85 },
  { day: "Wed", avg_confidence: 0.89 },
  { day: "Thu", avg_confidence: 0.86 },
  { day: "Fri", avg_confidence: 0.91 },
  { day: "Sat", avg_confidence: 0.88 },
  { day: "Sun", avg_confidence: 0.90 },
];

const SOURCES = Object.keys(SOURCE_LABELS) as SourceType[];

// ---------------------------------------------------------------------------
// Stat card
// ---------------------------------------------------------------------------

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 p-5 shadow-sm">
      <p className="text-sm text-gray-400 font-medium mb-1">{label}</p>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Analytics Page
// ---------------------------------------------------------------------------

export default function AnalyticsPage() {
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
  }, []);

  // Pie data — total by source
  const pieData = SOURCES.map((src) => ({
    name: SOURCE_LABELS[src],
    value: MOCK_WEEKLY_DATA.reduce((sum, d) => sum + (d as any)[src], 0),
    color: SOURCE_COLORS[src],
  }));

  const totalPredictions = pieData.reduce((s, d) => s + d.value, 0);
  const aiPercentage = (
    (totalPredictions - pieData.find((d) => d.name === "Real Photo")!.value) /
    totalPredictions *
    100
  ).toFixed(1);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Analytics</h1>
        <p className="text-gray-500">Platform-wide prediction statistics for the past 7 days.</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Total Predictions"
          value={totalPredictions.toLocaleString()}
          sub="Last 7 days"
        />
        <StatCard
          label="AI Detected"
          value={`${aiPercentage}%`}
          sub="Of all predictions"
        />
        <StatCard
          label="Avg Confidence"
          value="88.1%"
          sub="Calibrated probability"
        />
        <StatCard
          label="Model Version"
          value={health?.version || "—"}
          sub={health ? `Status: ${health.status}` : "Loading..."}
        />
      </div>

      {/* Bar chart: predictions by source */}
      <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm mb-6">
        <h2 className="text-base font-semibold text-gray-800 mb-4">
          Daily Predictions by Source
        </h2>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={MOCK_WEEKLY_DATA} margin={{ top: 0, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="day" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {SOURCES.map((src) => (
              <Bar key={src} dataKey={src} name={SOURCE_LABELS[src]}
                   fill={SOURCE_COLORS[src]} stackId="a" />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Bottom row: confidence trend + pie */}
      <div className="grid grid-cols-2 gap-6">
        {/* Line chart: confidence trend */}
        <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
          <h2 className="text-base font-semibold text-gray-800 mb-4">
            Avg Confidence Trend
          </h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={MOCK_CONFIDENCE_TREND}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="day" tick={{ fontSize: 12 }} />
              <YAxis domain={[0.7, 1.0]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
              <Line type="monotone" dataKey="avg_confidence" stroke="#7F77DD"
                    strokeWidth={2} dot={{ r: 4 }} name="Avg confidence" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Pie chart: share by source */}
        <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
          <h2 className="text-base font-semibold text-gray-800 mb-4">
            Share by Source
          </h2>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%" cy="50%"
                innerRadius={55} outerRadius={85}
                dataKey="value"
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {pieData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
