/**
 * DeepTrace API Client
 * Typed wrapper around the FastAPI backend.
 */

import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "dev-key-123";

export const apiClient = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  headers: { "X-API-Key": API_KEY },
});

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SourceType =
  | "stable_diffusion"
  | "midjourney"
  | "dalle3"
  | "flux"
  | "real";

export interface PredictionResult {
  image_id: string;
  filename: string;
  predicted_source: SourceType;
  confidence: number;
  is_ai_generated: boolean;
  per_class_probs: Record<SourceType, number>;
  gradcam_url?: string;
  explanation_text?: string;
  processing_ms: number;
  model_version: string;
  created_at: string;
}

export interface BatchJobStatus {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  total: number;
  results?: PredictionResult[];
  error?: string;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  model_loaded: boolean;
  redis_connected: boolean;
  version: string;
  uptime_seconds: number;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function predictImage(
  file: File,
  options: { gradcam?: boolean; explain?: boolean } = {}
): Promise<PredictionResult> {
  const form = new FormData();
  form.append("file", file);

  const params = new URLSearchParams();
  if (options.gradcam) params.set("gradcam", "true");
  if (options.explain) params.set("explain", "true");

  const { data } = await apiClient.post<PredictionResult>(
    `/predict?${params}`,
    form
  );
  return data;
}

export async function submitBatch(files: File[]): Promise<{ job_id: string; poll_url: string }> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const { data } = await apiClient.post("/predict/batch", form);
  return data;
}

export async function pollJob(job_id: string): Promise<BatchJobStatus> {
  const { data } = await apiClient.get<BatchJobStatus>(`/jobs/${job_id}`);
  return data;
}

export async function getHealth(): Promise<HealthStatus> {
  const { data } = await axios.get<HealthStatus>(`${BASE_URL}/health`);
  return data;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export const SOURCE_LABELS: Record<SourceType, string> = {
  stable_diffusion: "Stable Diffusion",
  midjourney: "Midjourney",
  dalle3: "DALL·E 3",
  flux: "Flux",
  real: "Real Photo",
};

export const SOURCE_COLORS: Record<SourceType, string> = {
  stable_diffusion: "#7F77DD",
  midjourney: "#1D9E75",
  dalle3: "#D85A30",
  flux: "#378ADD",
  real: "#639922",
};
