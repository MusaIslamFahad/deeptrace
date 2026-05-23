/**
 * DeepTrace Upload Page
 * Drag-and-drop image upload with real-time prediction results.
 */

import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Loader2, AlertCircle, CheckCircle2, Zap } from "lucide-react";
import { predictImage, SOURCE_LABELS, SOURCE_COLORS } from "../api/client";
import type { PredictionResult, SourceType } from "../api/client";

// ---------------------------------------------------------------------------
// Result Card
// ---------------------------------------------------------------------------

function ProbabilityBar({ label, value, color, isTop }: {
  label: string; value: number; color: string; isTop: boolean;
}) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-sm mb-1">
        <span className={`font-medium ${isTop ? "text-gray-900" : "text-gray-500"}`}>
          {label}
        </span>
        <span className={`font-mono ${isTop ? "font-bold" : "text-gray-400"}`}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2">
        <div
          className="h-2 rounded-full transition-all duration-700"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function ResultCard({ result, imageUrl }: { result: PredictionResult; imageUrl: string }) {
  const source = result.predicted_source as SourceType;
  const color = SOURCE_COLORS[source];
  const label = SOURCE_LABELS[source];

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100">
      <div className="flex gap-6 p-6">
        {/* Image */}
        <div className="flex-shrink-0">
          <img
            src={imageUrl}
            alt="Uploaded"
            className="w-40 h-40 object-cover rounded-xl"
          />
        </div>

        {/* Results */}
        <div className="flex-1 min-w-0">
          {/* Badge */}
          <div className="flex items-center gap-3 mb-3">
            <span
              className="px-3 py-1 rounded-full text-white text-sm font-semibold"
              style={{ backgroundColor: color }}
            >
              {label}
            </span>
            {result.is_ai_generated ? (
              <span className="flex items-center gap-1 text-amber-600 text-sm font-medium">
                <Zap size={14} /> AI Generated
              </span>
            ) : (
              <span className="flex items-center gap-1 text-green-600 text-sm font-medium">
                <CheckCircle2 size={14} /> Real Photo
              </span>
            )}
          </div>

          {/* Confidence */}
          <div className="text-3xl font-bold mb-1" style={{ color }}>
            {(result.confidence * 100).toFixed(1)}%
          </div>
          <div className="text-gray-400 text-sm mb-4">confidence</div>

          {/* Explanation */}
          {result.explanation_text && (
            <p className="text-sm text-gray-600 italic mb-4 bg-gray-50 rounded-lg p-3">
              {result.explanation_text}
            </p>
          )}

          {/* Meta */}
          <div className="flex gap-4 text-xs text-gray-400">
            <span>⚡ {result.processing_ms}ms</span>
            <span>Model {result.model_version}</span>
          </div>
        </div>
      </div>

      {/* Probability bars */}
      <div className="px-6 pb-6">
        <div className="h-px bg-gray-100 mb-4" />
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Per-source probabilities
        </p>
        {(Object.entries(result.per_class_probs) as [SourceType, number][])
          .sort((a, b) => b[1] - a[1])
          .map(([src, prob]) => (
            <ProbabilityBar
              key={src}
              label={SOURCE_LABELS[src]}
              value={prob}
              color={SOURCE_COLORS[src]}
              isTop={src === source}
            />
          ))}
      </div>

      {/* Grad-CAM */}
      {result.gradcam_url && (
        <div className="border-t border-gray-100 p-6">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            XAI Analysis
          </p>
          <img
            src={result.gradcam_url}
            alt="Grad-CAM explanation"
            className="w-full rounded-xl"
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload Page
// ---------------------------------------------------------------------------

type UploadState =
  | { status: "idle" }
  | { status: "uploading"; file: File; preview: string }
  | { status: "success"; result: PredictionResult; preview: string }
  | { status: "error"; message: string };

export default function UploadPage() {
  const [state, setState] = useState<UploadState>({ status: "idle" });
  const [includeXAI, setIncludeXAI] = useState(false);
  const [includeNL, setIncludeNL] = useState(false);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      const preview = URL.createObjectURL(file);
      setState({ status: "uploading", file, preview });

      try {
        const result = await predictImage(file, {
          gradcam: includeXAI,
          explain: includeNL,
        });
        setState({ status: "success", result, preview });
      } catch (err: any) {
        setState({
          status: "error",
          message: err?.response?.data?.detail || err.message || "Prediction failed",
        });
      }
    },
    [includeXAI, includeNL]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    disabled: state.status === "uploading",
  });

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Analyze Image</h1>
        <p className="text-gray-500">
          Upload any image to detect whether it was AI-generated and identify the source model.
        </p>
      </div>

      {/* Options */}
      <div className="flex gap-6 mb-6">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={includeXAI}
            onChange={(e) => setIncludeXAI(e.target.checked)}
            className="rounded"
          />
          <span className="text-sm text-gray-600">Include Grad-CAM explanation</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={includeNL}
            onChange={(e) => setIncludeNL(e.target.checked)}
            className="rounded"
          />
          <span className="text-sm text-gray-600">Natural language explanation</span>
        </label>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all
          ${isDragActive ? "border-indigo-400 bg-indigo-50" : "border-gray-200 hover:border-gray-300 bg-gray-50"}
          ${state.status === "uploading" ? "opacity-60 cursor-not-allowed" : ""}
        `}
      >
        <input {...getInputProps()} />

        {state.status === "uploading" ? (
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="animate-spin text-indigo-500" size={40} />
            <p className="text-gray-600 font-medium">Analyzing image...</p>
            <p className="text-gray-400 text-sm">{state.file.name}</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <Upload className="text-gray-300" size={40} />
            <p className="text-gray-600 font-medium">
              {isDragActive ? "Drop image here" : "Drag & drop or click to upload"}
            </p>
            <p className="text-gray-400 text-sm">JPEG, PNG, WebP — up to 10 MB</p>
          </div>
        )}
      </div>

      {/* Error */}
      {state.status === "error" && (
        <div className="mt-4 flex items-start gap-3 bg-red-50 border border-red-200 rounded-xl p-4">
          <AlertCircle className="text-red-500 mt-0.5 flex-shrink-0" size={18} />
          <div>
            <p className="text-red-700 font-medium text-sm">Analysis failed</p>
            <p className="text-red-500 text-sm mt-1">{state.message}</p>
          </div>
        </div>
      )}

      {/* Result */}
      {state.status === "success" && (
        <div className="mt-6">
          <ResultCard result={state.result} imageUrl={state.preview} />
          <button
            onClick={() => setState({ status: "idle" })}
            className="mt-4 w-full py-3 border border-gray-200 rounded-xl text-gray-600 hover:bg-gray-50 transition-colors text-sm font-medium"
          >
            Analyze another image
          </button>
        </div>
      )}
    </div>
  );
}
