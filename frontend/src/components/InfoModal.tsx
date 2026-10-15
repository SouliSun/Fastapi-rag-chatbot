import { X, Server, Database, Brain, Shield, Info, CheckCircle2, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { HealthResponse } from "@/types/api"
import { API_BASE_URL } from "@/services/api"

interface InfoModalProps {
  isOpen: boolean
  onClose: () => void
  health: HealthResponse | null
  onRefreshHealth: () => void
}

export const InfoModal: React.FC<InfoModalProps> = ({
  isOpen,
  onClose,
  health,
  onRefreshHealth,
}) => {
  if (!isOpen) return null

  const isHealthy = health?.status === "healthy" && health.test_retrieval_ok

  return (
    <div
      className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto"
      role="dialog"
      aria-modal="true"
      aria-labelledby="info-modal-title"
    >
      <div className="bg-card border border-border rounded-2xl shadow-xl max-w-lg w-full p-6 space-y-5 animate-in fade-in-0 zoom-in-95">
        {/* Modal Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
              <Info className="h-5 w-5" />
            </div>
            <div>
              <h2 id="info-modal-title" className="text-base font-semibold text-foreground">
                FAQ RAG System Telemetry
              </h2>
              <p className="text-xs text-muted-foreground">
                Runtime configuration & health diagnostics
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 text-muted-foreground"
            aria-label="Close dialog"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Live Service Status Banner */}
        <div
          className={`p-3.5 rounded-xl border flex items-start gap-3 ${
            isHealthy
              ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-800 dark:text-emerald-300"
              : "bg-destructive/10 border-destructive/30 text-destructive"
          }`}
        >
          {isHealthy ? (
            <CheckCircle2 className="h-5 w-5 shrink-0 mt-0.5 text-emerald-600 dark:text-emerald-400" />
          ) : (
            <AlertCircle className="h-5 w-5 shrink-0 mt-0.5 text-destructive" />
          )}
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2 font-semibold">
              <span>Status: {isHealthy ? "Operational (Healthy)" : "Unavailable"}</span>
              <Badge variant={isHealthy ? "success" : "destructive"} className="text-[10px]">
                {health?.status || "Unreachable"}
              </Badge>
            </div>
            <p className="leading-relaxed opacity-90 text-[11px]">
              {isHealthy
                ? `FastAPI backend connected and ChromaDB semantic search is responding at ${API_BASE_URL}.`
                : health?.error ||
                  `Could not establish connection to ${API_BASE_URL}. Run uvicorn api:app to start the backend.`}
            </p>
          </div>
        </div>

        {/* Technical Architecture Specs */}
        <div className="space-y-3 text-xs">
          <h3 className="font-semibold uppercase text-muted-foreground tracking-wider text-[11px]">
            System Architecture Components
          </h3>

          <div className="divide-y divide-border/60 rounded-xl border bg-muted/20">
            <div className="p-3 flex items-center justify-between">
              <span className="flex items-center gap-2 text-muted-foreground">
                <Database className="h-3.5 w-3.5 text-primary" />
                Indexed FAQ Records
              </span>
              <span className="font-mono font-medium text-foreground">
                {health?.indexed_faqs ?? "Not indexed"}
              </span>
            </div>

            <div className="p-3 flex items-center justify-between">
              <span className="flex items-center gap-2 text-muted-foreground">
                <Brain className="h-3.5 w-3.5 text-primary" />
                Embedding Model
              </span>
              <span className="font-mono font-medium text-foreground text-[11px]">
                {health?.embedding_model || "BAAI/bge-small-en-v1.5"}
              </span>
            </div>

            <div className="p-3 flex items-center justify-between">
              <span className="flex items-center gap-2 text-muted-foreground">
                <Server className="h-3.5 w-3.5 text-primary" />
                Vector Storage Path
              </span>
              <span className="font-mono text-foreground text-[11px]">
                {health?.database_path || "./chroma_db"}
              </span>
            </div>

            <div className="p-3 flex items-center justify-between">
              <span className="flex items-center gap-2 text-muted-foreground">
                <Shield className="h-3.5 w-3.5 text-primary" />
                Generation Provider
              </span>
              <span className="font-medium text-foreground text-[11px]">
                Groq API (openai/gpt-oss-20b)
              </span>
            </div>
          </div>
        </div>

        {/* Retrieval Score Interpretation Notice */}
        <div className="bg-muted/40 p-3 rounded-xl border border-border/60 text-[11px] text-muted-foreground space-y-1">
          <span className="font-semibold text-foreground block">
            About Retrieval Scores
          </span>
          <p className="leading-relaxed">
            Cosine similarity is used as an uncalibrated <strong>retrieval ranking score</strong> (ranging up to 1.0 for exact vector alignment), not a probability or confidence value.
          </p>
        </div>

        {/* Modal Actions */}
        <div className="flex items-center justify-between pt-2">
          <Button variant="outline" size="sm" onClick={onRefreshHealth} className="text-xs">
            Re-check Health Probe
          </Button>
          <Button size="sm" onClick={onClose} className="text-xs">
            Done
          </Button>
        </div>
      </div>
    </div>
  )
}
