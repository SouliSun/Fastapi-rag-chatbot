import { useState } from "react"
import { ChevronDown, ChevronUp, FileText } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { RetrievedFAQ, RetrievalDiagnostics } from "@/types/api"

interface SourcesDrawerProps {
  contexts?: RetrievedFAQ[]
  results?: RetrievedFAQ[]
  diagnostics?: RetrievalDiagnostics | null
  latencyMs?: number
  promptTokens?: number | null
  completionTokens?: number | null
}

export const SourcesDrawer: React.FC<SourcesDrawerProps> = ({
  contexts = [],
  results = [],
  diagnostics,
  latencyMs,
  promptTokens,
  completionTokens,
}) => {
  const [isOpen, setIsOpen] = useState(false)

  // Use filtered contexts if available, or fall back to results
  const itemsToDisplay = contexts.length > 0 ? contexts : results
  if (itemsToDisplay.length === 0 && !diagnostics) {
    return null
  }

  return (
    <div className="mt-3 border-t border-border/60 pt-2 text-xs">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="inline-flex items-center gap-1.5 text-muted-foreground hover:text-foreground font-medium py-1 px-2 rounded-md hover:bg-muted/60 transition-colors"
        aria-expanded={isOpen}
      >
        <FileText className="h-3.5 w-3.5 text-primary" />
        <span>
          {isOpen ? "Hide" : "Show"} Retrieved Sources & Diagnostics ({itemsToDisplay.length})
        </span>
        {isOpen ? (
          <ChevronUp className="h-3.5 w-3.5" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5" />
        )}
      </button>

      {isOpen && (
        <div className="mt-2.5 space-y-3 rounded-lg border bg-muted/30 p-3">
          {/* Diagnostics summary row */}
          {diagnostics && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pb-2 border-b border-border/50 text-[11px] text-muted-foreground">
              <div>
                <span className="block text-foreground font-medium">
                  {diagnostics.threshold_applied}
                </span>
                <span>Threshold Applied</span>
              </div>
              <div>
                <span className="block text-foreground font-medium">
                  {diagnostics.accepted_candidates} / {diagnostics.total_candidates}
                </span>
                <span>Candidates Accepted</span>
              </div>
              <div>
                <span className="block text-foreground font-medium">
                  {diagnostics.latency_ms} ms
                </span>
                <span>Vector Retrieval Time</span>
              </div>
              <div>
                <span className="block text-foreground font-medium">
                  {latencyMs ? `${latencyMs} ms` : "-"}
                </span>
                <span>Total Response Latency</span>
              </div>
            </div>
          )}

          {/* Individual FAQ context cards */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span className="font-semibold text-foreground uppercase tracking-wider text-[10px]">
                Grounding Evidence (Top Ranked FAQs)
              </span>
              <span className="text-[10px]">
                Cosine Similarity: 1.0 = exact match
              </span>
            </div>

            {itemsToDisplay.map((faq, idx) => (
              <div
                key={faq.id || idx}
                className="rounded-md border bg-card p-2.5 space-y-1.5 shadow-sm"
              >
                <div className="flex flex-wrap items-center justify-between gap-1.5">
                  <span className="font-medium text-foreground text-xs">
                    Q: {faq.question}
                  </span>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {/* CRITICAL: Explicitly labeled as 'Retrieval Score', NEVER confidence/probability */}
                    <Badge variant="secondary" className="font-mono text-[11px] px-1.5 py-0">
                      Retrieval Score: {faq.score.toFixed(3)}
                    </Badge>
                    <span className="text-[10px] text-muted-foreground font-mono">
                      dist: {faq.distance.toFixed(3)}
                    </span>
                  </div>
                </div>

                <p className="text-muted-foreground text-xs leading-relaxed bg-muted/40 p-2 rounded border border-border/40">
                  {faq.answer}
                </p>

                <div className="flex items-center justify-between text-[10px] text-muted-foreground font-mono">
                  <span>ID: {faq.id}</span>
                  {idx === 0 && (
                    <span className="text-primary font-sans font-medium">Top Candidate</span>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Tokens metadata if available */}
          {(promptTokens || completionTokens) && (
            <div className="pt-1 flex items-center justify-end gap-3 text-[10px] text-muted-foreground font-mono border-t border-border/40">
              {promptTokens && <span>Prompt: {promptTokens} tokens</span>}
              {completionTokens && <span>Completion: {completionTokens} tokens</span>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
