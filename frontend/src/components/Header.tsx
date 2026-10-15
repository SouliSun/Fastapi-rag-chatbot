import { Bot, RefreshCw, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { HealthResponse } from "@/types/api"

interface HeaderProps {
  health: HealthResponse | null
  healthLoading: boolean
  onNewChat: () => void
  onOpenInfo: () => void
  messageCount: number
}

export const Header: React.FC<HeaderProps> = ({
  health,
  healthLoading,
  onNewChat,
  onOpenInfo,
  messageCount,
}) => {
  const isHealthy = health?.status === "healthy" && health.test_retrieval_ok

  return (
    <header className="border-b bg-card/80 backdrop-blur sticky top-0 z-10 px-4 py-3 sm:px-6">
      <div className="max-w-5xl mx-auto flex items-center justify-between gap-4">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="h-10 w-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center shrink-0 border border-primary/20">
            <Bot className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-semibold tracking-tight text-foreground truncate">
                Customer Support FAQ
              </h1>
              <span className="hidden sm:inline-block text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded font-mono">
                RAG Core v2.0
              </span>
            </div>
            <p className="text-xs text-muted-foreground truncate">
              Answers grounded exclusively in verified company FAQs
            </p>
          </div>
        </div>

        {/* Status Indicator & Controls */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Health Status Indicator */}
          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-background/50 cursor-pointer hover:bg-accent transition-colors"
            onClick={onOpenInfo}
            title={
              isHealthy
                ? `RAG System Online: ${health.indexed_faqs} FAQs indexed in ChromaDB`
                : health?.error || "Cannot connect to backend API (GET /health)"
            }
            role="status"
            aria-live="polite"
          >
            {healthLoading ? (
              <>
                <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="text-muted-foreground hidden md:inline">Connecting...</span>
              </>
            ) : isHealthy ? (
              <>
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-emerald-700 dark:text-emerald-400 font-medium">
                  Online
                </span>
                {health?.indexed_faqs !== undefined && health.indexed_faqs !== null && (
                  <span className="text-muted-foreground hidden sm:inline text-[11px]">
                    ({health.indexed_faqs} FAQs)
                  </span>
                )}
              </>
            ) : (
              <>
                <span className="h-2 w-2 rounded-full bg-destructive" />
                <span className="text-destructive font-medium">Offline</span>
              </>
            )}
          </div>

          {/* Info Modal Trigger */}
          <Button
            variant="ghost"
            size="icon"
            onClick={onOpenInfo}
            aria-label="View system architecture and health info"
            title="System & Architecture Information"
            className="h-9 w-9 text-muted-foreground hover:text-foreground"
          >
            <HelpCircle className="h-4 w-4" />
          </Button>

          {/* New Conversation Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={onNewChat}
            disabled={messageCount === 0}
            className="gap-1.5 text-xs h-9 px-3"
            aria-label="Start new conversation"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New Chat</span>
          </Button>
        </div>
      </div>
    </header>
  )
}
