import { Bot, User, AlertTriangle, RotateCcw, Clock, ShieldCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import { SourcesDrawer } from "@/components/SourcesDrawer"
import type { ChatMessage } from "@/types/api"

interface MessageItemProps {
  message: ChatMessage
  onRetry?: (question: string) => void
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, onRetry }) => {
  const isUser = message.role === "user"

  if (isUser) {
    return (
      <div className="flex items-start justify-end gap-2.5 sm:gap-3 group">
        <div className="flex flex-col items-end max-w-[85%] sm:max-w-[75%]">
          <div className="bg-primary text-primary-foreground px-4 py-2.5 rounded-2xl rounded-tr-sm shadow-sm text-sm sm:text-[15px] leading-relaxed break-words whitespace-pre-wrap">
            {message.content}
          </div>
          <span className="text-[11px] text-muted-foreground mt-1 px-1 flex items-center gap-1">
            <Clock className="h-3 w-3 inline" />
            {message.timestamp}
          </span>
        </div>
        <div className="h-8 w-8 rounded-full bg-primary/20 text-primary flex items-center justify-center shrink-0 text-xs font-semibold border border-primary/30 mt-0.5">
          <User className="h-4 w-4" />
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex items-start gap-2.5 sm:gap-3 group">
      <div className="h-8 w-8 rounded-full bg-muted text-foreground flex items-center justify-center shrink-0 text-xs font-semibold border border-border mt-0.5">
        <Bot className="h-4 w-4 text-primary" />
      </div>

      <div className="flex flex-col items-start max-w-[88%] sm:max-w-[80%] min-w-0">
        <div className="bg-card border border-border/80 px-4 py-3 rounded-2xl rounded-tl-sm shadow-sm text-sm sm:text-[15px] leading-relaxed text-foreground space-y-2 w-full">
          {/* Fallback Alert Banner */}
          {message.fallback && (
            <div className="flex items-start gap-2 bg-amber-500/10 border border-amber-500/30 text-amber-800 dark:text-amber-300 text-xs px-2.5 py-1.5 rounded-md">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-amber-600 dark:text-amber-400" />
              <div className="space-y-0.5">
                <span className="font-semibold block">Polite Fallback Response</span>
                <span className="text-[11px] opacity-90 block">
                  {message.fallbackReason || "No retrieved FAQ met the similarity threshold."}
                </span>
              </div>
            </div>
          )}

          {/* Inline Error State */}
          {message.error ? (
            <div className="space-y-2">
              <div className="bg-destructive/10 border border-destructive/30 text-destructive text-xs p-2.5 rounded-md space-y-1">
                <span className="font-semibold block">Failed to get response</span>
                <p className="text-[11px] leading-relaxed opacity-90">{message.error}</p>
              </div>
              {onRetry && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onRetry(message.content)}
                  className="gap-1.5 text-xs h-7 px-2.5"
                >
                  <RotateCcw className="h-3 w-3" />
                  Retry Question
                </Button>
              )}
            </div>
          ) : (
            /* Answer Text */
            <div className="whitespace-pre-wrap break-words text-foreground font-normal">
              {message.content}
            </div>
          )}

          {/* Retrieved Sources & Telemetry */}
          {((message.contexts && message.contexts.length > 0) ||
            (message.results && message.results.length > 0) ||
            message.diagnostics) && (
            <SourcesDrawer
              contexts={message.contexts}
              results={message.results}
              diagnostics={message.diagnostics}
              latencyMs={message.latencyMs}
              promptTokens={message.promptTokens}
              completionTokens={message.completionTokens}
            />
          )}
        </div>

        {/* Timestamp and Grounding indicator */}
        <div className="flex items-center gap-2 mt-1 px-1 text-[11px] text-muted-foreground">
          <span>{message.timestamp}</span>
          {!message.fallback && !message.error && (
            <span className="inline-flex items-center gap-0.5 text-emerald-600 dark:text-emerald-400">
              <ShieldCheck className="h-3 w-3" />
              FAQ Grounded
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
