import { useEffect, useRef } from "react"
import { Bot, MessageSquare, Sparkles } from "lucide-react"
import { MessageItem } from "@/components/MessageItem"
import type { ChatMessage } from "@/types/api"

interface MessageListProps {
  messages: ChatMessage[]
  isPending: boolean
  onSelectPrompt: (prompt: string) => void
  onRetry: (question: string) => void
}

const STARTER_PROMPTS = [
  "How do I track my order?",
  "What is your return policy?",
  "Do you offer international shipping?",
  "Can I cancel or modify an order after placing it?",
]

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  isPending,
  onSelectPrompt,
  onRetry,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isPending])

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 space-y-6">
      <div className="max-w-3xl mx-auto space-y-6">
        {messages.length === 0 ? (
          /* Empty State */
          <div className="text-center py-8 sm:py-12 space-y-6">
            <div className="h-16 w-16 mx-auto rounded-2xl bg-primary/10 text-primary flex items-center justify-center border border-primary/20 shadow-sm">
              <Bot className="h-8 w-8" />
            </div>

            <div className="max-w-md mx-auto space-y-2">
              <h2 className="text-xl sm:text-2xl font-semibold tracking-tight text-foreground">
                How can we help you today?
              </h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Ask any question about orders, shipping, refunds, and store policies.
                Answers are grounded strictly in our verified FAQ database.
              </p>
            </div>

            {/* Quick Starter Chips */}
            <div className="pt-2 max-w-xl mx-auto">
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center justify-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5 text-primary" />
                Frequently Asked Inquiries
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-left">
                {STARTER_PROMPTS.map((prompt, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => onSelectPrompt(prompt)}
                    className="p-3 rounded-xl border bg-card hover:bg-accent/60 hover:border-primary/40 transition-all text-xs text-foreground group flex items-start gap-2.5 shadow-sm"
                  >
                    <MessageSquare className="h-4 w-4 text-muted-foreground group-hover:text-primary shrink-0 mt-0.5 transition-colors" />
                    <span className="font-medium leading-relaxed">{prompt}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Render Messages */
          messages.map((msg) => (
            <MessageItem key={msg.id} message={msg} onRetry={onRetry} />
          ))
        )}

        {/* Pending typing indicator */}
        {isPending && (
          <div className="flex items-start gap-2.5 sm:gap-3">
            <div className="h-8 w-8 rounded-full bg-muted text-foreground flex items-center justify-center shrink-0 border border-border mt-0.5">
              <Bot className="h-4 w-4 text-primary animate-pulse" />
            </div>
            <div className="bg-card border border-border/80 px-4 py-3 rounded-2xl rounded-tl-sm shadow-sm text-sm space-y-1.5">
              <div className="flex items-center gap-1.5 text-muted-foreground text-xs">
                <div className="flex gap-1 items-center">
                  <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce [animation-delay:-0.3s]" />
                  <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce [animation-delay:-0.15s]" />
                  <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce" />
                </div>
                <span className="text-xs font-medium ml-1">
                  Searching FAQ index & generating answer...
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
