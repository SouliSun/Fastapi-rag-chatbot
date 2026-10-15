import { useState, useEffect, useCallback } from "react"
import { Header } from "@/components/Header"
import { MessageList } from "@/components/MessageList"
import { Composer } from "@/components/Composer"
import { InfoModal } from "@/components/InfoModal"
import { fetchHealth, sendChatQuery, ApiError } from "@/services/api"
import type { ChatMessage, HealthResponse } from "@/types/api"

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isPending, setIsPending] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthLoading, setHealthLoading] = useState(true)
  const [isInfoOpen, setIsInfoOpen] = useState(false)

  // Periodic health probe
  const checkHealth = useCallback(async () => {
    setHealthLoading(true)
    try {
      const data = await fetchHealth()
      setHealth(data)
    } catch {
      setHealth({
        status: "unhealthy",
        error: "Failed to fetch health probe",
      })
    } finally {
      setHealthLoading(false)
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [checkHealth])

  const handleSendMessage = async (question: string) => {
    if (!question.trim() || isPending) return

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      role: "user",
      content: question,
      timestamp: formatTime(new Date()),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsPending(true)

    try {
      const response = await sendChatQuery(question)

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        role: "assistant",
        content: response.answer,
        timestamp: formatTime(new Date()),
        fallback: response.fallback,
        fallbackReason: response.fallback_reason,
        contexts: response.contexts,
        results: response.results,
        diagnostics: response.diagnostics,
        latencyMs: response.latency_ms,
        promptTokens: response.prompt_tokens,
        completionTokens: response.completion_tokens,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (err: unknown) {
      let errorDetail = "An unexpected error occurred while communicating with the service."
      if (err instanceof ApiError) {
        errorDetail = err.detail
      } else if (err instanceof Error) {
        errorDetail = err.message
      }

      const errorMessage: ChatMessage = {
        id: `assistant-err-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        role: "assistant",
        content: "Sorry, I encountered an issue while processing your question.",
        timestamp: formatTime(new Date()),
        error: errorDetail,
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsPending(false)
    }
  }

  const handleNewChat = () => {
    setMessages([])
  }

  return (
    <div className="flex flex-col h-screen w-full bg-background text-foreground overflow-hidden">
      {/* Top Navigation & Health Bar */}
      <Header
        health={health}
        healthLoading={healthLoading}
        onNewChat={handleNewChat}
        onOpenInfo={() => setIsInfoOpen(true)}
        messageCount={messages.length}
      />

      {/* Main Conversation Scrollable Feed */}
      <main className="flex-1 flex flex-col min-h-0 relative">
        <MessageList
          messages={messages}
          isPending={isPending}
          onSelectPrompt={handleSendMessage}
          onRetry={handleSendMessage}
        />
      </main>

      {/* Bottom Fixed Message Composer */}
      <footer className="shrink-0">
        <Composer
          onSend={handleSendMessage}
          disabled={false}
          isPending={isPending}
          maxLength={1000}
        />
      </footer>

      {/* Architecture & Telemetry Modal */}
      <InfoModal
        isOpen={isInfoOpen}
        onClose={() => setIsInfoOpen(false)}
        health={health}
        onRefreshHealth={checkHealth}
      />
    </div>
  )
}
