import { useState, useRef, useEffect } from "react"
import { Send, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

interface ComposerProps {
  onSend: (text: string) => void
  disabled: boolean
  isPending: boolean
  maxLength?: number
}

export const Composer: React.FC<ComposerProps> = ({
  onSend,
  disabled,
  isPending,
  maxLength = 1000,
}) => {
  const [input, setInput] = useState("")
  const [validationError, setValidationError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Focus textarea on mount and after submission
  useEffect(() => {
    if (!disabled && !isPending) {
      textareaRef.current?.focus()
    }
  }, [disabled, isPending])

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value
    setInput(val)

    if (val.length > maxLength) {
      setValidationError(
        `Question length (${val.length}) exceeds the maximum allowed limit of ${maxLength} characters.`
      )
    } else {
      setValidationError(null)
    }
  }

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    const trimmed = input.trim()

    if (!trimmed) {
      setValidationError("Please enter a question before sending.")
      return
    }

    if (trimmed.length > maxLength) {
      setValidationError(
        `Question exceeds maximum allowed limit of ${maxLength} characters.`
      )
      return
    }

    setValidationError(null)
    onSend(trimmed)
    setInput("")
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Send on Enter (without Shift)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const isTooLong = input.length > maxLength
  const charCount = input.length
  const canSend = input.trim().length > 0 && !isTooLong && !disabled && !isPending

  return (
    <div className="border-t bg-card/60 backdrop-blur px-4 py-3 sm:px-6">
      <div className="max-w-3xl mx-auto space-y-2">
        {/* Validation Error Banner */}
        {validationError && (
          <div className="flex items-center gap-2 text-xs text-destructive bg-destructive/10 border border-destructive/20 p-2 rounded-md">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />
            <span className="leading-tight">{validationError}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="relative flex items-end gap-2">
          <div className="relative flex-1">
            <label htmlFor="chat-composer-textarea" className="sr-only">
              Type your customer support question
            </label>
            <Textarea
              id="chat-composer-textarea"
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              disabled={disabled || isPending}
              placeholder="Ask a question about orders, shipping, returns, or policies... (Shift+Enter for newline)"
              className="min-h-[56px] max-h-36 resize-none pr-12 py-3 text-sm rounded-xl focus-visible:ring-primary/40 leading-relaxed"
              rows={1}
              aria-invalid={isTooLong}
            />

            {/* Character counter */}
            <div
              className={`absolute right-3 bottom-2.5 text-[11px] font-mono select-none ${
                isTooLong
                  ? "text-destructive font-bold"
                  : charCount > maxLength * 0.85
                  ? "text-amber-500 font-medium"
                  : "text-muted-foreground/60"
              }`}
            >
              {charCount}/{maxLength}
            </div>
          </div>

          {/* Send Button */}
          <Button
            type="submit"
            size="icon"
            disabled={!canSend}
            aria-label="Send message"
            className="h-[52px] w-[52px] rounded-xl shrink-0 shadow-sm transition-all"
          >
            <Send className={`h-4 w-4 ${isPending ? "animate-pulse" : ""}`} />
          </Button>
        </form>

        <div className="hidden sm:flex items-center justify-between text-[11px] text-muted-foreground px-1">
          <span>Press <kbd className="px-1 py-0.5 rounded bg-muted font-mono text-[10px]">Enter</kbd> to send, <kbd className="px-1 py-0.5 rounded bg-muted font-mono text-[10px]">Shift+Enter</kbd> for newline</span>
          <span>Answers are retrieved from ChromaDB FAQ embeddings</span>
        </div>
      </div>
    </div>
  )
}
