/**
 * TypeScript definitions matching FastAPI backend models in api.py
 */

export interface RetrievedFAQ {
  id: string
  question: string
  answer: string
  score: number // Calculated cosine similarity [-1.0 to 1.0]
  distance: number // Raw vector distance from ChromaDB
}

export interface RetrievalDiagnostics {
  query_length: number
  requested_k: number
  effective_k: number
  total_candidates: number
  accepted_candidates: number
  top_score: number | null
  latency_ms: number
  threshold_applied: number
  is_accepted: boolean
  error?: string | null
}

export interface ChatRequest {
  question: string
  k?: number
  threshold?: number
}

export interface ChatResponse {
  answer: string
  fallback: boolean
  fallback_reason?: string | null
  contexts: RetrievedFAQ[]
  results: RetrievedFAQ[]
  latency_ms: number
  prompt_tokens?: number | null
  completion_tokens?: number | null
  diagnostics?: RetrievalDiagnostics | null
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy' | string
  indexed_faqs?: number | null
  database_path?: string | null
  collection_name?: string | null
  embedding_model?: string | null
  test_retrieval_ok?: boolean | null
  error?: string | null
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  fallback?: boolean
  fallbackReason?: string | null
  contexts?: RetrievedFAQ[]
  results?: RetrievedFAQ[]
  diagnostics?: RetrievalDiagnostics | null
  latencyMs?: number
  promptTokens?: number | null
  completionTokens?: number | null
  error?: string | null
  isPending?: boolean
}

