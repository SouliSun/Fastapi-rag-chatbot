/**
 * API service for communicating with the FAQ RAG FastAPI backend.
 */

import type { ChatRequest, ChatResponse, HealthResponse } from '@/types/api'

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/**
 * Fetch service and RAG index health readiness.
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const url = `${API_BASE_URL}/health`
  try {
    const res = await fetch(url, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    })

    if (!res.ok) {
      let errorMsg = `Server returned HTTP ${res.status}`
      try {
        const body = await res.json()
        if (body.error) errorMsg = body.error
        else if (body.detail) errorMsg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      } catch {
        // use fallback message
      }
      return {
        status: 'unhealthy',
        error: errorMsg,
      }
    }

    const data: HealthResponse = await res.json()
    return data
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Network connection failed'
    return {
      status: 'unhealthy',
      error: `Cannot reach backend at ${API_BASE_URL} (${msg})`,
    }
  }
}

/**
 * Send a user question to the RAG pipeline.
 */
export async function sendChatQuery(
  question: string,
  options?: { k?: number; threshold?: number }
): Promise<ChatResponse> {
  const url = `${API_BASE_URL}/chat`
  const payload: ChatRequest = {
    question: question.trim(),
    ...(options?.k ? { k: options.k } : {}),
    ...(options?.threshold !== undefined ? { threshold: options.threshold } : {}),
  }

  let res: Response
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
    })
  } catch (networkErr: unknown) {
    const msg = networkErr instanceof Error ? networkErr.message : 'Network error'
    throw new ApiError(
      0,
      `Unable to reach RAG backend at ${API_BASE_URL}. Ensure uvicorn is running: ${msg}`
    )
  }

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status} error`
    try {
      const errBody = await res.json()
      if (errBody.detail) {
        if (typeof errBody.detail === 'string') {
          errorDetail = errBody.detail
        } else if (Array.isArray(errBody.detail)) {
          // Pydantic validation error format
          errorDetail = errBody.detail.map((e: { msg?: string }) => e.msg || 'Invalid field').join('; ')
        } else {
          errorDetail = JSON.stringify(errBody.detail)
        }
      } else if (errBody.error) {
        errorDetail = errBody.error
      }
    } catch {
      // Body was not JSON
    }

    throw new ApiError(res.status, errorDetail)
  }

  try {
    const data: ChatResponse = await res.json()
    return data
  } catch {
    throw new ApiError(
      res.status,
      'Received malformed response from RAG service.'
    )
  }
}

export { API_BASE_URL }
