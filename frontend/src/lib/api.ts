import type { Audience, Message, Source, ThinkingMode } from '@/types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:5001'

/**
 * Is the backend up and serving?
 *
 * Vercel and Railway deploy independently off the same push, and the frontend
 * almost always wins: Vite builds in seconds while the backend rebuilds its
 * image, remounts the Volume, and loads the graph + collections before it binds
 * a port. During that window georgebot.org is live but every question would
 * fail, so the disclaimer gate polls this before letting anyone in.
 *
 * `/health` is deliberately the cheapest route on the backend (async, no Chroma
 * or SQLite touch, chunk count snapshotted at boot), so polling it is free.
 * Any transport error, non-200, or non-"ok" body counts as not-ready — during a
 * redeploy this shows up as a network error or a Railway 502, never a clean
 * false, so failing closed on the catch is the whole point rather than an edge
 * case. Aborts after `timeoutMs` so a connection that hangs mid-deploy can't
 * wedge the poll loop.
 */
export async function checkHealth(timeoutMs = 8000): Promise<boolean> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: controller.signal,
      cache: 'no-store',
    })
    if (!res.ok) return false
    const data = await res.json()
    return data?.status === 'ok'
  } catch {
    return false
  } finally {
    clearTimeout(timer)
  }
}

// Raw source shape returned by the backend (chatbot.format_sources).
interface ApiSource {
  n: number
  url: string
  source: string
  title?: string | null
  course_code?: string | null
  term?: string | null
  historical?: boolean
  document_type?: string | null
  department?: string | null
}

interface ApiChatResponse {
  answer: string
  sources: ApiSource[]
  search_query: string
  n_chunks: number
  error?: string
}

// Backend metadata source -> frontend badge source.
const SOURCE_MAP: Record<string, Source['source']> = {
  webpage: 'uvic_html',
  document: 'uvic_docs',
  heat: 'heat',
  kuali: 'kuali',
  banner: 'banner',
  rmp: 'rmp',
}

function titleFromUrl(url: string): string {
  try {
    const u = new URL(url)
    const seg = u.pathname.split('/').filter(Boolean).pop()
    if (!seg) return u.hostname
    return decodeURIComponent(seg).replace(/[-_]/g, ' ').replace(/\.\w+$/, '')
  } catch {
    return url
  }
}

// Raw page <title>s are breadcrumb-y ("Program Declaration - Engineering -
// Faculty of Engineering and Computer Science"). Split on the breadcrumb
// separators (only when space-padded, so hyphenated words like "Co-operative"
// survive), drop trailing org/faculty/site segments, and keep the two most
// specific parts.
function cleanTitle(raw: string): string {
  const parts = raw
    .split(/\s+[|–—-]\s+/)
    .map((p) => p.trim())
    .filter(Boolean)
  const noise = /^(faculty of\b|university of victoria$|uvic$|home$|index$)/i
  const kept = parts.filter((p) => !noise.test(p))
  const use = (kept.length ? kept : parts).slice(0, 2)
  return use.join(' — ') || raw.trim()
}

function mapSource(s: ApiSource): Source {
  const title = s.course_code
    ? `${s.course_code}${s.term ? ` — ${s.term}` : ''}`
    : s.title
      ? cleanTitle(s.title)
      : titleFromUrl(s.url)
  return {
    url: s.url,
    title,
    source: SOURCE_MAP[s.source] ?? 'uvic_html',
    course: s.course_code ?? undefined,
    term: s.term ?? undefined,
    historical: s.historical ?? undefined,
  }
}

// Convert the local message list into the {role, content} history the API wants,
// dropping any loading/empty placeholder messages.
function toHistory(messages: Message[]): { role: string; content: string }[] {
  return messages
    .filter((m) => m.content && !m.loading)
    .map((m) => ({ role: m.role, content: m.content }))
}

export async function askGeorge(
  question: string,
  priorMessages: Message[],
  audience: Audience = 'undergrad',
): Promise<{ answer: string; sources: Source[] }> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history: toHistory(priorMessages), audience }),
  })
  if (!res.ok) {
    throw new Error(`API error ${res.status}`)
  }
  const data: ApiChatResponse = await res.json()
  if (data.error) throw new Error(data.error)
  return {
    answer: data.answer,
    sources: (data.sources ?? []).map(mapSource),
  }
}

interface StreamHandlers {
  onStatus: (text: string) => void
  onSources: (sources: Source[]) => void
  onToken: (text: string) => void
  onDone: () => void
  onError: (message: string) => void
}

// POST + manually parsed SSE — EventSource doesn't support POST bodies, so we
// read the fetch response stream directly and split on the "event:\ndata:\n\n"
// frame format the backend (/api/chat/stream) emits. Events: status
// (pre-answer phase text), sources, token, done, error.
export async function askGeorgeStream(
  question: string,
  priorMessages: Message[],
  audience: Audience,
  handlers: StreamHandlers,
  mode: ThinkingMode = 'quick',
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      history: toHistory(priorMessages),
      audience,
      mode,
    }),
  })
  if (!res.ok || !res.body) {
    // Rate-limit rejections (429/503) carry a human-readable `detail` from
    // the backend — show that instead of a bare status code.
    let message = `API error ${res.status}`
    try {
      const data = await res.json()
      if (typeof data?.detail === 'string') message = data.detail
    } catch {
      // no JSON body — keep the generic message
    }
    handlers.onError(message)
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sepIndex: number
    while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sepIndex)
      buffer = buffer.slice(sepIndex + 2)

      const eventLine = frame.split('\n').find((l) => l.startsWith('event:'))
      const dataLine = frame.split('\n').find((l) => l.startsWith('data:'))
      if (!eventLine || !dataLine) continue
      const event = eventLine.slice('event:'.length).trim()
      const rawData = dataLine.slice('data:'.length).trim()

      try {
        const data = JSON.parse(rawData)
        if (event === 'status') {
          handlers.onStatus(data as string)
        } else if (event === 'sources') {
          handlers.onSources((data as ApiSource[]).map(mapSource))
        } else if (event === 'token') {
          handlers.onToken(data as string)
        } else if (event === 'done') {
          handlers.onDone()
        } else if (event === 'error') {
          handlers.onError(data as string)
        }
      } catch {
        // malformed frame — skip rather than crash the stream
      }
    }
  }
}
