import { useState, useRef, useEffect } from 'react'
import { MessageBubble } from './components/MessageBubble'
import { ChatInput } from './components/ChatInput'
import { AudienceToggle } from './components/AudienceToggle'
import { ThemeToggle } from './components/ThemeToggle'
import { LoginPage } from './components/LoginPage'
import { AccountMenu } from './components/AccountMenu'
import { askGeorgeStream } from './lib/api'
import type { Audience, Message, Source, Theme } from './types'

// Tokens arrive from the backend in bursts (network + generation timing, not
// a steady per-character rate), which reads as jittery/too-fast on screen.
// Reveal characters from a local queue at a fixed pace instead of dumping
// each token straight to the DOM the instant it arrives — this only affects
// perceived typing speed, not actual backend response time.
const REVEAL_CHARS_PER_TICK = 3
const REVEAL_TICK_MS = 16

export default function App() {
  const [loggedIn, setLoggedIn] = useState(false)
  const [userName, setUserName] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [audience, setAudience] = useState<Audience>('undergrad')
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('theme')
    if (saved === 'light' || saved === 'dark') return saved
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  // Whether the user is stuck to the bottom. Only auto-scroll on new content
  // when true, so scrolling up mid-stream isn't yanked back down.
  const pinnedRef = useRef(true)

  function handleScroll() {
    const el = scrollRef.current
    if (!el) return
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  useEffect(() => {
    if (pinnedRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    localStorage.setItem('theme', theme)
  }, [theme])

  async function handleSend(text: string) {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
    }
    const loadingId = crypto.randomUUID()
    const loadingMsg: Message = {
      id: loadingId,
      role: 'assistant',
      content: '',
      loading: true,
    }

    // History the backend should see = everything before this new question.
    const priorMessages = messages
    setMessages((prev) => [...prev, userMsg, loadingMsg])
    setLoading(true)

    // `revealed` is what's on screen; `pending` is arrived-but-not-yet-shown
    // text, drained onto `revealed` at a fixed pace by the interval below.
    let revealed = ''
    let pending = ''
    let streamEnded = false
    let flushTimer: ReturnType<typeof setInterval> | null = null
    // Sources arrive up-front from the backend, but we hold them back and only
    // attach them once the answer has finished revealing — so they appear
    // together with the completed message rather than before it's done typing.
    let bufferedSources: Source[] | null = null

    function updateContent() {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId ? { ...m, loading: false, content: revealed } : m
        )
      )
    }

    // Streaming + reveal fully done: stop the timer, commit the final text, and
    // now reveal the buffered sources.
    function finish() {
      if (flushTimer) {
        clearInterval(flushTimer)
        flushTimer = null
      }
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? { ...m, loading: false, content: revealed, sources: bufferedSources ?? m.sources }
            : m
        )
      )
      setLoading(false)
    }

    function startFlushing() {
      if (flushTimer) return
      flushTimer = setInterval(() => {
        if (pending.length > 0) {
          revealed += pending.slice(0, REVEAL_CHARS_PER_TICK)
          pending = pending.slice(REVEAL_CHARS_PER_TICK)
          updateContent()
        } else if (streamEnded) {
          finish()
        }
      }, REVEAL_TICK_MS)
    }

    try {
      await askGeorgeStream(text, priorMessages, audience, {
        onStatus: (status) => {
          // Transient pre-answer phase line; overwritten by later status events
          // and cleared once the first token arrives (see onToken).
          setMessages((prev) =>
            prev.map((m) => (m.id === loadingId ? { ...m, status } : m))
          )
        },
        onSources: (sources) => {
          // Buffer only — finish() attaches these once the reveal completes.
          bufferedSources = sources
        },
        onToken: (token) => {
          // First token: drop the status line so it doesn't linger beside the
          // streaming answer.
          if (revealed === '' && pending === '') {
            setMessages((prev) =>
              prev.map((m) => (m.id === loadingId ? { ...m, status: undefined } : m))
            )
          }
          pending += token
          startFlushing()
        },
        onDone: () => {
          streamEnded = true
          if (pending.length === 0) finish()
        },
        onError: (message) => {
          streamEnded = true
          if (pending.length === 0) {
            if (flushTimer) {
              clearInterval(flushTimer)
              flushTimer = null
            }
            setMessages((prev) =>
              prev.map((m) =>
                m.id === loadingId
                  ? {
                      ...m,
                      loading: false,
                      content:
                        revealed ||
                        `Sorry — I couldn't reach the backend. ${message}`.trim(),
                    }
                  : m
              )
            )
            setLoading(false)
          }
        },
      })
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? {
                ...m,
                loading: false,
                content: `Sorry — I couldn't reach the backend. ${
                  err instanceof Error ? err.message : ''
                }`.trim(),
              }
            : m
        )
      )
      setLoading(false)
    }
  }

  if (!loggedIn) {
    return <LoginPage onLogin={(name) => { setUserName(name); setLoggedIn(true) }} />
  }

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      {/* Header */}
      <header className="border-b border-border px-6 py-4 flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold text-sm">
          G
        </div>
        <div className="flex-1">
          <h1 className="text-base font-semibold leading-none">GeorgeBot</h1>
          <p className="text-xs text-muted-foreground mt-0.5">UVic AI Assistant</p>
        </div>
        <ThemeToggle theme={theme} onToggle={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} />
        <AccountMenu name={userName} onLogout={() => { setLoggedIn(false); setUserName('') }} />
      </header>

      {/* Messages */}
      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3 text-muted-foreground pt-24">
            <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center text-2xl font-bold text-foreground">
              G
            </div>
            <p className="text-base font-medium text-foreground">Ask GeorgeBot anything about UVic</p>
            <p className="text-sm max-w-xs">
              Courses, registration, campus life, and anything else — all sourced from uvic.ca.
            </p>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto space-y-6">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-border px-4 py-4 shrink-0">
        <div className="max-w-2xl mx-auto">
          <div className="mb-4 flex justify-center">
            <AudienceToggle value={audience} onChange={setAudience} disabled={loading} />
          </div>
          <ChatInput onSend={handleSend} disabled={loading} />
          <p className="text-xs text-center text-muted-foreground mt-2">
            GeorgeBot can make mistakes. Verify important info with UVic directly.
          </p>
        </div>
      </div>
    </div>
  )
}
