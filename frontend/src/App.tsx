import { useState, useRef, useEffect } from 'react'
import { MessageBubble } from './components/MessageBubble'
import { ChatInput } from './components/ChatInput'
import { AudienceToggle } from './components/AudienceToggle'
import { ExtendedThinkingToggle } from './components/ExtendedThinkingToggle'
import { ThemeToggle } from './components/ThemeToggle'
import { DisclaimerPage } from './components/DisclaimerPage'
import { AccountMenu } from './components/AccountMenu'
import { askGeorgeStream } from './lib/api'
import type { Audience, Message, Source, Theme, ThinkingMode } from './types'

// Tokens arrive from the backend in bursts (network + generation timing, not
// a steady per-character rate), which reads as jittery/too-fast on screen.
// Reveal characters from a local queue at a fixed pace instead of dumping
// each token straight to the DOM the instant it arrives — this only affects
// perceived typing speed, not actual backend response time.
const REVEAL_CHARS_PER_TICK = 3
const REVEAL_TICK_MS = 16

export default function App() {
  const [acknowledged, setAcknowledged] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [audience, setAudience] = useState<Audience>('undergrad')
  const [thinkingMode, setThinkingMode] = useState<ThinkingMode>('default')
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('theme')
    if (saved === 'light' || saved === 'dark') return saved
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  // Whether we keep the view stuck to the bottom as new content streams in.
  // Any upward movement turns this off *immediately* (intent), and it only
  // turns back on once the user returns to the very bottom — otherwise the
  // 16ms re-snap traps them, since a single wheel notch or short drag is
  // smaller than any position threshold and gets snapped back before it
  // registers.
  const followRef = useRef(true)
  // Last scroll position we've seen, so `handleScroll` can tell direction.
  // The auto-scroll effect writes this too, so its own (downward) jump is
  // never mistaken for the user moving.
  const lastTopRef = useRef(0)
  // A finger is currently on the list. Writing scrollTop mid-drag cancels
  // Android's touch scroll outright, so auto-scroll stands down while true.
  const touchingRef = useRef(false)

  function handleWheel(e: React.WheelEvent) {
    if (e.deltaY < 0) followRef.current = false // scrolling up = stop following
  }

  function handleTouchStart() {
    touchingRef.current = true
  }

  function handleTouchEnd() {
    touchingRef.current = false
  }

  function handleScroll() {
    const el = scrollRef.current
    if (!el) return
    const top = el.scrollTop
    const prevTop = lastTopRef.current
    lastTopRef.current = top
    const dist = el.scrollHeight - top - el.clientHeight
    if (dist < 4) {
      followRef.current = true // returned to the bottom → resume
      return
    }
    // Touch scrolling fires no wheel event, so direction has to come from the
    // scroll position itself — without this, a short upward drag on Android
    // left `dist` under the 120px threshold and got re-snapped to the bottom
    // every ~16ms, making the answer impossible to scroll back through.
    if (top < prevTop - 1) followRef.current = false
    else if (dist > 120) followRef.current = false // clearly scrolled away (scrollbar/keys)
  }

  useEffect(() => {
    // Instant jump (not smooth): during the typewriter reveal this runs every
    // ~16ms, and overlapping smooth-scroll animations keep dragging toward the
    // bottom even after the user scrolls up, fighting them. A direct scrollTop
    // set has no in-flight animation to fight.
    const el = scrollRef.current
    if (el && followRef.current && !touchingRef.current) {
      el.scrollTop = el.scrollHeight
      lastTopRef.current = el.scrollTop
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
      }, thinkingMode)
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

  if (!acknowledged) {
    return <DisclaimerPage onContinue={() => setAcknowledged(true)} />
  }

  // No conversation yet → the composer is the hero: centered in the viewport
  // with the greeting sitting directly above it, rather than pinned to the
  // bottom edge of an empty page.
  const empty = messages.length === 0

  return (
    <div className="flex flex-col h-dvh bg-background overflow-hidden">
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
        <AccountMenu
          onReset={() => {
            setAcknowledged(false)
            setMessages([])
          }}
        />
      </header>

      {/* Messages */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        onWheel={handleWheel}
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
        onTouchCancel={handleTouchEnd}
        className="flex-1 overflow-y-auto overscroll-contain px-4 py-6"
      >
        {empty ? (
          // Bottom-aligned so the greeting hugs the composer below it; the
          // spacer after the composer is what lifts the pair to the middle.
          <div className="flex flex-col items-center justify-end h-full text-center gap-3 text-muted-foreground pb-8">
            <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center text-2xl font-bold text-foreground">
              G
            </div>
            <p className="text-lg font-medium text-foreground">Ask GeorgeBot anything about UVic</p>
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

      {/* Input — same element in both layouts (and at the same child index, so
          the trailing spacer appearing/disappearing never remounts ChatInput
          and drops focus on the first send). */}
      {/* `mb-24` while empty pulls the textbox up to look centered on the
          *screen*, not just within the area under the header. Equal flex
          spacers alone leave it sitting low twice over: the header eats ~65px
          off the top of the viewport, and inside this block the textbox has
          the toggles above it but only the thin disclaimer below. Bottom
          margin shifts the block up by half its value, correcting both. */}
      <div className={`px-4 py-4 shrink-0 ${empty ? 'mb-24' : 'border-t border-border'}`}>
        <div className="max-w-2xl mx-auto">
          <div className="mb-4 flex flex-wrap items-center justify-center gap-2">
            <AudienceToggle value={audience} onChange={setAudience} disabled={loading} />
            <ExtendedThinkingToggle
              value={thinkingMode}
              onChange={setThinkingMode}
              disabled={loading}
            />
          </div>
          <ChatInput onSend={handleSend} disabled={loading} />
          <p className="text-xs text-center text-muted-foreground mt-2">
            GeorgeBot can make mistakes. Verify important info with UVic directly.
          </p>
        </div>
      </div>

      {/* Balances the message area above, centering the composer while empty. */}
      {empty && <div className="flex-1 min-h-0" aria-hidden />}
    </div>
  )
}
