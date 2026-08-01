import { useEffect, useState } from 'react'
import { checkHealth } from '@/lib/api'

interface Props {
  onContinue: () => void
}

// How often to re-probe while the backend is down. A Railway rebuild takes
// minutes, so this only needs to be responsive enough that someone staring at
// the page gets let in promptly once it recovers.
const HEALTH_POLL_MS = 5000

type BackendState = 'checking' | 'ready' | 'unavailable'

export function DisclaimerPage({ onContinue }: Props) {
  const [backend, setBackend] = useState<BackendState>('checking')

  useEffect(() => {
    // Guards against a probe that resolves after unmount (or after we've
    // already gone through) writing state on a dead component.
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const poll = async () => {
      const ok = await checkHealth()
      if (cancelled) return
      setBackend(ok ? 'ready' : 'unavailable')
      // Keep polling while it's down so recovery lets people in on its own,
      // with no reload. Chained off the *completed* probe rather than an
      // interval, so a slow response can't stack overlapping requests.
      if (!ok) timer = setTimeout(poll, HEALTH_POLL_MS)
    }
    poll()

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  const ready = backend === 'ready'

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-background">
      <div className="w-full max-w-sm flex flex-col items-center gap-8 px-6">

        {/* Logo */}
        <div className="flex flex-col items-center gap-4">
          <div className="w-20 h-20 rounded-2xl bg-primary flex items-center justify-center text-primary-foreground font-bold text-3xl">
            G
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-semibold">GeorgeBot</h1>
            <p className="text-sm text-muted-foreground mt-1">UVic AI Assistant</p>
          </div>
        </div>

        {/* Intro card */}
        <div className="w-full rounded-2xl border border-border bg-card shadow-sm p-6 flex flex-col gap-4">
          <div className="text-center">
            <p className="text-sm font-medium">Ask anything about UVic</p>
          </div>

          <p className="text-sm text-muted-foreground text-center">
            GeorgeBot answers straight from{' '}
            <a
              href="https://www.uvic.ca"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-foreground"
            >
              uvic.ca
            </a>{' '}
            — course details, policies, live registration data, and even
            niche facts — so you get accurate, up-to-date information without
            digging through the site yourself.
          </p>

          {backend === 'unavailable' && (
            <div
              role="status"
              aria-live="polite"
              className="rounded-xl border border-border bg-muted/50 px-4 py-3 text-center"
            >
              <p className="text-sm font-medium">Down for maintenance</p>
              <p className="text-xs text-muted-foreground mt-1">
                GeorgeBot is updating. This usually takes a few minutes — this
                page will let you in automatically once it's back.
              </p>
            </div>
          )}

          <button
            onClick={onContinue}
            disabled={!ready}
            aria-busy={backend === 'checking'}
            className="w-full flex items-center justify-center rounded-xl bg-primary text-primary-foreground px-4 py-2.5 text-sm font-medium shadow-sm hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:opacity-50"
          >
            {backend === 'checking' ? 'Checking…' : backend === 'ready' ? 'Continue' : 'Unavailable'}
          </button>
        </div>
      </div>
    </div>
  )
}
