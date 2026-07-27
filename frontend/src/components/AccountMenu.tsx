import { useEffect, useRef, useState } from 'react'
import { RotateCcw, User } from 'lucide-react'

interface Props {
  onReset: () => void
}

export function AccountMenu({ onReset }: Props) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-8 h-8 rounded-full bg-muted flex items-center justify-center hover:bg-accent transition-colors"
        aria-label="Account menu"
      >
        <User className="w-4 h-4 text-muted-foreground" />
      </button>

      {open && (
        <div className="absolute right-0 top-10 w-56 rounded-xl border border-border bg-card shadow-md z-50 overflow-hidden">
          <div className="px-4 py-3 border-b border-border">
            <p className="text-xs text-muted-foreground">
              For educational use only — always verify important info on{' '}
              <a
                href="https://www.uvic.ca"
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-2 hover:text-foreground"
              >
                uvic.ca
              </a>
              .
            </p>
          </div>
          <button
            onClick={() => { setOpen(false); onReset() }}
            className="w-full flex items-center gap-2 px-4 py-2.5 text-sm hover:bg-muted transition-colors text-left"
          >
            <RotateCcw className="w-4 h-4 text-muted-foreground" />
            Start over
          </button>
        </div>
      )}
    </div>
  )
}
