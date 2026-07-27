import type { ThinkingMode } from '@/types'

interface Props {
  value: ThinkingMode
  onChange: (mode: ThinkingMode) => void
  disabled?: boolean
}

// On (checked) = "quick": today's single-shot answer path. Off = "default":
// retrieve, answer, then the model self-checks its own answer and does one
// targeted re-fetch if it flagged a gap. Sent with each question as `mode`.
export function ExtendedThinkingToggle({ value, onChange, disabled }: Props) {
  const checked = value === 'quick'

  // Same outer border/padding/rounded-lg shell as AudienceToggle so the two
  // controls line up at identical height when placed side by side.
  return (
    <div className="inline-flex items-center rounded-lg border border-border bg-muted/50 p-0.5">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label="Quick mode"
        disabled={disabled}
        onClick={() => onChange(checked ? 'default' : 'quick')}
        title={
          checked
            ? 'Quick mode: fast, single-pass answers.'
            : 'Double-checks its own answer and re-fetches once if it spots a gap.'
        }
        className="flex items-center gap-2 rounded-md px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
      >
        <span>Quick</span>
        <span
          className={`relative inline-flex h-3.5 w-6 shrink-0 items-center rounded-full transition-colors ${
            checked ? 'bg-primary' : 'bg-border'
          }`}
        >
          <span
            className={`inline-block h-2.5 w-2.5 transform rounded-full bg-background shadow-sm transition-transform ${
              checked ? 'translate-x-3' : 'translate-x-0.5'
            }`}
          />
        </span>
      </button>
    </div>
  )
}
