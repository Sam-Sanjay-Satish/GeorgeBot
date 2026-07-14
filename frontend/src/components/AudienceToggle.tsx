import type { Audience } from '@/types'

interface Props {
  value: Audience
  onChange: (audience: Audience) => void
  disabled?: boolean
}

// Which corpus to search. Undergrad and faculty are disjoint bodies of content
// (student services vs. HR/governance/research-admin); "Both" searches across
// them. Sent with each question and picked up server-side.
const OPTIONS: { value: Audience; label: string }[] = [
  { value: 'undergrad', label: 'Undergrad' },
  { value: 'faculty', label: 'Faculty' },
  { value: 'both', label: 'Both' },
]

export function AudienceToggle({ value, onChange, disabled }: Props) {
  return (
    <div
      role="radiogroup"
      aria-label="Search scope"
      className="inline-flex items-center gap-0.5 rounded-lg border border-border bg-muted/50 p-0.5"
    >
      {OPTIONS.map((opt) => {
        const active = opt.value === value
        return (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={disabled}
            onClick={() => onChange(opt.value)}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${
              active
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}
