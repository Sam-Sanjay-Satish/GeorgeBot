interface Props {
  value: boolean
  onChange: (value: boolean) => void
  disabled?: boolean
}

// Opt-in "extended thinking" mode. When on, the backend routes the question
// into the multi-mode orchestrator (deeper research, course planning, or
// situational guidance) instead of the single-shot answer path. Slower, so the
// user chooses per session. Sent with each question as `extended_thinking`.
export function ExtendedThinkingToggle({ value, onChange, disabled }: Props) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={value}
      aria-label="Extended thinking"
      disabled={disabled}
      onClick={() => onChange(!value)}
      title="Slower, multi-step answers: deep research, course planning, or help with a specific situation."
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${
        value
          ? 'border-primary/40 bg-primary/10 text-foreground shadow-sm'
          : 'border-border bg-muted/50 text-muted-foreground hover:text-foreground'
      }`}
    >
      <span
        aria-hidden
        className={`inline-block h-1.5 w-1.5 rounded-full transition-colors ${
          value ? 'bg-primary' : 'bg-muted-foreground/50'
        }`}
      />
      Extended thinking
    </button>
  )
}
