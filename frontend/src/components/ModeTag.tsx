import { Badge } from '@/components/ui/badge'

// Colors keyed by label text (backend sends the display label directly via
// thinking.PLAN_LABELS) rather than the internal plan id.
const MODE_COLORS: Record<string, string> = {
  'Course Planner': 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200',
  'Deep Research': 'bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200',
  Guidance: 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200',
}

interface Props {
  label: string
}

export function ModeTag({ label }: Props) {
  return (
    <Badge
      variant="secondary"
      className={`text-xs font-medium ${MODE_COLORS[label] ?? 'bg-muted text-muted-foreground'}`}
    >
      {label}
    </Badge>
  )
}
