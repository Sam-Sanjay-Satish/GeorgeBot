import { ExternalLink } from 'lucide-react'
import { Separator } from '@/components/ui/separator'
import { SourceBadge } from './SourceBadge'
import type { Source } from '@/types'

interface Props {
  sources: Source[]
}

export function SourcePanel({ sources }: Props) {
  if (sources.length === 0) return null

  return (
    <div className="mt-3 space-y-2">
      <Separator className="opacity-40" />
      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        Sources
      </p>
      <ul className="space-y-1.5">
        {sources.map((s) => (
          <li key={s.url} className="flex items-start gap-2">
            <SourceBadge source={s} />
            <a
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-foreground/80 hover:text-foreground hover:underline flex items-center gap-1 leading-snug"
            >
              {s.title}
              {s.term && (
                <span className="text-muted-foreground ml-1">({s.term})</span>
              )}
              <ExternalLink className="w-3 h-3 shrink-0 opacity-60" />
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}
