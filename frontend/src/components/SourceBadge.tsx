import { Badge } from '@/components/ui/badge'
import type { Source } from '@/types'

const SOURCE_LABELS: Record<Source['source'], string> = {
  uvic_html: 'uvic.ca',
  heat: 'HEAT',
  kuali: 'Catalog',
  uvic_docs: 'Document',
  banner: 'Live',
  rmp: 'Rating',
}

const SOURCE_COLORS: Record<Source['source'], string> = {
  uvic_html: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  heat: 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
  kuali: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  uvic_docs: 'bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200',
  banner: 'bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-200',
  rmp: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
}

interface Props {
  source: Source
}

export function SourceBadge({ source }: Props) {
  return (
    <Badge
      variant="secondary"
      className={`text-xs font-medium ${SOURCE_COLORS[source.source]}`}
    >
      {SOURCE_LABELS[source.source]}
    </Badge>
  )
}
