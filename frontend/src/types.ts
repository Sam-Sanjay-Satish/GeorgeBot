export type MessageRole = 'user' | 'assistant'

export interface Source {
  url: string
  title: string
  source: 'uvic_html' | 'heat' | 'kuali' | 'uvic_docs'
  course?: string
  term?: string
  historical?: boolean
}

export interface Message {
  id: string
  role: MessageRole
  content: string
  sources?: Source[]
  loading?: boolean
}
