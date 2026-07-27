export type MessageRole = 'user' | 'assistant'

// Which v2.2 corpus the backend should search for a question.
export type Audience = 'undergrad' | 'faculty' | 'both'

// "quick" -- flat retrieve->answer pipeline. "default" -- retrieve->answer,
// then the model self-verifies and does one targeted re-fetch if it flagged
// a gap. Replaces the old boolean extended-thinking toggle.
export type ThinkingMode = 'quick' | 'default'

export type Theme = 'light' | 'dark'

export interface Source {
  url: string
  title: string
  source: 'uvic_html' | 'heat' | 'kuali' | 'uvic_docs' | 'banner' | 'rmp'
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
  // Transient pre-answer phase text (e.g. "Looking up CSC 225…"), shown only
  // while `loading` and cleared once answer tokens begin.
  status?: string
}
