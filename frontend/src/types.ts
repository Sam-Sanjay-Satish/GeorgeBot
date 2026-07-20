export type MessageRole = 'user' | 'assistant'

// Which v2.2 corpus the backend should search for a question.
export type Audience = 'undergrad' | 'faculty' | 'both'

// "quick" -- today's flat retrieve->answer pipeline. "default" -- a planner
// call also decides simple vs. one of three extended-thinking plans
// (deep research, course planning, situational guidance) and auto-dispatches
// with no further user input. Replaces the old boolean extended-thinking toggle.
export type ThinkingMode = 'quick' | 'default'

export type Theme = 'light' | 'dark'

// Course-planning slots the backend resolved on a previous turn, echoed back
// unchanged on every subsequent request so it doesn't re-derive (and degrade)
// them from the raw conversation each time — e.g. a program the user already
// picked from a disambiguation list stays picked. Treat this as an opaque
// blob: the client never reads or edits it, it only stores and returns it.
// The backend re-validates and tolerates it being absent.
export interface PlanningState {
  program: { pid: string; title: string; credential: string } | null
  term_season: string | null
  term_year: number | null
  completed_courses: string[]
  completed_known: boolean
  constraints: {
    include_courses: string[]
    exclude_courses: string[]
    soft_preferences: { type: string; value: unknown }[]
  }
  non_course_resolved: Record<string, { satisfied: boolean; reason: string | null }>
  pending_clarify: {
    kind: string
    options: { pid: string; title: string; credential: string }[]
  } | null
}

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
  // Which extended-thinking plan produced this answer (e.g. "Course Planner",
  // "Deep Research", "Guidance") — set from the "mode" SSE event, arrives
  // before any tokens, and persists on the finished message. Undefined for
  // ordinary single-pass ("quick"/simple) answers.
  mode?: string
}
