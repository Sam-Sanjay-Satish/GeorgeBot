import type { Message } from './types'

export const MOCK_MESSAGES: Message[] = [
  {
    id: '1',
    role: 'user',
    content: 'What are the prerequisites for CSC 225?',
  },
  {
    id: '2',
    role: 'assistant',
    content:
      'CSC 225 (Algorithms and Data Structures I) requires completion of CSC 110 or CSC 115 with a minimum grade of C+, and MATH 100 or MATH 109 or MATH 110 or MATH 151. You must also be in a program that includes CSC 225.',
    sources: [
      {
        url: 'https://uvic.kuali.co/api/v1/catalog/course/5d9ccc4eab7506001ae4c225/csc225',
        title: 'CSC 225 — Algorithms and Data Structures I',
        source: 'kuali',
        course: 'CSC225',
      },
      {
        url: 'https://heat.csc.uvic.ca/coview/course/2024091/CSC225',
        title: 'CSC 225 Course Outline — Fall 2024',
        source: 'heat',
        course: 'CSC225',
        term: 'Fall 2024',
      },
    ],
  },
  {
    id: '3',
    role: 'user',
    content: 'When is the last day to drop a course without academic penalty?',
  },
  {
    id: '4',
    role: 'assistant',
    content:
      'The last day to drop a course without academic penalty (i.e. without a "W" appearing on your transcript) is typically in the third week of classes for a regular semester. For the exact date, check the UVic Academic Calendar for the current session, as dates vary by term.',
    sources: [
      {
        url: 'https://www.uvic.ca/students/registration/dates-deadlines/',
        title: 'Registration Dates & Deadlines',
        source: 'uvic_html',
      },
    ],
  },
]
