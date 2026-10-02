import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Analytics, type BeforeSendEvent } from '@vercel/analytics/react'
import './index.css'
import App from './App.tsx'

// Drop analytics events from devices opted out via /no-track.
function skipIfOptedOut(event: BeforeSendEvent): BeforeSendEvent | null {
  try {
    if (localStorage.getItem('va-disable') === 'true') return null
  } catch {
    // storage unavailable — fall through and send
  }
  return event
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
    <Analytics beforeSend={skipIfOptedOut} />
  </StrictMode>,
)
