import { useEffect, useRef } from 'react'

interface Props {
  open: boolean
  title: string
  description: string
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: () => void
  onCancel: () => void
}

// A small confirm modal in the same visual language as DisclaimerPage's card
// and AccountMenu's dropdown (rounded-2xl, border-border, bg-card, shadow) —
// there's no dialog primitive in components/ui, and pulling one in for a single
// two-button prompt would be more surface than this needs.
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
}: Props) {
  const confirmRef = useRef<HTMLButtonElement>(null)

  // Focus the primary action so the dialog is operable from the keyboard
  // immediately. Keyed on `open` alone — folding this into the key-handler
  // effect below would re-steal focus on every parent render, since callers
  // pass inline handlers.
  useEffect(() => {
    if (open) confirmRef.current?.focus()
  }, [open])

  useEffect(() => {
    if (!open) return
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onCancel()
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [open, onCancel])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm"
      // Backdrop click cancels; the guard keeps a click that started inside the
      // card (e.g. a drag over text) from closing it.
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel()
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-description"
        className="w-full max-w-sm rounded-2xl border border-border bg-card shadow-md p-6 flex flex-col gap-4"
      >
        <div className="flex flex-col gap-1.5">
          <h2 id="confirm-dialog-title" className="text-base font-semibold">
            {title}
          </h2>
          <p id="confirm-dialog-description" className="text-sm text-muted-foreground">
            {description}
          </p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={onCancel}
            className="flex-1 rounded-xl border border-border bg-background px-4 py-2.5 text-sm font-medium hover:bg-muted transition-colors"
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            onClick={onConfirm}
            className="flex-1 rounded-xl bg-primary text-primary-foreground px-4 py-2.5 text-sm font-medium shadow-sm hover:opacity-90 transition-opacity"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
