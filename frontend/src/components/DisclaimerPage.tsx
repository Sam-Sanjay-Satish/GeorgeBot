interface Props {
  onContinue: () => void
}

export function DisclaimerPage({ onContinue }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-screen bg-background">
      <div className="w-full max-w-sm flex flex-col items-center gap-8 px-6">

        {/* Logo */}
        <div className="flex flex-col items-center gap-4">
          <div className="w-20 h-20 rounded-2xl bg-primary flex items-center justify-center text-primary-foreground font-bold text-3xl">
            G
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-semibold">GeorgeBot</h1>
            <p className="text-sm text-muted-foreground mt-1">UVic AI Assistant</p>
          </div>
        </div>

        {/* Intro card */}
        <div className="w-full rounded-2xl border border-border bg-card shadow-sm p-6 flex flex-col gap-4">
          <div className="text-center">
            <p className="text-sm font-medium">Ask anything about UVic</p>
          </div>

          <p className="text-sm text-muted-foreground text-center">
            GeorgeBot answers straight from{' '}
            <a
              href="https://www.uvic.ca"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-foreground"
            >
              uvic.ca
            </a>{' '}
            — course details, policies, live registration data, and even
            niche facts — so you get accurate, up-to-date information without
            digging through the site yourself.
          </p>

          <button
            onClick={onContinue}
            className="w-full flex items-center justify-center rounded-xl bg-primary text-primary-foreground px-4 py-2.5 text-sm font-medium shadow-sm hover:opacity-90 transition-opacity"
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  )
}
