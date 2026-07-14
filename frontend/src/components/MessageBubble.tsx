import { User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { SourcePanel } from './SourcePanel'
import type { Message } from '@/types'

interface Props {
  message: Message
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      <div
        className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm ${
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted text-muted-foreground'
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <span className="font-bold text-sm leading-none">G</span>}
      </div>

      <div className={`max-w-[78%] ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-primary text-primary-foreground rounded-tr-sm'
              : 'bg-muted text-foreground rounded-tl-sm'
          }`}
        >
          {message.loading ? (
            <span className="flex gap-2 items-center h-5">
              <span className="flex gap-1 items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-bounce [animation-delay:0ms]" />
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-bounce [animation-delay:150ms]" />
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-bounce [animation-delay:300ms]" />
              </span>
              {message.status && (
                <span className="text-xs text-muted-foreground animate-pulse">
                  {message.status}
                </span>
              )}
            </span>
          ) : isUser ? (
            message.content
          ) : (
            <div
              className="prose prose-sm dark:prose-invert max-w-none
                prose-p:my-2 first:prose-p:mt-0 last:prose-p:mb-0
                prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5
                prose-headings:mt-3 prose-headings:mb-1.5 first:prose-headings:mt-0
                prose-pre:my-2 prose-strong:text-inherit prose-headings:text-inherit
                prose-a:text-inherit prose-a:underline"
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="px-4 w-full">
            <SourcePanel sources={message.sources} />
          </div>
        )}
      </div>
    </div>
  )
}
