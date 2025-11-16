import { useEffect, useRef, useState } from 'react'
import Message from './Message'

export default function MessageList({ messages, loading, isStreaming }) {
  const messagesEndRef = useRef(null)
  const messageListRef = useRef(null)
  const [userHasScrolled, setUserHasScrolled] = useState(false)
  const wasStreamingRef = useRef(false)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const isNearBottom = () => {
    if (!messageListRef.current) return true
    const container = messageListRef.current
    const threshold = 100 // pixels from bottom
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold
  }

  // Track user scroll to detect manual scrolling
  useEffect(() => {
    const container = messageListRef.current
    if (!container) return

    const handleScroll = () => {
      if (!isStreaming) {
        // Only track scroll when not streaming
        setUserHasScrolled(!isNearBottom())
      }
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [isStreaming])

  // Auto-scroll logic: only scroll when NOT streaming, or when user is near bottom
  useEffect(() => {
    // If streaming just started, preserve current position
    if (isStreaming && !wasStreamingRef.current) {
      wasStreamingRef.current = true
      setUserHasScrolled(!isNearBottom())
      return
    }

    // If streaming stopped, reset scroll tracking
    if (!isStreaming && wasStreamingRef.current) {
      wasStreamingRef.current = false
      setUserHasScrolled(false)
      // Scroll to bottom when streaming completes
      scrollToBottom()
      return
    }

    // Only auto-scroll if:
    // 1. Not currently streaming, OR
    // 2. User is near the bottom (within threshold)
    if (!isStreaming || (isStreaming && !userHasScrolled && isNearBottom())) {
      scrollToBottom()
    }
  }, [messages, loading, isStreaming, userHasScrolled])

  return (
    <div className="message-list" ref={messageListRef}>
      {messages.length === 0 ? (
        <div className="empty-state">
          <p>Ask a question about the diabetes guidelines to get started.</p>
        </div>
      ) : (
        messages.map((message, index) => (
          <Message key={index} message={message} />
        ))
      )}
      {loading && (
        <div className="message message-assistant loading">
          <div className="message-content">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  )
}

