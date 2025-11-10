import { useEffect, useRef } from 'react'
import Message from './Message'

export default function MessageList({ messages, loading }) {
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, loading])

  return (
    <div className="message-list">
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

