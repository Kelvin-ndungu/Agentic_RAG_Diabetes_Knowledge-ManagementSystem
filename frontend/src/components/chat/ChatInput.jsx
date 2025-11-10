import { useState } from 'react'

export default function ChatInput({ onSend, disabled }) {
  const [input, setInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (input.trim() && !disabled) {
      onSend(input.trim())
      setInput('')
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="chat-input-container">
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <textarea
          className="chat-input"
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={disabled}
          rows={1}
        />
        <button 
          type="submit" 
          className="chat-send-button"
          disabled={disabled || !input.trim()}
          aria-label="Send message"
        >
          <span>→</span>
        </button>
      </form>
    </div>
  )
}

