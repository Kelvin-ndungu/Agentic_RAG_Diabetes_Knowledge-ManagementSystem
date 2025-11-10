import { useState, useEffect, useRef } from 'react'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import { useChat } from '../../hooks/useChat'

export default function ChatInterface({ isOpen, onClose, initialQuery = '', isMobile = false, onWidthChange }) {
  const { messages, sendMessage, loading, clearMessages } = useChat(initialQuery)
  const [width, setWidth] = useState(33.33) // Default to 1/3 (33.33%)
  const [isDragging, setIsDragging] = useState(false)
  const chatRef = useRef(null)
  const startXRef = useRef(0)
  const startWidthRef = useRef(0)

  useEffect(() => {
    if (isOpen && !isMobile) {
      // Reset to 1/3 when opening
      setWidth(33.33)
      if (onWidthChange) {
        onWidthChange(33.33)
      }
    }
  }, [isOpen, isMobile, onWidthChange])

  const handleMouseDown = (e) => {
    if (isMobile) return
    setIsDragging(true)
    startXRef.current = e.clientX
    startWidthRef.current = width
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    e.preventDefault()
  }

  const handleMouseMove = (e) => {
    if (!isDragging || isMobile) return
    
    const windowWidth = window.innerWidth
    const sidebarWidth = 240 // Fixed sidebar width
    const deltaX = startXRef.current - e.clientX // Inverted because we're dragging left
    const deltaPercent = (deltaX / windowWidth) * 100 // Percentage of full viewport
    let newWidth = startWidthRef.current + deltaPercent
    
    // Constrain between 33.33% (1/3) and 50% (1/2) of viewport
    newWidth = Math.max(33.33, Math.min(50, newWidth))
    
    setWidth(newWidth)
    if (onWidthChange) {
      onWidthChange(newWidth)
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  }

  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [])

  const handleSend = (message) => {
    sendMessage(message)
  }

  if (!isOpen) return null

  const chatWidth = isMobile ? '100vw' : `${width}%`

  return (
    <>
      {isMobile && <div className="chat-overlay" onClick={onClose} />}
      <div 
        ref={chatRef}
        className={`chat-interface ${isMobile ? 'mobile' : 'desktop'} ${isDragging ? 'dragging' : ''}`}
        style={{ width: chatWidth }}
      >
        {!isMobile && (
          <div 
            className="chat-resizer"
            onMouseDown={handleMouseDown}
          />
        )}
        <div className="chat-header">
          <h3>Ask about Diabetes Guidelines</h3>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button 
              className="chat-clear"
              onClick={clearMessages}
              aria-label="Clear conversation"
              title="Clear conversation"
            >
              Clear
            </button>
            <button 
              className="chat-close"
              onClick={onClose}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>
        </div>
        
        <MessageList messages={messages} loading={loading} />
        
        <ChatInput 
          onSend={handleSend}
          disabled={loading}
        />
      </div>
    </>
  )
}

