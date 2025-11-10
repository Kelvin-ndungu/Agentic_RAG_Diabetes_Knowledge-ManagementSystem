import { useState, useEffect, useRef } from 'react'
import { sendMessage as sendChatMessage, clearChat } from '../services/chatService'

export function useChat(initialQuery = '') {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const statusMessageIdRef = useRef(null)

  useEffect(() => {
    // Initialize with welcome message
    const welcomeMessage = {
      role: 'assistant',
      content: 'Welcome! I can help you find information about diabetes management guidelines. Ask me anything!',
      timestamp: new Date().toISOString()
    }

    setMessages([welcomeMessage])
  }, [])

  // Handle initial query separately
  const initialQuerySentRef = useRef(false)
  
  useEffect(() => {
    if (initialQuery && !initialQuerySentRef.current && messages.length === 1) {
      // Only send if we just have the welcome message and haven't sent yet
      const welcomeMsg = messages[0]
      if (welcomeMsg && welcomeMsg.content.includes('Welcome')) {
        initialQuerySentRef.current = true
        // Use setTimeout to avoid calling sendMessage during render
        setTimeout(() => {
          sendMessage(initialQuery)
        }, 100)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuery, messages.length])
  
  // Reset initial query sent flag when initialQuery changes
  useEffect(() => {
    initialQuerySentRef.current = false
  }, [initialQuery])

  const sendMessage = async (content) => {
    // Add user message
    const userMessage = {
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setLoading(true)
    statusMessageIdRef.current = null

    try {
      // Create temporary status message
      const statusId = `status-${Date.now()}`
      statusMessageIdRef.current = statusId
      
      const statusMessage = {
        id: statusId,
        role: 'assistant',
        content: 'Processing...',
        timestamp: new Date().toISOString(),
        isStatus: true
      }

      setMessages(prev => [...prev, statusMessage])

      // Stream response from API
      let finalAnswer = null
      let finalSources = []
      let finalSessionId = sessionId

      for await (const chunk of sendChatMessage(content, sessionId)) {
        if (chunk.type === 'status') {
          // Update status message
          setMessages(prev => {
            const updated = [...prev]
            const statusIndex = updated.findIndex(msg => msg.id === statusId)
            if (statusIndex !== -1) {
              updated[statusIndex] = {
                ...updated[statusIndex],
                content: chunk.message
              }
            }
            return updated
          })
        } else if (chunk.type === 'answer') {
          // Final answer received
          finalAnswer = chunk.content
          finalSources = chunk.sources || []
          if (chunk.session_id) {
            finalSessionId = chunk.session_id
            setSessionId(chunk.session_id)
          }
        } else if (chunk.type === 'error') {
          // Error occurred
          throw new Error(chunk.message || 'An error occurred')
        }
      }

      // Replace status message with final answer
      if (finalAnswer !== null) {
        setMessages(prev => {
          const updated = [...prev]
          const statusIndex = updated.findIndex(msg => msg.id === statusId)
          
          if (statusIndex !== -1) {
            // Replace status message with final answer
            updated[statusIndex] = {
              role: 'assistant',
              content: finalAnswer,
              sources: finalSources,
              timestamp: new Date().toISOString()
            }
          } else {
            // Status message not found, add new message
            updated.push({
              role: 'assistant',
              content: finalAnswer,
              sources: finalSources,
              timestamp: new Date().toISOString()
            })
          }
          
          return updated
        })
      }

      statusMessageIdRef.current = null
      setLoading(false)
    } catch (error) {
      console.error('Chat error:', error)
      
      // Replace status message with error message
      setMessages(prev => {
        const updated = [...prev]
        const statusIndex = updated.findIndex(msg => msg.id === statusMessageIdRef.current)
        
        if (statusIndex !== -1) {
          updated[statusIndex] = {
            role: 'assistant',
            content: `I'm sorry, I encountered an error: ${error.message}. Please try again.`,
            timestamp: new Date().toISOString()
          }
        } else {
          updated.push({
            role: 'assistant',
            content: `I'm sorry, I encountered an error: ${error.message}. Please try again.`,
            timestamp: new Date().toISOString()
          })
        }
        
        return updated
      })
      
      statusMessageIdRef.current = null
      setLoading(false)
    }
  }

  const clearMessages = async () => {
    if (sessionId) {
      try {
        await clearChat(sessionId)
      } catch (error) {
        console.error('Error clearing chat:', error)
      }
    }
    
    // Reset to welcome message
    const welcomeMessage = {
      role: 'assistant',
      content: 'Welcome! I can help you find information about diabetes management guidelines. Ask me anything!',
      timestamp: new Date().toISOString()
    }
    
    setMessages([welcomeMessage])
    setSessionId(null)
  }

  return { messages, sendMessage, loading, clearMessages, sessionId }
}

