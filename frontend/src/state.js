// Justification: centralize state hooks to reduce file count.
import { useState, useEffect, useRef } from 'react'
import documentData from './data/document_structure.json'
import { sendMessage as sendChatMessage, clearChat } from './api'

export function useChat(initialQuery = '') {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true)

  const statusMessageIdRef = useRef(null)

  useEffect(() => {
    const welcomeMessage = {
      role: 'assistant',
      content: 'Welcome! I can help you find information about diabetes management guidelines. Ask me anything!',
      timestamp: new Date().toISOString(),
    }

    setMessages([welcomeMessage])
  }, [])

  const initialQuerySentRef = useRef(false)

  useEffect(() => {
    if (initialQuery && !initialQuerySentRef.current && messages.length === 1) {
      const welcomeMsg = messages[0]
      if (welcomeMsg && welcomeMsg.content.includes('Welcome')) {
        initialQuerySentRef.current = true
        setTimeout(() => {
          sendMessage(initialQuery)
        }, 100)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuery, messages.length])

  useEffect(() => {
    initialQuerySentRef.current = false
  }, [initialQuery])

  const sendMessage = async (content) => {
    setShouldAutoScroll(true)

    const userMessage = {
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setLoading(true)
    statusMessageIdRef.current = null

    try {
      const statusId = `status-${Date.now()}`
      statusMessageIdRef.current = statusId

      const statusMessage = {
        id: statusId,
        role: 'assistant',
        content: 'Processing...',
        timestamp: new Date().toISOString(),
        isStatus: true,
      }

      setMessages((prev) => [...prev, statusMessage])

      let finalAnswer = null
      let finalSources = []
      let finalSessionId = sessionId
      let streamingContent = ''
      let answerMessageId = null

      for await (const chunk of sendChatMessage(content, sessionId)) {
        if (chunk.type === 'status') {
          setMessages((prev) => {
            const updated = [...prev]
            const statusIndex = updated.findIndex((msg) => msg.id === statusId)
            if (statusIndex !== -1) {
              updated[statusIndex] = {
                ...updated[statusIndex],
                content: chunk.message,
              }
            }
            return updated
          })
        } else if (chunk.type === 'stream_start') {
          setIsStreaming(true)
          setMessages((prev) => {
            const updated = [...prev]
            const statusIndex = updated.findIndex((msg) => msg.id === statusId)
            if (statusIndex !== -1) {
              answerMessageId = statusId
              updated[statusIndex] = {
                id: statusId,
                role: 'assistant',
                content: '',
                sources: [],
                timestamp: new Date().toISOString(),
              }
            }
            return updated
          })
        } else if (chunk.type === 'token') {
          streamingContent += chunk.content
          if (answerMessageId) {
            setMessages((prev) => {
              const updated = [...prev]
              const answerIndex = updated.findIndex((msg) => msg.id === answerMessageId)
              if (answerIndex !== -1) {
                updated[answerIndex] = {
                  ...updated[answerIndex],
                  content: streamingContent,
                }
              }
              return updated
            })
          }
        } else if (chunk.type === 'stream_end') {
          setIsStreaming(false)
          finalAnswer = chunk.content || streamingContent
          finalSources = chunk.sources || []
          if (chunk.session_id) {
            finalSessionId = chunk.session_id
            setSessionId(chunk.session_id)
          }

          setMessages((prev) => {
            const updated = [...prev]
            const answerIndex = updated.findIndex((msg) => msg.id === (answerMessageId || statusId))
            if (answerIndex !== -1) {
              updated[answerIndex] = {
                ...updated[answerIndex],
                content: finalAnswer,
                sources: finalSources,
              }
            }
            return updated
          })

          setTimeout(() => {
            setShouldAutoScroll(false)
          }, 100)
        } else if (chunk.type === 'answer') {
          finalAnswer = chunk.content
          finalSources = chunk.sources || []
          if (chunk.session_id) {
            finalSessionId = chunk.session_id
            setSessionId(chunk.session_id)
          }
        } else if (chunk.type === 'error') {
          throw new Error(chunk.message || 'An error occurred')
        }
      }

      if (finalAnswer !== null && !answerMessageId) {
        setMessages((prev) => {
          const updated = [...prev]
          const statusIndex = updated.findIndex((msg) => msg.id === statusId)

          if (statusIndex !== -1) {
            updated[statusIndex] = {
              role: 'assistant',
              content: finalAnswer,
              sources: finalSources,
              timestamp: new Date().toISOString(),
            }
          } else {
            updated.push({
              role: 'assistant',
              content: finalAnswer,
              sources: finalSources,
              timestamp: new Date().toISOString(),
            })
          }

          return updated
        })

        setTimeout(() => {
          setShouldAutoScroll(false)
        }, 100)
      }

      statusMessageIdRef.current = null
      setLoading(false)
      setIsStreaming(false)
    } catch (error) {
      console.error('Chat error:', error)
      setIsStreaming(false)

      setMessages((prev) => {
        const updated = [...prev]
        const statusIndex = updated.findIndex((msg) => msg.id === statusMessageIdRef.current)

        if (statusIndex !== -1) {
          updated[statusIndex] = {
            role: 'assistant',
            content: `I'm sorry, I encountered an error: ${error.message}. Please try again.`,
            timestamp: new Date().toISOString(),
          }
        } else {
          updated.push({
            role: 'assistant',
            content: `I'm sorry, I encountered an error: ${error.message}. Please try again.`,
            timestamp: new Date().toISOString(),
          })
        }

        return updated
      })

      statusMessageIdRef.current = null
      setLoading(false)

      setTimeout(() => {
        setShouldAutoScroll(false)
      }, 100)
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

    const welcomeMessage = {
      role: 'assistant',
      content: 'Welcome! I can help you find information about diabetes management guidelines. Ask me anything!',
      timestamp: new Date().toISOString(),
    }

    setMessages([welcomeMessage])
    setSessionId(null)
    setShouldAutoScroll(true)
  }

  const disableAutoScroll = () => {
    setShouldAutoScroll(false)
  }

  return {
    messages,
    sendMessage,
    loading,
    isStreaming,
    clearMessages,
    sessionId,
    shouldAutoScroll,
    disableAutoScroll,
  }
}

export function useDocument() {
  const [document, setDocument] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    try {
      setTimeout(() => {
        setDocument(documentData)
        setLoading(false)
      }, 100)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }, [])

  const findSectionById = (id) => {
    if (!document) return null

    for (const item of document.document.frontMatter) {
      if (item.id === id) return item
      if (item.sections) {
        const found = findInSections(item.sections, id)
        if (found) return found
      }
    }

    for (const chapter of document.document.chapters) {
      if (chapter.id === id) return chapter
      if (chapter.sections) {
        const found = findInSections(chapter.sections, id)
        if (found) return found
      }
    }

    return null
  }

  const findInSections = (sections, id) => {
    for (const section of sections) {
      if (section.id === id) return section
      if (section.subsections) {
        const found = findInSections(section.subsections, id)
        if (found) return found
      }
    }
    return null
  }

  const findSectionBySlug = (slug) => {
    if (!document) return null

    for (const item of document.document.frontMatter) {
      if (item.slug === slug) return item
      if (item.sections) {
        const found = findInSectionsBySlug(item.sections, slug)
        if (found) return found
      }
    }

    for (const chapter of document.document.chapters) {
      if (chapter.slug === slug) return chapter
      if (chapter.sections) {
        const found = findInSectionsBySlug(chapter.sections, slug)
        if (found) return found
      }
    }

    return null
  }

  const findInSectionsBySlug = (sections, slug) => {
    for (const section of sections) {
      if (section.slug === slug) return section
      if (section.subsections) {
        const found = findInSectionsBySlug(section.subsections, slug)
        if (found) return found
      }
    }
    return null
  }

  return {
    document,
    loading,
    error,
    findSectionById,
    findSectionBySlug,
  }
}
