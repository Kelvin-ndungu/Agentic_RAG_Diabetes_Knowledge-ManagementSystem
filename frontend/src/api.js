// Justification: centralize API calls and markdown helpers in one module.
/**
 * API + shared helpers.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export async function* sendMessage(message, sessionId = null) {
  const url = `${API_BASE_URL}/api/chat`

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }))
    throw new Error(error.message || `HTTP error! status: ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line)
            yield data
          } catch (err) {
            console.error('Failed to parse JSON line:', line, err)
          }
        }
      }
    }

    if (buffer.trim()) {
      try {
        const data = JSON.parse(buffer)
        yield data
      } catch (err) {
        console.error('Failed to parse final buffer:', buffer, err)
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export async function clearChat(sessionId) {
  const url = `${API_BASE_URL}/api/chat/clear`

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }))
    throw new Error(error.message || `HTTP error! status: ${response.status}`)
  }

  return await response.json()
}

export async function healthCheck() {
  const url = `${API_BASE_URL}/api/health`
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  return await response.json()
}

export function resolveImagePaths(content) {
  if (!content || typeof content !== 'string') {
    return content
  }

  return content.replace(
    /!\[([^\]]*)\]\(images\/([^\)]+)\)/g,
    '![$1](/images/$2)'
  )
}

export function removeDuplicateHeading(content, title) {
  if (!content || !title) return content

  const titleText = title.replace(/^\d+\.\d+(\.\d+)*\.?\s*/, '').trim()

  const headingPatterns = [
    new RegExp(`^###\\s+${title.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\s*\\n+`, 'i'),
    new RegExp(`^###\\s+${titleText.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\s*\\n+`, 'i'),
    new RegExp(`^##\\s+${title.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\s*\\n+`, 'i'),
    new RegExp(`^#\\s+${title.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\s*\\n+`, 'i'),
  ]

  let cleanedContent = content

  for (const pattern of headingPatterns) {
    if (pattern.test(cleanedContent)) {
      cleanedContent = cleanedContent.replace(pattern, '')
      break
    }
  }

  return cleanedContent.trim()
}
