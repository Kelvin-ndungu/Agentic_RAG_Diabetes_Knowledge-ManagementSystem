/**
 * Chat service for API integration.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Send a chat message with streaming response.
 * 
 * @param {string} message - User's message
 * @param {string|null} sessionId - Optional session ID for conversation continuity
 * @returns {Promise<AsyncGenerator>} Async generator yielding response chunks
 */
export async function* sendMessage(message, sessionId = null) {
  const url = `${API_BASE_URL}/api/chat`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new Error(error.message || `HTTP error! status: ${response.status}`);
  }

  // Read streaming response (newline-delimited JSON)
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }

      // Decode chunk and add to buffer
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete lines
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);
            yield data;
          } catch (e) {
            console.error('Failed to parse JSON line:', line, e);
          }
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      try {
        const data = JSON.parse(buffer);
        yield data;
      } catch (e) {
        console.error('Failed to parse final buffer:', buffer, e);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Clear conversation history for a session.
 * 
 * @param {string} sessionId - Session ID to clear
 * @returns {Promise<Object>} Response object
 */
export async function clearChat(sessionId) {
  const url = `${API_BASE_URL}/api/chat/clear`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new Error(error.message || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Health check endpoint.
 * 
 * @returns {Promise<Object>} Health status
 */
export async function healthCheck() {
  const url = `${API_BASE_URL}/api/health`;
  
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

