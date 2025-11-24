

// import { useEffect, useRef } from 'react'
// import Message from './Message'

// export default function MessageList({ messages, loading, isStreaming, shouldAutoScroll, disableAutoScroll }) {
//   const messagesEndRef = useRef(null)
//   const messageListRef = useRef(null)
  
//   // NEW: Track last scroll time to prevent detecting our own auto-scroll as user scroll
//   const lastAutoScrollTimeRef = useRef(0)
  
//   // NEW: Track touch start position for mobile scroll detection
//   const touchStartRef = useRef({ x: 0, y: 0 })

//   const scrollToBottom = () => {
//     if (messagesEndRef.current) {
//       // Record when we auto-scroll to avoid false detection
//       lastAutoScrollTimeRef.current = Date.now()
//       messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
//     }
//   }

//   const isNearBottom = () => {
//     if (!messageListRef.current) return true
//     const container = messageListRef.current
//     const threshold = 100 // pixels from bottom
//     return container.scrollHeight - container.scrollTop - container.clientHeight < threshold
//   }

//   // CHANGED: Improved scroll detection that ignores our own auto-scrolls
//   useEffect(() => {
//     const container = messageListRef.current
//     if (!container) return

//     const handleScroll = () => {
//       // Check if this scroll happened very recently after our auto-scroll
//       const timeSinceAutoScroll = Date.now() - lastAutoScrollTimeRef.current
      
//       // If this scroll is within 100ms of our auto-scroll, ignore it
//       if (timeSinceAutoScroll < 100) {
//         return
//       }
      
//       // If user scrolled away from bottom, disable auto-scroll
//       if (!isNearBottom()) {
//         disableAutoScroll()
//         console.log('User scrolled up - auto-scroll disabled') // Debug log
//       }
//     }

//     container.addEventListener('scroll', handleScroll, { passive: true })
//     return () => container.removeEventListener('scroll', handleScroll)
//   }, [disableAutoScroll])

//   // NEW: Handle touch scrolling for mobile devices
//   useEffect(() => {
//     const container = messageListRef.current
//     if (!container) return

//     const handleTouchStart = (e) => {
//       touchStartRef.current = {
//         x: e.touches[0].clientX,
//         y: e.touches[0].clientY
//       }
//     }

//     const handleTouchMove = (e) => {
//       const deltaY = e.touches[0].clientY - touchStartRef.current.y
      
//       // If user is swiping up (negative deltaY), they want to scroll up
//       if (deltaY < -10) {
//         if (!isNearBottom()) {
//           disableAutoScroll()
//           console.log('Touch scroll up detected - auto-scroll disabled') // Debug log
//         }
//       }
//     }

//     container.addEventListener('touchstart', handleTouchStart, { passive: true })
//     container.addEventListener('touchmove', handleTouchMove, { passive: true })
    
//     return () => {
//       container.removeEventListener('touchstart', handleTouchStart)
//       container.removeEventListener('touchmove', handleTouchMove)
//     }
//   }, [disableAutoScroll])

//   // CHANGED: Simplified auto-scroll logic
//   useEffect(() => {
//     // Only auto-scroll if shouldAutoScroll is true
//     // This flag is controlled by the useChat hook
//     if (shouldAutoScroll) {
//       scrollToBottom()
//       console.log('Auto-scrolling...', { 
//         shouldAutoScroll, 
//         isStreaming, 
//         messageCount: messages.length 
//       }) // Debug log
//     }
//   }, [messages, shouldAutoScroll])

//   return (
//     <div className="message-list" ref={messageListRef}>
//       {messages.length === 0 ? (
//         <div className="empty-state">
//           <p>Ask a question about the diabetes guidelines to get started.</p>
//         </div>
//       ) : (
//         messages.map((message, index) => (
//           <Message key={index} message={message} />
//         ))
//       )}
//       {loading && (
//         <div className="message message-assistant loading">
//           <div className="message-content">
//             <div className="typing-indicator">
//               <span></span>
//               <span></span>
//               <span></span>
//             </div>
//           </div>
//         </div>
//       )}
//       <div ref={messagesEndRef} />
//     </div>
//   )
// }


import { useEffect, useRef } from 'react'
import Message from './Message'

export default function MessageList({ messages, loading, isStreaming, shouldAutoScroll, disableAutoScroll }) {
  const messagesEndRef = useRef(null)
  const messageListRef = useRef(null)
  
  // NEW: Track last scroll time to prevent detecting our own auto-scroll as user scroll
  const lastAutoScrollTimeRef = useRef(0)
  
  // NEW: Track when auto-scroll was just enabled to ignore scroll events briefly
  const autoScrollJustEnabledRef = useRef(false)
  
  // NEW: Track the previous shouldAutoScroll value to detect when it changes
  const prevShouldAutoScrollRef = useRef(shouldAutoScroll)
  
  // NEW: Track touch start position for mobile scroll detection
  const touchStartRef = useRef({ x: 0, y: 0 })

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      // Record when we auto-scroll to avoid false detection
      lastAutoScrollTimeRef.current = Date.now()
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }

  const isNearBottom = () => {
    if (!messageListRef.current) return true
    const container = messageListRef.current
    const threshold = 100 // pixels from bottom
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold
  }

  // CRITICAL: Detect when shouldAutoScroll changes from false to true (new message sent)
  useEffect(() => {
    if (shouldAutoScroll && !prevShouldAutoScrollRef.current) {
      // shouldAutoScroll just changed from false to true - new message was sent!
      console.log('New message sent - ignoring scroll events briefly')
      autoScrollJustEnabledRef.current = true
      
      // After 500ms, allow scroll detection again
      setTimeout(() => {
        autoScrollJustEnabledRef.current = false
        console.log('Scroll detection re-enabled')
      }, 500)
    }
    
    // Update the previous value
    prevShouldAutoScrollRef.current = shouldAutoScroll
  }, [shouldAutoScroll])

  // CHANGED: Improved scroll detection that ignores our own auto-scrolls AND ignores scroll when new message just sent
  useEffect(() => {
    const container = messageListRef.current
    if (!container) return

    const handleScroll = () => {
      // CRITICAL: If auto-scroll was just enabled (new message sent), ignore scroll events
      if (autoScrollJustEnabledRef.current) {
        console.log('Ignoring scroll - new message just sent')
        return
      }
      
      // Check if this scroll happened very recently after our auto-scroll
      const timeSinceAutoScroll = Date.now() - lastAutoScrollTimeRef.current
      
      // If this scroll is within 150ms of our auto-scroll, ignore it
      if (timeSinceAutoScroll < 150) {
        console.log('Ignoring scroll - auto-scroll in progress')
        return
      }
      
      // If user scrolled away from bottom, disable auto-scroll
      if (!isNearBottom()) {
        disableAutoScroll()
        console.log('User scrolled up - auto-scroll disabled')
      }
    }

    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => container.removeEventListener('scroll', handleScroll)
  }, [disableAutoScroll])

  // NEW: Handle touch scrolling for mobile devices
  useEffect(() => {
    const container = messageListRef.current
    if (!container) return

    const handleTouchStart = (e) => {
      touchStartRef.current = {
        x: e.touches[0].clientX,
        y: e.touches[0].clientY
      }
    }

    const handleTouchMove = (e) => {
      // CRITICAL: If auto-scroll was just enabled (new message sent), ignore touch events
      if (autoScrollJustEnabledRef.current) {
        console.log('Ignoring touch - new message just sent')
        return
      }
      
      const deltaY = e.touches[0].clientY - touchStartRef.current.y
      
      // If user is swiping up (negative deltaY), they want to scroll up
      if (deltaY < -10) {
        if (!isNearBottom()) {
          disableAutoScroll()
          console.log('Touch scroll up detected - auto-scroll disabled')
        }
      }
    }

    container.addEventListener('touchstart', handleTouchStart, { passive: true })
    container.addEventListener('touchmove', handleTouchMove, { passive: true })
    
    return () => {
      container.removeEventListener('touchstart', handleTouchStart)
      container.removeEventListener('touchmove', handleTouchMove)
    }
  }, [disableAutoScroll])

  // CHANGED: Simplified auto-scroll logic
  useEffect(() => {
    // Only auto-scroll if shouldAutoScroll is true
    // This flag is controlled by the useChat hook
    if (shouldAutoScroll) {
      scrollToBottom()
      console.log('Auto-scrolling...', { 
        shouldAutoScroll, 
        isStreaming, 
        messageCount: messages.length 
      })
    }
  }, [messages, shouldAutoScroll, isStreaming])

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