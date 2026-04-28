// Justification: consolidate UI components into one file for a portable case-study bundle.
import { useState, useEffect, useRef, useCallback } from 'react'
import { Link, useLocation } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { useChat } from './state'
import { resolveImagePaths, removeDuplicateHeading } from './api'

export function Header({ onMenuClick, onSearchClick, isMobile, chatOpen, viewMode, onViewModeChange }) {
  const [searchQuery, setSearchQuery] = useState('')

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      onSearchClick(searchQuery.trim())
      setSearchQuery('')
    } else {
      onSearchClick('')
    }
  }

  const handleInputFocus = () => {
    if (isMobile) {
      onSearchClick('')
    }
  }

  return (
    <header className="main-header">
      <div className="header-content">
        <form className="search-form-centered" onSubmit={handleSearchSubmit}>
          <input
            type="text"
            className="search-input-centered"
            placeholder={chatOpen ? 'Use the chat interface to ask questions...' : 'Ask any question in plain language...'}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={handleInputFocus}
            disabled={chatOpen}
            style={{
              opacity: chatOpen ? 0.5 : 1,
              cursor: chatOpen ? 'not-allowed' : 'text',
            }}
          />
        </form>

        <div className="header-controls">
          <div className="view-mode-toggle" aria-label="View mode">
            <button
              className={`view-mode-btn ${viewMode === 'normal' ? 'active' : ''}`}
              onClick={() => onViewModeChange('normal')}
              title="Split view (document + chat)"
              aria-label="Split view"
            >
              Split
            </button>
            <button
              className={`view-mode-btn ${viewMode === 'chat-only' ? 'active' : ''}`}
              onClick={() => onViewModeChange('chat-only')}
              title="Chat panel only"
              aria-label="Chat view"
            >
              Chat
            </button>
            <button
              className={`view-mode-btn ${viewMode === 'document-only' ? 'active' : ''}`}
              onClick={() => onViewModeChange('document-only')}
              title="Document only"
              aria-label="Document view"
            >
              Document
            </button>
            <button
              className={`view-mode-btn ${viewMode === 'chat-alone' ? 'active' : ''}`}
              onClick={() => onViewModeChange('chat-alone')}
              title="Centered chat focus"
              aria-label="Chat focus view"
            >
              Focus
            </button>
          </div>

          <button className="menu-button" onClick={onMenuClick} aria-label="Toggle navigation menu" title="Toggle sidebar">
            <span className="hamburger-icon">Menu</span>
          </button>
        </div>
      </div>
    </header>
  )
}

export function Sidebar({ document, isOpen, onClose, isMobile }) {
  if (!document) return null

  return (
    <>
      {isMobile && isOpen && (
        <div className="sidebar-overlay" onClick={onClose} aria-hidden="true" />
      )}

      <aside className={`sidebar ${isOpen ? 'open' : ''} ${isMobile ? 'mobile' : ''}`}>
        <div className="sidebar-header">
          <Link to="/" className="sidebar-title" onClick={isMobile ? onClose : undefined}>
            <h2>{document.title}</h2>
          </Link>
          {isMobile && (
            <button className="sidebar-close" onClick={onClose} aria-label="Close sidebar">
              x
            </button>
          )}
        </div>

        <nav className="sidebar-nav">
          {document.chapters && document.chapters.length > 0 && (
            <div className="nav-section">
              <NavTree items={document.chapters} onNavigate={isMobile ? onClose : undefined} />
            </div>
          )}

          {document.frontMatter && document.frontMatter.length > 0 && (
            <div className="nav-section">
              <NavTree items={document.frontMatter} onNavigate={isMobile ? onClose : undefined} />
            </div>
          )}
        </nav>
      </aside>
    </>
  )
}

export function NavTree({ items, level = 0, onNavigate }) {
  if (!items || items.length === 0) return null

  return (
    <ul className={`nav-tree level-${level}`}>
      {items.map((item) => (
        <NavItem key={item.id} item={item} level={level} onNavigate={onNavigate} />
      ))}
    </ul>
  )
}

export function NavItem({ item, level = 0, onNavigate }) {
  const [isExpanded, setIsExpanded] = useState(level === 0)
  const location = useLocation()

  const isActive = location.pathname === item.url || location.pathname.includes(item.url)

  const hasChildren = (item.sections && item.sections.length > 0) || (item.subsections && item.subsections.length > 0)
  const children = item.sections || item.subsections || []

  const handleToggle = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsExpanded(!isExpanded)
  }

  const handleClick = () => {
    if (onNavigate) {
      onNavigate()
    }
  }

  return (
    <li className={`nav-item level-${level} ${isActive ? 'active' : ''}`}>
      <div className="nav-item-header">
        <Link to={item.url} className="nav-item-link" onClick={handleClick}>
          <span className="nav-title">{item.title}</span>
        </Link>

        {hasChildren && (
          <button className="nav-toggle" onClick={handleToggle} aria-label={isExpanded ? 'Collapse' : 'Expand'} aria-expanded={isExpanded}>
            {isExpanded ? 'v' : '>'}
          </button>
        )}
      </div>

      {hasChildren && isExpanded && <NavTree items={children} level={level + 1} onNavigate={onNavigate} />}
    </li>
  )
}

export function HomePage({ document }) {
  return (
    <div className="home-page">
      <div className="home-container">
        <header className="home-header">
          <h1>{document.title}</h1>
          <p className="version">Version: {document.version}</p>
        </header>

        <section className="home-content">
          <div className="info-section">
            <h2>About This Document</h2>
            <p>
              This is the {document.version} of the Kenya National Clinical Guidelines for the Management of Diabetes
              Mellitus. These guidelines provide a standardized approach to managing diabetes in Kenya, developed by
              the National Diabetes Prevention and Control Program, Division of Non-communicable Diseases, Ministry of
              Health, Kenya.
            </p>
          </div>

          <div className="info-section">
            <h2>How to Use This Guide</h2>
            <ol className="instructions-list">
              <li>Use the sidebar to navigate through chapters and sections.</li>
              <li>Click on any section to view its content with images and diagrams.</li>
              <li>On mobile devices, tap the menu icon to open the navigation sidebar.</li>
            </ol>
          </div>

          <div className="info-section">
            <h2>Source & Attribution</h2>
            <div className="source-info">
              <p><strong>Produced by:</strong> The National Diabetes Prevention and Control Program</p>
              <p><strong>Division:</strong> Division of Non-communicable Diseases, Ministry of Health, Kenya</p>
              <p><strong>Funded by:</strong> Ministry of Health, Kenya Diabetes Management and Information Centre and World Diabetes Foundation (WDF)</p>
              <p><strong>Publication Year:</strong> 2018</p>
            </div>
          </div>

          <div className="info-section">
            <h2>Developer Information</h2>
            <div className="developer-info">
              <p><strong>Developed by:</strong> Kelvin Ndungu Kinyanjui</p>
              <p><strong>Mobile:</strong> <a href="tel:+254713281876">+254 713 281 876</a></p>
              <p><strong>Email:</strong> <a href="mailto:Kinyanjuikelvin047@gmail.com">Kinyanjuikelvin047@gmail.com</a></p>
              <p><strong>GitHub Repository:</strong> <a href="https://github.com/Kelvin-ndungu/Agentic_RAG_Diabetes_Knowledge-ManagementSystem" target="_blank" rel="noopener noreferrer">View on GitHub</a></p>
            </div>
          </div>

          <div className="info-section">
            <h2>Educational Purpose</h2>
            <div className="educational-purpose">
              <p>
                This application has been developed for <strong>educational purposes</strong> to demonstrate the
                potential of Artificial Intelligence (AI) in creating natural language chat interfaces for knowledge
                management systems.
              </p>
              <p>The system showcases how AI can be integrated with knowledge bases to provide:</p>
              <ul>
                <li>Question-and-answer chatbots that understand natural language queries</li>
                <li>Semantic search capabilities that retrieve relevant information based on meaning, not just keywords</li>
                <li>Integration of structured knowledge bases with conversational AI interfaces</li>
                <li>Retrieval-Augmented Generation (RAG) systems for accurate, source-cited responses</li>
              </ul>
              <p>
                This project serves as a demonstration of how modern AI technologies can make complex medical knowledge
                more accessible through intuitive interfaces.
              </p>
            </div>
          </div>

          <div className="info-section disclaimer-section">
            <h2>Disclaimer & Terms</h2>
            <div className="disclaimer-content">
              <p>
                <strong>Source Material:</strong> The content presented in this application is derived from the
                <strong> Kenya National Clinical Guidelines for the Management of Diabetes Mellitus, 2nd Edition (2018)</strong>,
                produced by the National Diabetes Prevention and Control Program, Division of Non-communicable Diseases,
                Ministry of Health, Kenya.
              </p>
              <p>
                Any part of the original document may be freely reviewed, quoted, reproduced or translated in full or
                in part so long as the source is acknowledged. It is not for sale or for use in commercial purposes.
              </p>
              <p>
                <strong>Medical Disclaimer:</strong> These guidelines are intended for healthcare professionals and
                should be used in conjunction with clinical judgment and patient-specific considerations. The
                information provided is based on evidence available at the time of publication and may be subject to
                updates as new evidence emerges.
              </p>
              <p>
                <strong>Important:</strong> This application and the information it provides are <strong>not a substitute
                for professional medical advice, diagnosis, or treatment</strong>. Always seek the advice of qualified
                health providers with any questions regarding medical conditions. Never disregard professional medical
                advice or delay in seeking it because of something you have read or accessed through this application.
              </p>
              <p>
                <strong>AI Limitations:</strong> While this system uses advanced AI technologies, the responses are
                generated based on the source material and may not reflect the most current medical knowledge. Users
                should verify critical information with authoritative sources and consult healthcare professionals for
                medical decisions.
              </p>
            </div>
          </div>

          <div className="action-section">
            <Link to="/guidelines" className="start-button">
              Start Browsing Guidelines
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}

function findSectionByUrl(document, url) {
  if (!document || !url) return null

  for (const item of document.frontMatter || []) {
    if (item.url === url) return item
    if (item.sections) {
      const found = findInSectionsByUrl(item.sections, url)
      if (found) return found
    }
  }

  for (const chapter of document.chapters || []) {
    if (chapter.url === url) return chapter
    if (chapter.sections) {
      const found = findInSectionsByUrl(chapter.sections, url)
      if (found) return found
    }
  }

  return null
}

function findInSectionsByUrl(sections, url) {
  for (const section of sections) {
    if (section.url === url) return section
    if (section.subsections) {
      const found = findInSectionsByUrl(section.subsections, url)
      if (found) return found
    }
  }
  return null
}

export function DocumentViewer({ document }) {
  const location = useLocation()
  const currentUrl = location.pathname
  const fromChat = location.state && location.state.fromChat
  const sourceTitle = location.state && location.state.sourceTitle

  let section = findSectionByUrl(document, currentUrl)

  if (!section) {
    if (document.chapters && document.chapters.length > 0) {
      section = document.chapters[0]
    } else if (document.frontMatter && document.frontMatter.length > 0) {
      section = document.frontMatter[0]
    }
  }

  if (!section) {
    return (
      <div className="document-viewer">
        <div className="error-message">
          <h2>Section not found</h2>
          <p>The requested section could not be found.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="document-viewer">
      <article className="document-article" id={section.id}>
        {fromChat && (
          <div className="source-callout">
            <div className="source-callout-title">Opened From Chat</div>
            <div className="source-callout-subtitle">
              {sourceTitle ? `Source: ${sourceTitle}` : 'Navigated from a chat citation'}
            </div>
          </div>
        )}
        {section.breadcrumb && section.breadcrumb.length > 0 && (
          <nav className="breadcrumbs" aria-label="Breadcrumb">
            <ol>
              {section.breadcrumb.map((crumb, index) => (
                <li key={index}>
                  {index < section.breadcrumb.length - 1 ? <span>{crumb}</span> : <span className="current">{crumb}</span>}
                  {index < section.breadcrumb.length - 1 && <span className="separator"></span>}
                </li>
              ))}
            </ol>
          </nav>
        )}

        <header className="document-header">
          <h1 className="document-title" id={`${section.id}-title`}>
            {section.title}
          </h1>
        </header>

        {section.introContent && (
          <div className="intro-content">
            <MarkdownRenderer content={section.introContent.content} />
          </div>
        )}

        {section.content && !section.subsections && (!section.sections || section.sections.length === 0) && (
          <div className="main-content">
            <MarkdownRenderer content={removeDuplicateHeading(section.content, section.title)} />
          </div>
        )}

        {section.subsections && section.subsections.length > 0 && (
          <div className="subsections">
            {section.subsections.map((subsection) => (
              <div key={subsection.id} className="subsection" id={subsection.id}>
                <h2 className="subsection-title" id={`${subsection.id}-title`}>
                  {subsection.title}
                </h2>
                {subsection.introContent && (
                  <div className="intro-content">
                    <MarkdownRenderer content={subsection.introContent.content} />
                  </div>
                )}
                {subsection.content && <MarkdownRenderer content={removeDuplicateHeading(subsection.content, subsection.title)} />}
              </div>
            ))}
          </div>
        )}

        {section.sections && section.sections.length > 0 && (
          <div className="sections">
            {section.sections.map((subSection) => (
              <div key={subSection.id} className="section" id={subSection.id}>
                <h2 className="section-title" id={`${subSection.id}-title`}>
                  {subSection.title}
                </h2>
                {subSection.introContent && (
                  <div className="intro-content">
                    <MarkdownRenderer content={subSection.introContent.content} />
                  </div>
                )}
                {subSection.content && (!subSection.subsections || subSection.subsections.length === 0) && (
                  <MarkdownRenderer content={removeDuplicateHeading(subSection.content, subSection.title)} />
                )}
                {subSection.subsections && subSection.subsections.length > 0 && (
                  <div className="subsections">
                    {subSection.subsections.map((subsection) => (
                      <div key={subsection.id} className="subsection" id={subsection.id}>
                        <h3 className="subsection-title" id={`${subsection.id}-title`}>
                          {subsection.title}
                        </h3>
                        {subsection.introContent && (
                          <div className="intro-content">
                            <MarkdownRenderer content={subsection.introContent.content} />
                          </div>
                        )}
                        {subsection.content && <MarkdownRenderer content={removeDuplicateHeading(subsection.content, subsection.title)} />}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </article>
    </div>
  )
}

export function MarkdownRenderer({ content }) {
  if (!content) return null

  const processedContent = resolveImagePaths(content)

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          img: ({ node, ...props }) => (
            <img {...props} alt={props.alt || 'Image'} className="markdown-image" loading="lazy" />
          ),
          table: ({ node, ...props }) => (
            <div className="table-wrapper">
              <table {...props} />
            </div>
          ),
          h1: ({ node, ...props }) => <h1 className="markdown-h1" {...props} />,
          h2: ({ node, ...props }) => <h2 className="markdown-h2" {...props} />,
          h3: ({ node, ...props }) => <h3 className="markdown-h3" {...props} />,
          h4: ({ node, ...props }) => <h4 className="markdown-h4" {...props} />,
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  )
}

export function ChatInterface({ isOpen, onClose, initialQuery = '', isMobile = false, onWidthChange, isCentered = false }) {
  const {
    messages,
    sendMessage,
    loading,
    isStreaming,
    clearMessages,
    shouldAutoScroll,
    disableAutoScroll,
  } = useChat(initialQuery)

  const [width, setWidth] = useState(33.33)
  const [isDragging, setIsDragging] = useState(false)
  const chatRef = useRef(null)
  const startXRef = useRef(0)
  const startWidthRef = useRef(0)
  const isDraggingRef = useRef(false)

  useEffect(() => {
    if (isOpen && !isMobile) {
      if (isCentered) {
        setWidth(79)
        if (onWidthChange) {
          onWidthChange(79)
        }
      } else {
        setWidth(33.33)
        if (onWidthChange) {
          onWidthChange(33.33)
        }
      }
    }
  }, [isOpen, isMobile, onWidthChange, isCentered])

  const handleMouseMove = useCallback(
    (e) => {
      if (!isDraggingRef.current || isMobile || isCentered) return

      const windowWidth = window.innerWidth
      const deltaX = startXRef.current - e.clientX
      const deltaPercent = (deltaX / windowWidth) * 100
      let newWidth = startWidthRef.current + deltaPercent

      newWidth = Math.max(33.33, Math.min(50, newWidth))

      setWidth(newWidth)
      if (onWidthChange) {
        onWidthChange(newWidth)
      }
    },
    [isMobile, onWidthChange, isCentered]
  )

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false
    setIsDragging(false)
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  }, [handleMouseMove])

  const handleMouseDown = useCallback(
    (e) => {
      if (isMobile || isCentered) return
      e.preventDefault()
      e.stopPropagation()

      isDraggingRef.current = true
      setIsDragging(true)
      startXRef.current = e.clientX
      startWidthRef.current = width

      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    },
    [isMobile, isCentered, width, handleMouseMove, handleMouseUp]
  )

  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [handleMouseMove, handleMouseUp])

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
        className={`chat-interface ${isMobile ? 'mobile' : 'desktop'} ${isDragging ? 'dragging' : ''} ${isCentered ? 'centered' : ''}`}
        style={{ width: chatWidth }}
      >
        {!isMobile && !isCentered && <div className="chat-resizer" onMouseDown={handleMouseDown} />}
        <div className="chat-header">
          <div className="chat-header-text">
            <div className="chat-panel-label">Chat Panel</div>
            <h3>Ask about Diabetes Guidelines</h3>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="chat-clear" onClick={clearMessages} aria-label="Clear conversation" title="Clear conversation">
              Clear
            </button>
            <button className="chat-close" onClick={onClose} aria-label="Close chat">
              x
            </button>
          </div>
        </div>

        <MessageList
          messages={messages}
          loading={loading}
          isStreaming={isStreaming}
          shouldAutoScroll={shouldAutoScroll}
          disableAutoScroll={disableAutoScroll}
        />

        <ChatInput onSend={handleSend} disabled={loading} />
      </div>
    </>
  )
}

export function MessageList({ messages, loading, isStreaming, shouldAutoScroll, disableAutoScroll }) {
  const messagesEndRef = useRef(null)
  const messageListRef = useRef(null)
  const lastAutoScrollTimeRef = useRef(0)
  const autoScrollJustEnabledRef = useRef(false)
  const prevShouldAutoScrollRef = useRef(shouldAutoScroll)
  const touchStartRef = useRef({ x: 0, y: 0 })

  const scrollToMidpoint = () => {
    if (messageListRef.current) {
      const container = messageListRef.current
      const scrollTop = container.scrollHeight / 2 - container.clientHeight / 2

      lastAutoScrollTimeRef.current = Date.now()

      container.scrollTo({
        top: scrollTop,
        behavior: 'smooth',
      })
    }
  }

  const isNearBottom = () => {
    if (!messageListRef.current) return true
    const container = messageListRef.current
    const threshold = 100
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold
  }

  useEffect(() => {
    if (shouldAutoScroll && !prevShouldAutoScrollRef.current) {
      autoScrollJustEnabledRef.current = true
      setTimeout(() => {
        autoScrollJustEnabledRef.current = false
      }, 500)
    }

    prevShouldAutoScrollRef.current = shouldAutoScroll
  }, [shouldAutoScroll])

  useEffect(() => {
    const container = messageListRef.current
    if (!container) return

    const handleScroll = () => {
      if (autoScrollJustEnabledRef.current) {
        return
      }

      const timeSinceAutoScroll = Date.now() - lastAutoScrollTimeRef.current
      if (timeSinceAutoScroll < 150) {
        return
      }

      if (!isNearBottom()) {
        disableAutoScroll()
      }
    }

    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => container.removeEventListener('scroll', handleScroll)
  }, [disableAutoScroll])

  useEffect(() => {
    const container = messageListRef.current
    if (!container) return

    const handleTouchStart = (e) => {
      touchStartRef.current = {
        x: e.touches[0].clientX,
        y: e.touches[0].clientY,
      }
    }

    const handleTouchMove = (e) => {
      if (autoScrollJustEnabledRef.current) {
        return
      }

      const deltaY = e.touches[0].clientY - touchStartRef.current.y

      if (deltaY < -10) {
        if (!isNearBottom()) {
          disableAutoScroll()
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

  useEffect(() => {
    if (shouldAutoScroll) {
      scrollToMidpoint()
    }
  }, [messages, shouldAutoScroll, isStreaming])

  return (
    <div className="message-list" ref={messageListRef}>
      {messages.length === 0 ? (
        <div className="empty-state">
          <p>Ask a question about the diabetes guidelines to get started.</p>
        </div>
      ) : (
        messages.map((message, index) => <Message key={index} message={message} />)
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

function processNumberedReferences(content, sources) {
  if (!sources || sources.length === 0) {
    return content
  }

  const citationNumbers = new Set()
  const referencePattern = /\[(\d+)\](?!\()/g
  let match
  referencePattern.lastIndex = 0
  while ((match = referencePattern.exec(content)) !== null) {
    citationNumbers.add(parseInt(match[1], 10))
  }

  const sortedCitations = Array.from(citationNumbers).sort((a, b) => a - b)
  const citationMap = new Map()
  sortedCitations.forEach((citationNum, arrayIndex) => {
    if (arrayIndex < sources.length) {
      citationMap.set(citationNum, sources[arrayIndex])
    }
  })

  referencePattern.lastIndex = 0
  return content.replace(referencePattern, (matchText, num) => {
    const citationNum = parseInt(num, 10)
    const source = citationMap.get(citationNum)
    if (source) {
      const sourceUrl = source.url || '#'
      return `[${num}](${sourceUrl})`
    }
    return matchText
  })
}

export function Message({ message }) {
  const isStatus = message.isStatus || false
  const sources = message.sources || []

  const markdownComponents = {
    a: ({ node, href, children, ...props }) => {
      if (href && href.startsWith('/guidelines')) {
        return (
          <Link
            to={href}
            state={{ fromChat: true }}
            {...props}
          >
            {children}
          </Link>
        )
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
          {children}
        </a>
      )
    },
    h1: ({ node, ...props }) => <h1 className="markdown-h1" {...props} />,
    h2: ({ node, ...props }) => <h2 className="markdown-h2" {...props} />,
    h3: ({ node, ...props }) => <h3 className="markdown-h3" {...props} />,
    h4: ({ node, ...props }) => <h4 className="markdown-h4" {...props} />,
    table: ({ node, ...props }) => (
      <div className="table-wrapper">
        <table {...props} />
      </div>
    ),
    img: ({ node, ...props }) => (
      <img {...props} alt={props.alt || 'Image'} className="markdown-image" loading="lazy" />
    ),
  }

  return (
    <div className={`message ${message.role} ${isStatus ? 'status' : ''}`}>
      <div className="message-content">
        {isStatus ? (
          <div className="status-message">
            <em>{message.content}</em>
          </div>
        ) : (
          <div className="markdown-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {processNumberedReferences(message.content, sources)}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {sources && sources.length > 0 && !isStatus && (
        <div className="message-sources" id="message-sources">
          <div className="sources-title">Sources:</div>
          <ul className="sources-list">
            {sources.map((source, index) => {
              const sourceUrl = source.url || '#'
              const isInternal = sourceUrl.startsWith('/guidelines')

              return (
                <li key={index} className="source-item" id={`source-${index + 1}`}>
                  {isInternal ? (
                    <Link
                      to={sourceUrl}
                      state={{ fromChat: true, sourceTitle: source.title || `Source ${index + 1}` }}
                      className="source-link"
                    >
                      {index + 1}. {source.title || `Source ${index + 1}`}
                    </Link>
                  ) : (
                    <a href={sourceUrl} target="_blank" rel="noopener noreferrer" className="source-link">
                      {index + 1}. {source.title || `Source ${index + 1}`}
                    </a>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {message.timestamp && !isStatus && (
        <div className="message-timestamp">{new Date(message.timestamp).toLocaleTimeString()}</div>
      )}
    </div>
  )
}

export function ChatInput({ onSend, disabled }) {
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
        <button type="submit" className="chat-send-button" disabled={disabled || !input.trim()} aria-label="Send message">
          <span>-></span>
        </button>
      </form>
    </div>
  )
}
