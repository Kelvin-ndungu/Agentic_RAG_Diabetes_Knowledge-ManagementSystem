import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useState, useEffect, useCallback } from 'react'
import { useDocument } from './state'
import { Header, Sidebar, HomePage, DocumentViewer, ChatInterface } from './ui'
import './App.css'

function App() {
  const { document, loading, error } = useDocument()
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768)
  const [chatOpen, setChatOpen] = useState(false)
  const [chatInitialQuery, setChatInitialQuery] = useState('')
  const [chatWidth, setChatWidth] = useState(33.33)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)
  const [viewMode, setViewMode] = useState('normal')

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev)
  }, [])

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768
      const wasMobile = isMobile
      setIsMobile(mobile)
      if (!mobile && wasMobile) {
        setSidebarOpen(true)
      } else if (mobile && !wasMobile) {
        setSidebarOpen(false)
      }
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [isMobile])

  const handleSearchClick = (query = '') => {
    setChatInitialQuery(query)
    setChatOpen(true)
    if (isMobile) {
      setSidebarOpen(false)
    }
  }

  const handleViewModeChange = (mode) => {
    setViewMode(mode)
    if (mode === 'chat-only' || mode === 'chat-alone') {
      setChatOpen(true)
      setSidebarOpen(false)
    } else if (mode === 'document-only') {
      setChatOpen(false)
      setSidebarOpen(true)
    } else if (mode === 'normal') {
      setSidebarOpen(window.innerWidth >= 768)
    }
  }

  if (loading) {
    return (
      <div className="app">
        <div className="loading-container">
          <h2>Loading document...</h2>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <div className="error-container">
          <h2>Error loading document</h2>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  if (!document || !document.document) {
    return (
      <div className="app">
        <div className="error-container">
          <h2>No document data available</h2>
        </div>
      </div>
    )
  }

  const showSidebar = (viewMode === 'normal' || viewMode === 'document-only') && sidebarOpen
  const showContent = viewMode === 'normal' || viewMode === 'document-only'
  const showChat = chatOpen && (viewMode === 'normal' || viewMode === 'chat-only' || viewMode === 'chat-alone')
  const isChatCentered = viewMode === 'chat-only' || viewMode === 'chat-alone'

  return (
    <BrowserRouter>
      <div className={`app view-mode-${viewMode}`}>
        <Header
          onMenuClick={toggleSidebar}
          onSearchClick={handleSearchClick}
          isMobile={isMobile}
          chatOpen={chatOpen}
          viewMode={viewMode}
          onViewModeChange={handleViewModeChange}
        />

        <div className="main-container">
          {showSidebar && (
            <Sidebar document={document.document} isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} isMobile={isMobile} />
          )}

          {showContent && (
            <div
              className={`content-wrapper ${!sidebarOpen && !isMobile ? 'sidebar-closed' : ''}`}
              style={
                chatOpen && !isMobile && viewMode === 'normal'
                  ? {
                      width: `calc(${100 - chatWidth}vw - 240px)`,
                    }
                  : {}
              }
            >
              <div className="content-panel-header">
                <div className="panel-title">Original Guidelines</div>
                <div className="panel-subtitle">Browse the source document</div>
              </div>
              <main className={`content-area ${sidebarOpen && isMobile ? 'sidebar-open' : ''}`}>
                <Routes>
                  <Route path="/" element={<HomePage document={document.document} />} />
                  <Route path="/guidelines/*" element={<DocumentViewer document={document.document} />} />
                </Routes>
              </main>
            </div>
          )}

          {showChat && (
            <ChatInterface
              isOpen={chatOpen}
              onClose={() => setChatOpen(false)}
              initialQuery={chatInitialQuery}
              isMobile={isMobile}
              onWidthChange={setChatWidth}
              isCentered={isChatCentered}
            />
          )}
        </div>
      </div>
    </BrowserRouter>
  )
}

export default App
