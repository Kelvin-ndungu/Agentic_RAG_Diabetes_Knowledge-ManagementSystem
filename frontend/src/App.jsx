import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { useDocument } from './hooks/useDocument'
import Header from './components/layout/Header'
import Sidebar from './components/layout/Sidebar'
import HomePage from './components/HomePage'
import DocumentViewer from './components/content/DocumentViewer'
import ChatInterface from './components/chat/ChatInterface'
import './App.css'

function App() {
  const { document, loading, error } = useDocument()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [chatInitialQuery, setChatInitialQuery] = useState('')
  const [chatWidth, setChatWidth] = useState(33.33)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768
      setIsMobile(mobile)
      // Auto-close sidebar on mobile when switching to desktop
      if (!mobile && sidebarOpen) {
        setSidebarOpen(false)
      }
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [sidebarOpen])

  const handleSearchClick = (query = '') => {
    setChatInitialQuery(query)
    setChatOpen(true)
    // Close sidebar on mobile when opening chat
    if (isMobile) {
      setSidebarOpen(false)
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

  return (
    <BrowserRouter>
      <div className="app">
        <Header 
          onMenuClick={() => setSidebarOpen(!sidebarOpen)}
          onSearchClick={handleSearchClick}
          isMobile={isMobile}
        />
        
        <div className="main-container">
          <Sidebar 
            document={document.document}
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
            isMobile={isMobile}
          />
          
          <div className="content-wrapper" style={chatOpen && !isMobile ? { 
            width: `calc(${100 - chatWidth}vw - 240px)`
          } : {}}>
            <main 
              className={`content-area ${sidebarOpen && isMobile ? 'sidebar-open' : ''}`}
            >
              <Routes>
                <Route path="/" element={<HomePage document={document.document} />} />
                <Route path="/guidelines/*" element={
                  <DocumentViewer document={document.document} />
                } />
              </Routes>
            </main>
          </div>

          {chatOpen && (
            <ChatInterface 
              isOpen={chatOpen}
              onClose={() => setChatOpen(false)}
              initialQuery={chatInitialQuery}
              isMobile={isMobile}
              onWidthChange={setChatWidth}
            />
          )}
        </div>
      </div>
    </BrowserRouter>
  )
}

export default App
