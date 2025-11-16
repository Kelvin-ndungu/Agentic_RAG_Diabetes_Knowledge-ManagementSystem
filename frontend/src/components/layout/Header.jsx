import { useState } from 'react'

export default function Header({ onMenuClick, onSearchClick, isMobile, chatOpen }) {
  const [searchQuery, setSearchQuery] = useState('')

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      onSearchClick(searchQuery.trim())
      setSearchQuery('')
    } else {
      // If empty, just open chat
      onSearchClick('')
    }
  }

  const handleSearchClick = () => {
    if (isMobile) {
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
        <button 
          className="menu-button"
          onClick={onMenuClick}
          aria-label="Toggle navigation menu"
        >
          <span className="hamburger-icon">☰</span>
        </button>

        <form className="search-form-centered" onSubmit={handleSearchSubmit}>
          <input
            type="text"
            className="search-input-centered"
            placeholder={chatOpen ? "Use the chat interface to ask questions..." : "Ask any question in plain language..."}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={handleInputFocus}
            disabled={chatOpen}
            style={{ 
              opacity: chatOpen ? 0.5 : 1,
              cursor: chatOpen ? 'not-allowed' : 'text'
            }}
          />
        </form>
      </div>
    </header>
  )
}

