import { Link } from 'react-router-dom'
import NavTree from '../navigation/NavTree'

export default function Sidebar({ document, isOpen, onClose, isMobile }) {
  if (!document) return null

  return (
    <>
      {/* Overlay for mobile */}
      {isMobile && isOpen && (
        <div 
          className="sidebar-overlay" 
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      
      <aside className={`sidebar ${isOpen ? 'open' : ''} ${isMobile ? 'mobile' : ''}`}>
        <div className="sidebar-header">
          <Link to="/" className="sidebar-title" onClick={isMobile ? onClose : undefined}>
            <h2>{document.title}</h2>
          </Link>
          {isMobile && (
            <button 
              className="sidebar-close"
              onClick={onClose}
              aria-label="Close sidebar"
            >
              ×
            </button>
          )}
        </div>
        
        <nav className="sidebar-nav">
          {/* Chapters - Show first */}
          {document.chapters && document.chapters.length > 0 && (
            <div className="nav-section">
              <NavTree 
                items={document.chapters} 
                onNavigate={isMobile ? onClose : undefined}
              />
            </div>
          )}
          
          {/* Front Matter - Show after chapters */}
          {document.frontMatter && document.frontMatter.length > 0 && (
            <div className="nav-section">
              <NavTree 
                items={document.frontMatter} 
                onNavigate={isMobile ? onClose : undefined}
              />
            </div>
          )}
        </nav>
      </aside>
    </>
  )
}

