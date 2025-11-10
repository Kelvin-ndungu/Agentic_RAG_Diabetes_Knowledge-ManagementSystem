import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import NavTree from './NavTree'

export default function NavItem({ item, level = 0, onNavigate }) {
  const [isExpanded, setIsExpanded] = useState(level === 0) // Auto-expand first level
  const location = useLocation()
  
  // Determine if this item is active
  const isActive = location.pathname === item.url || 
                   location.pathname.includes(item.url)

  const hasChildren = (item.sections && item.sections.length > 0) ||
                     (item.subsections && item.subsections.length > 0)
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
        <Link 
          to={item.url}
          className="nav-item-link"
          onClick={handleClick}
        >
          <span className="nav-title">{item.title}</span>
        </Link>
        
        {hasChildren && (
          <button 
            className="nav-toggle"
            onClick={handleToggle}
            aria-label={isExpanded ? 'Collapse' : 'Expand'}
            aria-expanded={isExpanded}
          >
            {isExpanded ? '▼' : '▶'}
          </button>
        )}
      </div>
      
      {hasChildren && isExpanded && (
        <NavTree 
          items={children} 
          level={level + 1}
          onNavigate={onNavigate}
        />
      )}
    </li>
  )
}

