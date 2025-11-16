import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Link } from 'react-router-dom'

/**
 * Process numbered references [1], [2], etc. and convert them to clickable links
 * that point to the corresponding source URL.
 * 
 * Note: The citation numbers [1], [2] refer to the original source order.
 * The sources array contains only cited sources, so we need to find the source
 * that matches the citation number. We extract all citation numbers from the
 * content and map them to sources in the order they appear.
 */
function processNumberedReferences(content, sources) {
  if (!sources || sources.length === 0) {
    return content
  }
  
  // Extract all unique citation numbers from content
  const citationNumbers = new Set()
  const referencePattern = /\[(\d+)\](?!\()/g
  let match
  // Reset regex lastIndex to ensure we scan from the beginning
  referencePattern.lastIndex = 0
  while ((match = referencePattern.exec(content)) !== null) {
    citationNumbers.add(parseInt(match[1], 10))
  }
  
  // Sort citation numbers to create ordered mapping
  // Map citation numbers to sources in the order they appear in sources array
  const sortedCitations = Array.from(citationNumbers).sort((a, b) => a - b)
  const citationMap = new Map()
  sortedCitations.forEach((citationNum, arrayIndex) => {
    if (arrayIndex < sources.length) {
      citationMap.set(citationNum, sources[arrayIndex])
    }
  })
  
  // Replace numbered references with markdown links
  // Reset regex for replacement
  referencePattern.lastIndex = 0
  return content.replace(referencePattern, (match, num) => {
    const citationNum = parseInt(num, 10)
    const source = citationMap.get(citationNum)
    if (source) {
      const sourceUrl = source.url || '#'
      // Create a markdown link that will be rendered by ReactMarkdown
      return `[${num}](${sourceUrl})`
    }
    return match // Return original if source not found
  })
}

export default function Message({ message }) {
  const isUser = message.role === 'user'
  const isStatus = message.isStatus || false
  const sources = message.sources || []
  
  // Custom components for markdown rendering
  const markdownComponents = {
    // Use React Router Link for internal routes, regular links for external
    a: ({ node, href, children, ...props }) => {
      // Check if it's an internal route (starts with /guidelines)
      if (href && href.startsWith('/guidelines')) {
        // Use React Router Link for internal navigation (keeps chat open)
        // Prevent default navigation behavior to keep chat visible
        return (
          <Link 
            to={href} 
            {...props}
            onClick={(e) => {
              // Keep chat open - don't prevent default, but ensure smooth navigation
              // The Link component will handle navigation while keeping the app state
            }}
          >
            {children}
          </Link>
        )
      }
      // External links open in new tab
      return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>
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
      <img 
        {...props} 
        alt={props.alt || 'Image'}
        className="markdown-image"
        loading="lazy"
      />
    ),
  }
  
  return (
    <div className={`message ${message.role} ${isStatus ? 'status' : ''}`}>
      <div className="message-content">
        {isStatus ? (
          // Status messages - plain text with italic styling
          <div className="status-message">
            <em>{message.content}</em>
          </div>
        ) : (
          // Regular messages - render markdown with numbered reference support
          <div className="markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {processNumberedReferences(message.content, sources)}
            </ReactMarkdown>
          </div>
        )}
      </div>
      
      {/* Sources section */}
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
                      className="source-link"
                      onClick={(e) => {
                        // Keep chat open - Link will handle navigation smoothly
                        // The router will update the URL but the chat interface should remain visible
                      }}
                    >
                      {index + 1}. {source.title || `Source ${index + 1}`}
                    </Link>
                  ) : (
                    <a 
                      href={sourceUrl} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="source-link"
                    >
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
        <div className="message-timestamp">
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      )}
    </div>
  )
}

