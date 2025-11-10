import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Link } from 'react-router-dom'

/**
 * Process numbered references [1], [2], etc. and convert them to clickable links
 * that point to the corresponding source URL (same as in sources section).
 */
function processNumberedReferences(content, sources) {
  if (!sources || sources.length === 0) {
    return content
  }
  
  // Pattern to match numbered references like [1], [2], [10], etc.
  // But avoid matching markdown links [text](url) or images ![alt](url)
  // We want to match standalone [number] patterns (not followed by parentheses)
  // Using a simpler pattern that checks the character after ]
  const referencePattern = /\[(\d+)\](?!\()/g
  
  return content.replace(referencePattern, (match, num) => {
    const index = parseInt(num, 10) - 1 // Convert to 0-based index
    if (index >= 0 && index < sources.length) {
      const source = sources[index]
      const sourceUrl = source.url || '#'
      // Create a markdown link that will be rendered by ReactMarkdown
      return `[${num}](${sourceUrl})`
    }
    return match // Return original if index is invalid
  })
}

export default function Message({ message }) {
  const isUser = message.role === 'user'
  const isStatus = message.isStatus || false
  const sources = message.sources || []
  
  // Custom components for markdown rendering
  const markdownComponents = {
    // Use React Router Link for internal routes, regular links for external
    a: ({ node, href, ...props }) => {
      // Check if it's an internal route (starts with /guidelines)
      if (href && href.startsWith('/guidelines')) {
        // Use React Router Link for internal navigation (keeps chat open)
        return <Link to={href} {...props} />
      }
      // External links open in new tab
      return <a href={href} target="_blank" rel="noopener noreferrer" {...props} />
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

