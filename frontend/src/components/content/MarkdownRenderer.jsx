import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { resolveImagePaths } from '../../utils/imagePathResolver'

export default function MarkdownRenderer({ content }) {
  if (!content) return null

  // Fix image paths: images/picture_XXX.png -> /images/picture_XXX.png
  const processedContent = resolveImagePaths(content)

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          img: ({ node, ...props }) => (
            <img 
              {...props} 
              alt={props.alt || 'Image'}
              className="markdown-image"
              loading="lazy"
            />
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

