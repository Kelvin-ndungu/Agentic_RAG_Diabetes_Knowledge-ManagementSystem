import { useLocation } from 'react-router-dom'
import MarkdownRenderer from './MarkdownRenderer'
import { removeDuplicateHeading } from '../../utils/imagePathResolver'

// Helper function to find section by URL
function findSectionByUrl(document, url) {
  if (!document || !url) return null
  
  // Search in front matter
  for (const item of document.frontMatter || []) {
    if (item.url === url) return item
    if (item.sections) {
      const found = findInSectionsByUrl(item.sections, url)
      if (found) return found
    }
  }
  
  // Search in chapters
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

// Helper function to find section by ID (fallback)
function findSectionById(document, id) {
  if (!document || !id) return null
  
  for (const item of document.frontMatter || []) {
    if (item.id === id) return item
    if (item.sections) {
      const found = findInSectionsById(item.sections, id)
      if (found) return found
    }
  }
  
  for (const chapter of document.chapters || []) {
    if (chapter.id === id) return chapter
    if (chapter.sections) {
      const found = findInSectionsById(chapter.sections, id)
      if (found) return found
    }
  }
  
  return null
}

function findInSectionsById(sections, id) {
  for (const section of sections) {
    if (section.id === id) return section
    if (section.subsections) {
      const found = findInSectionsById(section.subsections, id)
      if (found) return found
    }
  }
  return null
}

export default function DocumentViewer({ document }) {
  const location = useLocation()
  const currentUrl = location.pathname

  // Find the section by URL
  let section = findSectionByUrl(document, currentUrl)

  // If no section found, show first chapter or front matter
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
        {/* Breadcrumbs */}
        {section.breadcrumb && section.breadcrumb.length > 0 && (
          <nav className="breadcrumbs" aria-label="Breadcrumb">
            <ol>
              {section.breadcrumb.map((crumb, index) => (
                <li key={index}>
                  {index < section.breadcrumb.length - 1 ? (
                    <span>{crumb}</span>
                  ) : (
                    <span className="current">{crumb}</span>
                  )}
                  {index < section.breadcrumb.length - 1 && <span className="separator">›</span>}
                </li>
              ))}
            </ol>
          </nav>
        )}

        {/* Title */}
        <header className="document-header">
          <h1 className="document-title" id={`${section.id}-title`}>
            {section.title}
          </h1>
        </header>

        {/* Intro Content (orphan sections) */}
        {section.introContent && (
          <div className="intro-content">
            <MarkdownRenderer content={section.introContent.content} />
          </div>
        )}

        {/* Main Content - Only render if section has NO children to avoid duplication */}
        {section.content && !section.subsections && (!section.sections || section.sections.length === 0) && (
          <div className="main-content">
            <MarkdownRenderer content={removeDuplicateHeading(section.content, section.title)} />
          </div>
        )}

        {/* Subsections */}
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
                {/* Subsection content - always render since subsections are leaf nodes */}
                {subsection.content && (
                  <MarkdownRenderer content={removeDuplicateHeading(subsection.content, subsection.title)} />
                )}
              </div>
            ))}
          </div>
        )}

        {/* Sections (for chapters) */}
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
                {/* Only render content if section has NO subsections to avoid duplication */}
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
                        {/* Subsection content - always render since subsections are leaf nodes */}
                        {subsection.content && (
                          <MarkdownRenderer content={removeDuplicateHeading(subsection.content, subsection.title)} />
                        )}
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

