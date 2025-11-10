import { resolveImagePaths, extractImages } from '../utils/imagePathResolver'

/**
 * Test component to verify image path resolution
 */
export default function ImageTest() {
  // Test markdown content with image paths
  const testContent = `
    Here's some content with images:
    
    ![Ministry Logo](images/picture_000_page_1.png)
    ![Another Image](images/picture_012_page_17.png)
    ![Test Image](images/picture_034_page_61.png)
  `

  const resolvedContent = resolveImagePaths(testContent)
  const extractedImages = extractImages(testContent)

  return (
    <div className="image-test">
      <h3>Image Path Resolution Test</h3>
      
      <div className="test-section">
        <h4>Original Content:</h4>
        <pre>{testContent}</pre>
      </div>

      <div className="test-section">
        <h4>Resolved Content:</h4>
        <pre>{resolvedContent}</pre>
      </div>

      <div className="test-section">
        <h4>Extracted Images ({extractedImages.length}):</h4>
        <ul>
          {extractedImages.map((img, idx) => (
            <li key={idx}>
              <strong>Alt:</strong> {img.alt} | <strong>Path:</strong> {img.path}
            </li>
          ))}
        </ul>
      </div>

      <div className="test-section">
        <h4>Image Loading Test:</h4>
        <div className="image-grid">
          {extractedImages.map((img, idx) => (
            <div key={idx} className="image-item">
              <img 
                src={img.path}
                alt={img.alt}
                onError={(e) => {
                  e.target.style.border = '3px solid red'
                  e.target.alt = 'FAILED TO LOAD'
                }}
                onLoad={() => {
                  console.log(`✓ Image loaded: ${img.path}`)
                }}
              />
              <p>{img.alt}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

