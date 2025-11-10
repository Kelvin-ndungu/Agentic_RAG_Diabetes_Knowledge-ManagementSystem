/**
 * Convert markdown image paths to web-accessible paths
 * 
 * @param {string} content - Markdown content with image references
 * @returns {string} - Content with resolved image paths
 * 
 * Example:
 * Input:  "![alt text](images/picture_012_page_17.png)"
 * Output: "![alt text](/images/picture_012_page_17.png)"
 */
export function resolveImagePaths(content) {
  if (!content || typeof content !== 'string') {
    return content;
  }

  // Replace relative image paths with absolute paths
  // Pattern: ![alt](images/filename.png) -> ![alt](/images/filename.png)
  return content.replace(
    /!\[([^\]]*)\]\(images\/([^\)]+)\)/g,
    '![$1](/images/$2)'
  );
}

/**
 * Extract all image paths from markdown content
 * 
 * @param {string} content - Markdown content
 * @returns {Array<{alt: string, path: string}>} - Array of image objects
 */
export function extractImages(content) {
  if (!content || typeof content !== 'string') {
    return [];
  }

  const images = [];
  const regex = /!\[([^\]]*)\]\(images\/([^\)]+)\)/g;
  let match;

  while ((match = regex.exec(content)) !== null) {
    images.push({
      alt: match[1] || 'Image',
      path: `/images/${match[2]}`
    });
  }

  return images;
}

/**
 * Check if an image path exists (client-side check)
 * Note: This is a basic check. Full validation requires server-side verification
 * 
 * @param {string} imagePath - Path to image (e.g., "/images/picture_012_page_17.png")
 * @returns {boolean} - True if path looks valid
 */
export function isValidImagePath(imagePath) {
  if (!imagePath) return false;
  
  // Check if path starts with /images/ and ends with .png
  const pattern = /^\/images\/picture_\d+_page_\d+\.png$/;
  return pattern.test(imagePath);
}

/**
 * Remove duplicate heading from markdown content if it matches the section title
 * 
 * @param {string} content - Markdown content
 * @param {string} title - Section title
 * @returns {string} - Content with duplicate heading removed
 */
export function removeDuplicateHeading(content, title) {
  if (!content || !title) return content;
  
  // Extract the title text (remove number if present, keep just the text part)
  const titleText = title.replace(/^\d+\.\d+(\.\d+)*\.?\s*/, '').trim();
  
  // Patterns to match markdown headings that might match the title
  const headingPatterns = [
    // Matches ### 1.2.1. Title or ### Title
    new RegExp(`^###\\s+${title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\n+`, 'i'),
    // Matches ### Title (without number)
    new RegExp(`^###\\s+${titleText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\n+`, 'i'),
    // Matches ## Title
    new RegExp(`^##\\s+${title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\n+`, 'i'),
    // Matches # Title
    new RegExp(`^#\\s+${title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\n+`, 'i'),
  ];
  
  let cleanedContent = content;
  
  // Try each pattern and remove the first match
  for (const pattern of headingPatterns) {
    if (pattern.test(cleanedContent)) {
      cleanedContent = cleanedContent.replace(pattern, '');
      break; // Only remove the first match
    }
  }
  
  return cleanedContent.trim();
}

