import { useState, useEffect } from 'react';
import documentData from '../data/document_structure.json';

/**
 * Custom hook to load and manage document structure
 * Provides document data, loading state, and helper functions
 */
export function useDocument() {
  const [document, setDocument] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    try {
      // Simulate async loading (in future, could fetch from API)
      setTimeout(() => {
        setDocument(documentData);
        setLoading(false);
      }, 100);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }, []);

  /**
   * Find a section by its ID
   * @param {string} id - Section ID to find
   * @returns {object|null} - Section object or null if not found
   */
  const findSectionById = (id) => {
    if (!document) return null;
    
    // Search in front matter
    for (const item of document.document.frontMatter) {
      if (item.id === id) return item;
      if (item.sections) {
        const found = findInSections(item.sections, id);
        if (found) return found;
      }
    }
    
    // Search in chapters
    for (const chapter of document.document.chapters) {
      if (chapter.id === id) return chapter;
      if (chapter.sections) {
        const found = findInSections(chapter.sections, id);
        if (found) return found;
      }
    }
    
    return null;
  };

  /**
   * Recursively search in sections and subsections
   */
  const findInSections = (sections, id) => {
    for (const section of sections) {
      if (section.id === id) return section;
      if (section.subsections) {
        const found = findInSections(section.subsections, id);
        if (found) return found;
      }
    }
    return null;
  };

  /**
   * Find a section by URL slug
   * @param {string} slug - URL slug to find
   * @returns {object|null} - Section object or null if not found
   */
  const findSectionBySlug = (slug) => {
    if (!document) return null;
    
    // Search in front matter
    for (const item of document.document.frontMatter) {
      if (item.slug === slug) return item;
      if (item.sections) {
        const found = findInSectionsBySlug(item.sections, slug);
        if (found) return found;
      }
    }
    
    // Search in chapters
    for (const chapter of document.document.chapters) {
      if (chapter.slug === slug) return chapter;
      if (chapter.sections) {
        const found = findInSectionsBySlug(chapter.sections, slug);
        if (found) return found;
      }
    }
    
    return null;
  };

  /**
   * Recursively search by slug in sections
   */
  const findInSectionsBySlug = (sections, slug) => {
    for (const section of sections) {
      if (section.slug === slug) return section;
      if (section.subsections) {
        const found = findInSectionsBySlug(section.subsections, slug);
        if (found) return found;
      }
    }
    return null;
  };

  return { 
    document, 
    loading, 
    error,
    findSectionById,
    findSectionBySlug
  };
}

