// This script is a Node.js implementation of a resume parser that extracts sections from PDF files.
const fs = require('fs');
const path = require('path');

// Import a simpler PDF text extraction library that won't conflict with PDF.js
const pdfParse = require('pdf-parse/lib/pdf-parse.js');  // Use direct import to avoid conflicts

// Common section headers in resumes
const FALLBACK_SECTION_HEADERS = {
  CONTACT: ['contact', 'contact information', 'contact details', 'personal details'],
  SUMMARY: ['summary', 'professional summary', 'profile', 'about me', 'career objective', 'objective'],
  EDUCATION: ['education', 'academic background', 'academic qualifications', 'qualifications', 'academic', 'university', 'college'],
  EXPERIENCE: ['experience', 'work experience', 'employment', 'work history', 'professional experience', 'internship', 'internships'],
  SKILLS: ['skills', 'technical skills', 'core competencies', 'competencies', 'expertise', 'abilities'],
  PROJECTS: ['projects', 'project experience', 'key projects', 'personal projects', 'academic projects'],
  CERTIFICATIONS: ['certifications', 'certification', 'certificates', 'professional certifications'],
  LANGUAGES: ['languages', 'language proficiency', 'spoken languages'],
  INTERESTS: ['interests', 'hobbies', 'activities', 'personal interests'],
};

/**
 * Extract text from PDF using a direct approach that won't conflict with PDF.js
 * @param {string} pdfPath - Path to the PDF file
 * @returns {Promise<string>} - Extracted text
 */
async function extractTextFromPDF(pdfPath) {
  try {
    // Use a more robust approach to read the PDF file
    const dataBuffer = fs.readFileSync(pdfPath);
    
    // Try different approaches to extract text
    try {
      // First attempt with pdf-parse directly
      const data = await pdfParse(dataBuffer);
      return data.text;
    } catch (parseError) {
      console.error(`Standard extraction failed, trying alternative method: ${parseError.message}`);
      
      // Alternative approach - use the main parser's PDF.js if available
      try {
        // Use pdfjsLib directly if it's been imported by the main parser
        const pdfjsLib = global.pdfjsLib || require('pdfjs-dist/legacy/build/pdf.js');
        const doc = await pdfjsLib.getDocument({data: dataBuffer}).promise;
        
        let text = '';
        for (let i = 1; i <= doc.numPages; i++) {
          const page = await doc.getPage(i);
          const content = await page.getTextContent();
          text += content.items.map(item => item.str).join(' ') + '\n';
        }
        
        return text;
      } catch (pdfJsError) {
        // Final fallback - try to use a simple extraction
        console.error(`PDF.js extraction failed, using very basic extraction: ${pdfJsError.message}`);
        
        // Very basic text extraction (may not work well but better than nothing)
        let text = '';
        for (let i = 0; i < dataBuffer.length; i++) {
          if (dataBuffer[i] >= 32 && dataBuffer[i] < 127) {
            text += String.fromCharCode(dataBuffer[i]);
          }
        }
        
        return text;
      }
    }
  } catch (error) {
    console.error(`Error extracting text from PDF (fallback): ${error.message}`);
    throw new Error('Failed to extract text from PDF');
  }
}

/**
 * Identify section boundaries in resume text
 * @param {string} text - Resume text
 * @returns {Object} - Object with section names as keys and their content as values
 */
function identifySections(text) {
  if (!text) return {};
  
  // Normalize line endings and split text into lines
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  
  // Create a flat array of all section header keywords
  const allHeaders = Object.entries(FALLBACK_SECTION_HEADERS)
    .reduce((acc, [type, keywords]) => {
      // Add the type itself as a keyword too
      return acc.concat(keywords.map(k => ({keyword: k, type})));
    }, []);
  
  // Find potential section headers
  const headerLines = [];
  lines.forEach((line, index) => {
    // Clean the line
    const cleanLine = line.trim().toLowerCase();
    
    // Skip empty lines
    if (!cleanLine) return;
    
    // Check if this line is a potential section header
    // 1. Is it ALL CAPS and short?
    const isAllCaps = line.trim() === line.trim().toUpperCase() && 
                      line.trim().length > 2 && 
                      line.trim().length < 30;
    
    // 2. Does it contain a known section keyword?
    const matchingKeyword = allHeaders.find(({keyword}) => 
      cleanLine === keyword || 
      cleanLine === keyword + ':' ||
      cleanLine.startsWith(keyword + ' ') ||
      cleanLine.startsWith(keyword + ':')
    );
    
    // 3. Is it a short line with a colon?
    const hasColon = cleanLine.includes(':') && cleanLine.length < 30;
    
    // If any criteria match, consider it a header
    if (isAllCaps || matchingKeyword || hasColon) {
      // Determine section type
      let sectionType = 'OTHER';
      
      if (matchingKeyword) {
        sectionType = matchingKeyword.type;
      } else {
        // Try to infer section type from content
        for (const [type, keywords] of Object.entries(FALLBACK_SECTION_HEADERS)) {
          if (keywords.some(keyword => cleanLine.includes(keyword))) {
            sectionType = type;
            break;
          }
        }
      }
      
      headerLines.push({ 
        index, 
        text: line.trim(),
        type: sectionType,
        confidence: isAllCaps ? 0.8 : (matchingKeyword ? 0.9 : 0.6)
      });
    }
  });
  
  // Sort headers by position (top to bottom)
  headerLines.sort((a, b) => a.index - b.index);
  
  // Determine section boundaries
  const sections = {};
  headerLines.forEach((header, idx) => {
    // Get content until next section header or end of text
    const startIndex = header.index + 1;
    const endIndex = idx < headerLines.length - 1 ? headerLines[idx + 1].index : lines.length;
    
    // Extract section content
    const content = lines.slice(startIndex, endIndex)
                        .filter(line => line.trim()) // Remove empty lines
                        .join('\n');
    
    // Store section
    if (content.trim()) {
      sections[header.type] = sections[header.type] 
        ? sections[header.type] + '\n\n' + content 
        : content;
    }
  });
  
  return {
    sections,
    headerLines
  };
}

/**
 * Parse a resume PDF using the fallback text-based approach
 * @param {string} pdfPath - Path to the PDF file
 * @returns {Promise<Object>} - Parsed resume data
 */
async function parseResumeTextBased(pdfPath) {
  try {
    console.log(`Using fallback text-based parser for: ${pdfPath}`);
    const startTime = Date.now();
    
    // Extract text from PDF
    const text = await extractTextFromPDF(pdfPath);
    if (!text) {
      throw new Error('Failed to extract text from PDF');
    }
    
    // Identify sections
    const { sections, headerLines } = identifySections(text);
    
    // Convert headings to the same format used by the main parser
    const headings = headerLines.map((header, index) => ({
      page: 0, // We don't have page info in the text-based approach
      text: header.text,
      x0: 0,
      y0: header.index,
      x1: 100,
      y1: header.index + 1,
      type: header.type,
      column: "center", // We don't detect columns in the text-based approach
      confidence: header.confidence
    }));
    
    // Convert sections to the same format used by the main parser
    const sectionTexts = {};
    Object.entries(sections).forEach(([sectionType, content]) => {
      sectionTexts[sectionType] = `--- TEXT-BASED EXTRACTION ---\n${content}`;
    });
    
    // Calculate processing time
    const processingTime = (Date.now() - startTime) / 1000;
    
    return {
      pdfPath,
      headings,
      sectionTexts,
      processingTime,
      usedFallback: true,
      fallbackHeaderCount: headerLines.length,
      // Add empty or default values for compatibility with main parser's result structure
      columnDivider: 0,
      fontStatistics: {
        mean: 0,
        median: 0,
        std: 0,
        min: 0,
        max: 0,
        mostCommon: 0
      }
    };
  } catch (error) {
    console.error(`Error in fallback parser: ${error.message}`);
    throw error; // Re-throw so the main parser can catch it
  }
}

// Enhanced version specifically for student resumes
function identifyStudentSections(text) {
  if (!text) return {};
  
  // Student resume specific keywords
  const STUDENT_SECTIONS = {
    ...FALLBACK_SECTION_HEADERS,
    EDUCATION: [...FALLBACK_SECTION_HEADERS.EDUCATION, 'b.tech', 'bachelor', 'b tech', 'intermediate', 'secondary', 'high school'],
    INTERNSHIPS: ['internships', 'internship experience', 'virtual internship'],
    COURSEWORK: ['coursework', 'relevant coursework', 'courses'],
    POSITIONS: ['positions of responsibility', 'leadership', 'extracurricular'],
    ACHIEVEMENTS: ['achievements', 'awards', 'honors', 'recognitions']
  };
  
  // Normalize line endings and split text into lines
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  
  // Create a flat array of all section header keywords
  const allHeaders = Object.entries(STUDENT_SECTIONS)
    .reduce((acc, [type, keywords]) => {
      return acc.concat(keywords.map(k => ({keyword: k, type})));
    }, []);
  
  // Find potential section headers
  const headerLines = [];
  lines.forEach((line, index) => {
    // Clean the line
    const cleanLine = line.trim().toLowerCase();
    
    // Skip empty lines
    if (!cleanLine) return;
    
    // Very strong header signals
    const isAllCaps = line.trim() === line.trim().toUpperCase() && 
                     line.trim().length > 2 && 
                     line.trim().length < 30;
                     
    // Check for known section keywords
    const matchingKeyword = allHeaders.find(({keyword}) => 
      cleanLine === keyword || 
      cleanLine === keyword + ':' ||
      cleanLine.startsWith(keyword + ' ') ||
      cleanLine.startsWith(keyword + ':') ||
      cleanLine.includes(keyword)
    );
    
    // Is it a short line with a colon or at position 0?
    const hasColon = cleanLine.includes(':') && cleanLine.length < 30;
    const isAtLeftMargin = line.trim() === line; // No leading whitespace
    
    // If any criteria match, consider it a header
    if (isAllCaps || matchingKeyword || (hasColon && isAtLeftMargin)) {
      // Determine section type
      let sectionType = 'OTHER';
      
      if (matchingKeyword) {
        sectionType = matchingKeyword.type;
      } else {
        // Try to infer section type from content
        for (const [type, keywords] of Object.entries(STUDENT_SECTIONS)) {
          if (keywords.some(keyword => cleanLine.includes(keyword))) {
            sectionType = type;
            break;
          }
        }
      }
      
      headerLines.push({ 
        index, 
        text: line.trim(),
        type: sectionType,
        confidence: isAllCaps ? 0.8 : (matchingKeyword ? 0.9 : 0.6)
      });
    }
  });
  
  // Sort headers by position
  headerLines.sort((a, b) => a.index - b.index);
  
  // Determine section boundaries
  const sections = {};
  headerLines.forEach((header, idx) => {
    // Get content until next section header or end of text
    const startIndex = header.index + 1;
    const endIndex = idx < headerLines.length - 1 ? headerLines[idx + 1].index : lines.length;
    
    // Extract section content
    const content = lines.slice(startIndex, endIndex)
                        .filter(line => line.trim()) // Remove empty lines
                        .join('\n');
    
    // Store section
    if (content.trim()) {
      sections[header.type] = sections[header.type] 
        ? sections[header.type] + '\n\n' + content 
        : content;
    }
  });
  
  return {
    sections,
    headerLines
  };
}

/**
 * Parse a student resume using specialized text-based approach
 */
async function parseStudentResumeTextBased(pdfPath) {
  try {
    console.log(`Using student resume text-based parser for: ${pdfPath}`);
    const startTime = Date.now();
    
    // Extract text from PDF
    const text = await extractTextFromPDF(pdfPath);
    if (!text) {
      throw new Error('Failed to extract text from PDF');
    }
    
    // Identify sections using student-focused algorithm
    const { sections, headerLines } = identifyStudentSections(text);
    
    // Convert headings to the same format used by the main parser
    const headings = headerLines.map((header, index) => ({
      page: 0, // We don't have page info in the text-based approach
      text: header.text,
      x0: 0,
      y0: header.index,
      x1: 100,
      y1: header.index + 1,
      type: header.type,
      column: "center", // We don't detect columns in the text-based approach
      confidence: header.confidence
    }));
    
    // Convert sections to the same format used by the main parser
    const sectionTexts = {};
    Object.entries(sections).forEach(([sectionType, content]) => {
      sectionTexts[sectionType] = `--- STUDENT RESUME EXTRACTION ---\n${content}`;
    });
    
    // Calculate processing time
    const processingTime = (Date.now() - startTime) / 1000;
    
    return {
      pdfPath,
      headings,
      sectionTexts,
      processingTime,
      usedFallback: true,
      fallbackHeaderCount: headerLines.length,
      // Add empty or default values for compatibility with main parser's result structure
      columnDivider: 0,
      fontStatistics: {
        mean: 0,
        median: 0,
        std: 0,
        min: 0,
        max: 0,
        mostCommon: 0
      }
    };
  } catch (error) {
    console.error(`Error in student resume parser: ${error.message}`);
    throw error; // Re-throw so the main parser can catch it
  }
}

module.exports = {
  parseResumeTextBased,
  parseStudentResumeTextBased,
  extractTextFromPDF,
  identifySections,
  identifyStudentSections
};