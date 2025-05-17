// Import required libraries
const fs = require('fs');
const path = require('path');
const math = require('mathjs');
const { PDFDocument, rgb, StandardFonts } = require('pdf-lib');
const fallbackParser = require('./fallback-parser');


// Set up DOM environment for PDF.js
const { JSDOM } = require('jsdom');
const { window } = new JSDOM("");
global.window = window;
global.document = window.document;
global.navigator = { userAgent: 'node.js' };
global.Image = window.Image;

// Canvas is needed for some PDF.js operations
const canvas = require('canvas');
global.Canvas = canvas.Canvas;

// Import and configure PDF.js for Node.js
const pdfjsLib = require('pdfjs-dist/legacy/build/pdf.js');
pdfjsLib.GlobalWorkerOptions.workerSrc = path.join(
  path.dirname(require.resolve('pdfjs-dist/package.json')),
  'legacy/build/pdf.worker.js'
);

// Constants for section detection
const MIN_HEADING_LENGTH = 2;  // Minimum length for heading text
const LINE_SPACING_TOLERANCE = 2;  // Tolerance for grouping elements into lines (pixels)
const CONFIDENCE_THRESHOLD = 0.71; // Threshold for heading detection confidence

// Enhanced Section keywords - similar to your Python implementation
const SECTION_KEYWORDS = {
    "OBJECTIVE": [
        "objective", "career objective", "professional objective", "job objective",
        "employment objective", "career goal", "professional goal", "career aim",
        "professional summary", "personal statement", "career summary"
    ],
    "EDUCATION": [
        "education", "academic", "academics", "qualification", "qualifications", "educational", 
        "edu", "schooling", "school", "studies", "study", "learning", "degree", "degrees", 
        "university", "universities", "college", "colleges", "educational background", 
        "academic background", "educational qualifications", "academic record", 
        "academic credentials", "academic history", "scholastic record", "academic profile", 
        "university education", "college education", "school education", "tertiary education", 
        "higher education"
    ],
    "EXPERIENCE": [
        "experience", "experiences", "work", "workexperience", "work-experience", 
        "work/experience", "employment", "career", "job", "jobs", "workhistory", 
        "work history", "professional", "prof", "professionalexperience", 
        "professional background", "employment history", "internship", "internships", 
        "exp", "wrk", "work exp", "prof exp", "occupation", "occupations", "positions", 
        "Work Experience"
    ],
    "SKILLS": [
        "skills", "skill", "abilities", "ability", "expertise", "competencies", "competences", 
        "capabilities", "capability", "proficiencies", "proficiency", "talents", "talent", 
        "aptitudes", "technicalskills", "technical skills", "softskill", "soft skills", 
        "hardskills", "hard skills", "core competencies", "skls", "proficiencies", 
        "competency", "tech skills"
    ],
    "PROJECTS": [
        "projects", "project", "works", "portfolio", "showcase", "achievements", "initiatives", 
        "implementations", "developments", "applications", "software projects", 
        "development projects", "proj", "key projects", "personal projects", 
        "academic projects", "research projects"
    ],
    "CONTACT": [
        "contact", "contacts", "contact information", "contact details", "contact info", 
        "personal", "personal information", "personal details", "personal info", 
        "personal data", "personaldetails", "personal-details", "info", "details", 
        "get in touch", "reach me", "contact me", "connect"
    ],
    "SUMMARY": [
        "summary", "professional summary", "career summary", "profile", "professional profile", 
        "overview", "about", "about me", "introduction", "bio", "biography", "background", 
        "brief", "executive summary", "career objective", "career goal", "careerobjective", 
        "objective", "objective", "summary of qualifications", "qualifications summary", 
        "career profile", "summ", "curriculum", "vitae", "cv", "resume summary", 
        "professional background", "candidate summary"
    ],
    "CERTIFICATIONS": [
        "certifications", "certification", "certificates", "certificate", "CERTIFICATES", 
        "credentials", "licenses", "license", "accreditation", "accreditations", 
        "qualification", "qualifications", "certs", "certified", "diplomas", "diploma", 
        "professional certifications", "professional development"
    ],
    "LANGUAGES": [
    "languages", "language", "language skills", "language proficiency", "language proficiencies", 
    "language abilities", "spoken languages", "linguistic skills", "idiomas", "foreign languages",
    "multilingual", "bilingual", "fluency", "lang", "linguistic", "spoken",
    "lang proficiency", "language competency", "fluent in", "native", "mother tongue"
]
    // Other sections can be added as needed
};

// Flatten the section keywords for easier lookup
const FLAT_SECTION_KEYWORDS = {};
Object.entries(SECTION_KEYWORDS).forEach(([sectionType, keywords]) => {
    keywords.forEach(keyword => {
        FLAT_SECTION_KEYWORDS[keyword] = sectionType;
    });
});

// Helper functions
function shouldUseFallback(pdfPath, elements, headings) {
  console.log("heading-l",headings.length);

  if (!headings || headings.length < 3 || headings.length > 7) {
      // Log the decision factors
    console.log(`- Has too few headings (${headings.length})`);
    return true;
  }
 
}

function normalize(text) {
    // Normalize text by removing whitespace and converting to lowercase
    if (!text) return "";
    return text.replace(/\s+/g, '').toLowerCase();
}

function preprocessText(text) {
    // Preprocess text to handle special characters and unicode
    if (!text) return "";
    
    // Replace common unicode characters
    const replacements = {
        '\u2013': '-',  // en dash
        '\u2014': '-',  // em dash
        '\u2018': "'",  // left single quote
        '\u2019': "'",  // right single quote
        '\u201c': '"',  // left double quote
        '\u201d': '"',  // right double quote
        '\u2022': '*',  // bullet
        '\u2026': '...', // ellipsis
        '\u00a0': ' ',  // non-breaking space
        '\u00ad': '-',  // soft hyphen
    };
    
    Object.entries(replacements).forEach(([char, replacement]) => {
        text = text.replace(new RegExp(char, 'g'), replacement);
    });
    
    // Remove control characters
    text = text.replace(/[\x00-\x1F\x7F-\x9F]/g, '');
    
    return text.trim();
}

function isHeadingText(text, sectionKeywords = null) {
    // Improved heading detection with better confidence scoring
    if (!sectionKeywords) {
        sectionKeywords = FLAT_SECTION_KEYWORDS;
    }
    
    // Normalize and preprocess the text for comparison
    const clean = normalize(preprocessText(text));
    const originalText = text.trim().toLowerCase();
    
    if (!clean || clean.length < MIN_HEADING_LENGTH) {
        return [false, 0.0, null];
    }
    
    // Direct match first (highest confidence)
    if (sectionKeywords[clean]) {
        return [true, 1.0, sectionKeywords[clean]];
    }
    
    // Check for full matches with common patterns
    const sectionPatterns = {
        "\\b(education|academic|degree|qualification)s?\\b": "EDUCATION",
        "\\b(experience|work|employment|job|career|professional)s?\\b": "EXPERIENCE",
        "\\b(skill|ability|proficiency|competency|expertise)s?\\b": "SKILLS",
        "\\b(technical|hard|programming|computer|software|coding)\\s+skill": "SKILLS",
        "\\b(soft|personal|interpersonal|communication)\\s+skill": "SKILLS",
        "\\b(project|portfolio|work|assignment|application)s?\\b": "PROJECTS",
        "\\b(certification|certificate|credential|qualification)s?\\b": "CERTIFICATIONS",
        "\\b(award|honor|achievement|recognition|scholarship)s?\\b": "AWARDS",
        "\\b(language|linguistic|idioma|tongue)\\b": "LANGUAGES",
        "\\b(objective|goal|summary|profile|about)s?\\b": "SUMMARY",
        "\\b(leadership|participation|involvement|activity)s?\\b": "ACTIVITIES",
        "\\b(hackathon|competition|contest|challenge|event)s?\\b": "ACTIVITIES",
        "\\b(reference|recommendation|endorsement|referee)s?\\b": "REFERENCES",
        "\\b(interest|hobby|pastime|leisure|extracurricular)s?\\b": "INTERESTS",
        "\\b(strength|forte|capability|talent|power)s?\\b": "STRENGTH",
        "\\b(contact|detail|information|reach|connect)\\b": "CONTACT",
        "\\b(publication|paper|article|journal|research)s?\\b": "PUBLICATIONS",
        "\\b(internship|training|practical)s?\\b": "EXPERIENCE"
    };
    
    // Check patterns
    for (const [pattern, sectionType] of Object.entries(sectionPatterns)) {
        const regex = new RegExp(pattern, 'i');
        const match = originalText.match(regex);
        if (match) {
            // Return high confidence if it's a strong match
            const matchStrength = (originalText.match(new RegExp(pattern, 'gi')) || []).length;
            if (matchStrength > 0) {
                return [true, Math.min(0.95, 0.75 + matchStrength * 0.1), sectionType];
            }
        }
    }
    
    // Check for partial matches with standard approach
    const matches = [];
    
    // Check if any keyword is contained in the text
    Object.entries(sectionKeywords).forEach(([keyword, sectionType]) => {
        if (clean.includes(keyword)) {
            // Calculate confidence based on keyword length and position
            let confidence = 0;
            if (clean.startsWith(keyword)) {
                confidence = 0.9;
            } else {
                confidence = (keyword.length / clean.length) * 0.8;
            }
            matches.push([confidence, sectionType]);
        }
    });
    
    // Check if the text is contained in any keyword (abbreviated form)
    if (clean.length >= MIN_HEADING_LENGTH + 1) {
        Object.entries(sectionKeywords).forEach(([keyword, sectionType]) => {
            if (keyword.includes(clean) && clean.length >= keyword.length * 0.5) {
                const confidence = (clean.length / keyword.length) * 0.7;
                matches.push([confidence, sectionType]);
            }
        });
    }
    
    // Special check for title formats with colons 
    // e.g., "Skills:" or "Technical Experience:"
    if (originalText.includes(":")) {
        const prefix = originalText.split(":")[0].trim().toLowerCase();
        if (prefix.length >= MIN_HEADING_LENGTH) {
            Object.entries(sectionKeywords).forEach(([keyword, sectionType]) => {
                if (prefix.endsWith(keyword) || keyword.endsWith(prefix)) {
                    const confidence = Math.min(0.9, 0.7 + prefix.length / 20);
                    matches.push([confidence, sectionType]);
                }
            });
        }
    }
    
    // Find the best match
    if (matches.length > 0) {
        matches.sort((a, b) => b[0] - a[0]); // Sort by confidence (descending)
        const bestMatch = matches[0];
        // Lower the threshold for detection to catch more headings
        if (bestMatch[0] >= CONFIDENCE_THRESHOLD - 0.1) {
            return [true, bestMatch[0], bestMatch[1]];
        }
    }
    
    return [false, 0.0, null];
}

// Helper function to calculate percentiles
function percentile(arr, p) {
    if (arr.length === 0) return 0;
    const sortedArr = [...arr].sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sortedArr.length) - 1;
    return sortedArr[index];
}

// Main extraction function using PDF.js
async function extractTextWithStyle(pdfPath) {
    try {
        const data = new Uint8Array(fs.readFileSync(pdfPath));
        const loadingTask = pdfjsLib.getDocument({ data });
        const pdf = await loadingTask.promise;
        
        const allElements = [];
        
        // Process each page
        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
            try {
                const page = await pdf.getPage(pageNum);
                const textContent = await page.getTextContent();
                
                // Get page viewport for positioning
                const viewport = page.getViewport({ scale: 1.0 });
                
                // Process text items
                for (const item of textContent.items) {
                    if (item.str === undefined) continue;
                    
                    const text = item.str.trim();
                    if (!text) continue;
                    
                    // Clean and preprocess the text
                    const processedText = preprocessText(text);
                    
                    // Extract style information
                    const fontName = item.fontName || '';
                    const fontSize = item.height || 0;
                    const isBold = 
                        fontName.toLowerCase().includes('bold') ||
                        fontName.toLowerCase().includes('heavy') ||
                        fontName.toLowerCase().includes('black') ||
                        fontName.toLowerCase().includes('extrabold');
                    const isCapital = text === text.toUpperCase() && text.length > 2;
                    const isSpaced = text.includes(' ') && 
                                   text.split(' ').every(part => part.length === 1);
                    
                    // Calculate position (PDF.js uses different coordinate system)
                    const x0 = item.transform[4];
                    const y0 = viewport.height - (item.transform[5] + item.height);
                    const x1 = x0 + (item.width || text.length * (fontSize * 0.6));
                    const y1 = y0 + (item.height || fontSize);
                    
                    allElements.push({
                        page: pageNum - 1, // 0-based page number
                        text: processedText,
                        x0: x0,
                        y0: y0,
                        x1: x1,
                        y1: y1,
                        fontName: fontName,
                        fontSize: fontSize,
                        isBold: isBold,
                        isCapital: isCapital,
                        isSpaced: isSpaced,
                        isLikelyHeading: false, // Will be determined later
                        confidence: 0.0,        // Will store confidence in heading detection
                        sectionType: null       // Will store detected section type
                    });
                }
            } catch (e) {
                console.warn(`Error processing page ${pageNum}: ${e.message}`);
            }
        }
        
        return allElements;
    } catch (e) {
        console.error(`Failed to extract text from ${pdfPath}: ${e.message}`);
        console.error(e.stack);
        return [];
    }
}

// Analyze font statistics
function analyzeFontStatistics(elements) {
    if (!elements || elements.length === 0) {
        return {
            mean: 0,
            median: 0,
            std: 0,
            p25: 0,
            p50: 0,
            p75: 0,
            p90: 0,
            p95: 0,
            min: 0,
            max: 0,
            mostCommon: 0,
            headingThreshold: 0
        };
    }
    
    // Extract font sizes
    const fontSizes = elements
        .filter(elem => elem.fontSize > 0)
        .map(elem => elem.fontSize);
    
    if (fontSizes.length === 0) {
        return {
            mean: 0,
            median: 0,
            std: 0,
            p25: 0,
            p50: 0,
            p75: 0,
            p90: 0,
            p95: 0,
            min: 0,
            max: 0,
            mostCommon: 0,
            headingThreshold: 0
        };
    }
    
    // Calculate statistics
    const meanSize = math.mean(fontSizes);
    const medianSize = math.median(fontSizes);
    const stdSize = math.std(fontSizes);
    
    // Calculate percentiles
    fontSizes.sort((a, b) => a - b);
    const p25 = percentile(fontSizes, 25);
    const p50 = percentile(fontSizes, 50);
    const p75 = percentile(fontSizes, 75);
    const p90 = percentile(fontSizes, 90);
    const p95 = percentile(fontSizes, 95);
    
    // Find min and max
    const minSize = Math.min(...fontSizes);
    const maxSize = Math.max(...fontSizes);
    
    // Find most common font size (mode)
    const sizeFrequency = {};
    fontSizes.forEach(size => {
        sizeFrequency[size] = (sizeFrequency[size] || 0) + 1;
    });
    
    let mostCommon = fontSizes[0];
    let maxFreq = 0;
    Object.entries(sizeFrequency).forEach(([size, freq]) => {
        if (freq > maxFreq) {
            maxFreq = freq;
            mostCommon = parseFloat(size);
        }
    });
    
    // Calculate a good threshold for heading detection
    const headingThreshold = Math.max(p75, mostCommon * 1.1);
    
    return {
        mean: meanSize,
        median: medianSize,
        std: stdSize,
        p25: p25,
        p50: p50,
        p75: p75,
        p90: p90,
        p95: p95,
        min: minSize,
        max: maxSize,
        mostCommon: mostCommon,
        headingThreshold: headingThreshold
    };
}

// Detect columns in the PDF
async function detectColumnWhitespace(elements, pdfPath) {
    try {
        // Load PDF to get page dimensions
        const data = new Uint8Array(fs.readFileSync(pdfPath));
        const loadingTask = pdfjsLib.getDocument({ data });
        const pdf = await loadingTask.promise;
        const page = await pdf.getPage(1);  // Get first page
        const viewport = page.getViewport({ scale: 1.0 });
        
        const pageWidth = viewport.width;
        const pageHeight = viewport.height;
        
        // Skip header/footer by focusing on the middle 80% of the page
        const topMargin = pageHeight * 0.15;
        const bottomMargin = pageHeight * 0.8;
        
        // Only include elements in the main content area (exclude header/footer)
        const contentElements = elements.filter(e => 
            topMargin <= e.y0 && e.y0 <= bottomMargin);
        
        if (contentElements.length < 10) {
            return pageWidth;  // Default if not enough elements
        }
        
        // Divide the page width into bins (e.g., 50 bins for more granularity)
        const numBins = 50;
        const binWidth = pageWidth / numBins;
        
        // Create a density map for the x-axis
        const xDensity = new Array(numBins).fill(0);
        
        // Calculate density
        contentElements.forEach(elem => {
            // Mark bins that this element covers
            const startBin = Math.max(0, Math.min(numBins - 1, Math.floor(elem.x0 / binWidth)));
            const endBin = Math.max(0, Math.min(numBins - 1, Math.floor(elem.x1 / binWidth)));
            
            for (let binIdx = startBin; binIdx <= endBin; binIdx++) {
                xDensity[binIdx]++;
            }
        });
        
        // Find the longest continuous whitespace gap in the middle 60% of the page
        const middleStart = Math.floor(numBins * 0.3);  // Start at 30% of page width
        const middleEnd = Math.floor(numBins * 0.9);    // End at 90% of page width
        
        let maxGapLength = 0;
        let maxGapCenter = Math.floor(numBins / 2);  // Default to middle
        
        let currentGapStart = null;
        
        for (let i = middleStart; i < middleEnd; i++) {
            // If bin is empty (or nearly empty), it's part of a gap
            if (xDensity[i] <= contentElements.length * 0.02) {  // Allow some noise (2%)
                if (currentGapStart === null) {
                    currentGapStart = i;
                }
            } else {
                // End of a gap
                if (currentGapStart !== null) {
                    const gapLength = i - currentGapStart;
                    if (gapLength > maxGapLength) {
                        maxGapLength = gapLength;
                        maxGapCenter = (currentGapStart + i) / 2;
                    }
                    currentGapStart = null;
                }
            }
        }
        
        // If we ended with an open gap, close it
        if (currentGapStart !== null) {
            const gapLength = middleEnd - currentGapStart;
            if (gapLength > maxGapLength) {
                maxGapLength = gapLength;
                maxGapCenter = (currentGapStart + middleEnd) / 2;
            }
        }
        
        // If we found a significant gap (at least 1% of page width), use it
        if (maxGapLength >= numBins * 0.01) {
            return maxGapCenter * binWidth;
        }
        
        // Fallback to page width if no clear gap is found
        return pageWidth;
    } catch (e) {
        console.error(`Error detecting columns: ${e.message}`);
        return 612;  // Default letter page width
    }
}

// Identify potential headings
function identifyPotentialHeadings(elements, fontStats) {
    if (!elements || !fontStats) {
        return elements;
    }
    
    // Calculate the document-wide font statistics
    const avgFontSize = fontStats.mean;
    const maxFontSize = fontStats.max;
    const mostCommonSize = fontStats.mostCommon;
    const medianSize = fontStats.median;
    
    // Adjust the heading threshold to be more inclusive
    const headingBaseThreshold = Math.max(mostCommonSize * 1.05, medianSize * 1.1);
    
    // Find all bold text elements
    const boldElements = elements.filter(e => e.isBold);
    const boldFontSizes = boldElements
        .filter(e => e.fontSize > 0)
        .map(e => e.fontSize);
    
    // If there are enough bold elements, calculate their own statistics
    let headingThreshold = headingBaseThreshold;
    if (boldFontSizes.length >= 5) {
        const boldMedianSize = math.median(boldFontSizes);
        // Lower the threshold if bold text is used strategically
        headingThreshold = Math.min(headingBaseThreshold, boldMedianSize * 0.95);
    }
    
    // First pass: Reset all elements
    elements.forEach(elem => {
        elem.isLikelyHeading = false;
        elem.confidence = 0.0;
        elem.sectionType = null;
    });
    
    // Second pass: Calculate confidence scores with improved algorithm
    elements.forEach(elem => {
        // Skip very short text (likely not headings)
        if (elem.text.trim().length < MIN_HEADING_LENGTH) {
            return;
        }
        
        // Base confidence starts at 0
        let confidenceScore = 0.0;
        
        // 1. Font size factor: How much larger is this than the document's typical text?
        let fontSizeConfidence = 0.0;
        if (elem.fontSize > 0) {
            const sizeRatio = elem.fontSize / mostCommonSize;
            // Exponential scaling with smoothing
            fontSizeConfidence = Math.min(1.0, Math.max(0, (sizeRatio - 1) * 1.5));
        }
        
        // 2. Styling factor: Bold, all caps, and positioning
        let styleConfidence = 0.0;
        
        // Bold text is very indicative of headings
        if (elem.isBold) {
            styleConfidence += 0.4;
        }
        
        // All caps are often used for headings
        if (elem.isCapital && elem.text.length >= 3) {
            styleConfidence += 0.25;
        }
        
        // Check if text appears to be a single line (may be a heading)
        if (!elem.text.includes("\n") && elem.text.split(" ").length <= 5) {
            styleConfidence += 0.15;
        }
        
        // Position factor - headings often start at the left margin or page edge
        const pageStart = 50;  // Assume page margin is within 50 pixels
        if (elem.x0 <= pageStart) {
            styleConfidence += 0.1;
        }
        
        // 3. Keyword matching - is this text similar to common section headers?
        const [isHeading, keywordConfidence, sectionType] = isHeadingText(elem.text);
        
        // 4. Length factor - headings are typically short
        // Use an exponential decay function - shorter text gets higher scores
        const textLength = elem.text.trim().length;
        let lengthFactor = 0.0;
        if (textLength <= 30) {
            lengthFactor = 1.0;  // Perfect length
        } else {
            // Exponential decay for longer text
            lengthFactor = Math.max(0.4, Math.exp(-0.02 * (textLength - 30)));
        }
        
        // Combined confidence score with weighted factors
        // The weights sum to 1.0
        confidenceScore = (
            keywordConfidence * 0.35 +  // Keywords are strong indicators
            fontSizeConfidence * 0.3 +  // Font size is very important
            styleConfidence * 0.25 +    // Styling matters a lot
            lengthFactor * 0.1          // Length is a secondary factor
        );
        
        // Apply a non-linear scaling to boost mid-to-high confidence scores
        if (confidenceScore > 0.5) {
            // Apply a positive bias to scores above 0.5
            confidenceScore = 0.5 + (confidenceScore - 0.5) * 1.5;
        }
        
        // Ensure we don't exceed 1.0
        confidenceScore = Math.min(1.0, confidenceScore);
        
        // Apply threshold with reduced stringency
        if (confidenceScore >= CONFIDENCE_THRESHOLD - 0.1) {  // More inclusive
            elem.isLikelyHeading = true;
            elem.confidence = confidenceScore;
            elem.sectionType = sectionType;
            
            // Extra boost for extremely strong signals
            if (elem.isBold && elem.isCapital && 
                fontSizeConfidence > 0.4 && keywordConfidence > 0.6) {
                elem.confidence = Math.min(1.0, confidenceScore * 1.2);
            }
        }
    });
    
    return elements;
}

// Identify section headings
function identifySectionHeadings(elements, columnDivider) {
    const headings = [];
    
    // Sort elements first by confidence score (descending)
    const sortedElements = [...elements].sort((a, b) => b.confidence - a.confidence);
    
    // Track which text positions we've already marked as headings
    const headingPositions = new Set();
    
    // First pass - collect all high-confidence headings
    const highConfidenceThreshold = 0.75;
    sortedElements.forEach(elem => {
        // Only consider high-confidence elements in this pass
        if (elem.confidence < highConfidenceThreshold || !elem.isLikelyHeading) {
            return;
        }
        
        // Skip if we've already included this position as a heading
        const positionKey = `${elem.page}-${Math.round(elem.y0)}-${Math.round(elem.x0)}`;
        if (headingPositions.has(positionKey)) {
            return;
        }
        
        // Skip very short text
        if (elem.text.trim().length < MIN_HEADING_LENGTH) {
            return;
        }
        
        // Determine which column this heading belongs to
        const column = elem.x0 < columnDivider ? "left" : "right";
        
        // Add to headings list
        headings.push({
            page: elem.page,
            text: elem.text,
            x0: elem.x0,
            y0: elem.y0,
            x1: elem.x1,
            y1: elem.y1,
            type: elem.sectionType || "OTHER",
            column: column,
            confidence: elem.confidence
        });
        
        // Mark this position as used
        headingPositions.add(positionKey);
    });
    
    // Second pass - collect medium confidence headings
    sortedElements.forEach(elem => {
        // Skip if not likely a heading or high confidence (already processed)
        if (!elem.isLikelyHeading || elem.confidence >= highConfidenceThreshold) {
            return;
        }
        
        // Skip if we've already included this position as a heading
        const positionKey = `${elem.page}-${Math.round(elem.y0)}-${Math.round(elem.x0)}`;
        if (headingPositions.has(positionKey)) {
            return;
        }
        
        // Skip very short text
        if (elem.text.trim().length < MIN_HEADING_LENGTH) {
            return;
        }
        
        // Apply formatting-based confidence boost
        let boostedConfidence = elem.confidence;
        
        // 1. Boost for all caps text
        if (elem.isCapital && elem.text.length >= 3) {
            boostedConfidence = Math.min(0.9, boostedConfidence + 0.15);
        }
        
        // 2. Boost for text with colon ending (e.g., "Skills:")
        if (elem.text.trim().endsWith(":")) {
            boostedConfidence = Math.min(0.9, boostedConfidence + 0.15);
        }
        
        // 3. Boost for short text (likely a heading)
        if (elem.text.trim().length < 20) {
            const boostFactor = 0.05 + Math.max(0, (20 - elem.text.trim().length) / 100);
            boostedConfidence = Math.min(0.9, boostedConfidence + boostFactor);
        }
        
        // 4. Boost for text at the left margin
        const columnMargin = elem.x0 < columnDivider ? columnDivider * 0.1 : columnDivider * 1.1;
        if (Math.abs(elem.x0 - columnMargin) < 20) {
            boostedConfidence = Math.min(0.9, boostedConfidence + 0.05);
        }
        
        // Determine which column this heading belongs to
        const column = elem.x0 < columnDivider ? "left" : "right";
        
        // Add to headings list
        headings.push({
            page: elem.page,
            text: elem.text,
            x0: elem.x0,
            y0: elem.y0,
            x1: elem.x1,
            y1: elem.y1,
            type: elem.sectionType || "OTHER",
            column: column,
            confidence: boostedConfidence
        });
        
        // Mark this position as used
        headingPositions.add(positionKey);
    });
    
    // Sort headings by page and y-coordinate for final output
    return headings.sort((a, b) => {
        if (a.page !== b.page) return a.page - b.page;
        return a.y0 - b.y0;
    });
}

// Extract sections from the PDF
function extractSections(headings, elements, columnDivider) {
    const sections = {};
    
    if (!headings || headings.length === 0) {
        console.warn("No headings found");
        return sections;
    }
    
    // Separate headings by column
    const leftHeadings = headings.filter(h => h.column === "left");
    const rightHeadings = headings.filter(h => h.column === "right");
    
    // Sort headings in each column by page and vertical position
    leftHeadings.sort((a, b) => {
        if (a.page !== b.page) return a.page - b.page;
        return a.y0 - b.y0;
    });
    
    rightHeadings.sort((a, b) => {
        if (a.page !== b.page) return a.page - b.page;
        return a.y0 - b.y0;
    });
    
    // Group elements by page for faster access
    const elementsByPage = {};
    elements.forEach(elem => {
        if (!elementsByPage[elem.page]) {
            elementsByPage[elem.page] = [];
        }
        elementsByPage[elem.page].push(elem);
    });
    
    // Process each column separately
    [["left", leftHeadings], ["right", rightHeadings]].forEach(([column, colHeadings]) => {
        colHeadings.forEach((heading, i) => {
            const headingPage = heading.page;
            const headingY = heading.y1;  // Bottom of the heading
            
            // Find the next heading in the same column
            let endY = Infinity;
            let endPage = Infinity;
            
            if (i < colHeadings.length - 1) {
                const nextHeading = colHeadings[i + 1];
                endY = nextHeading.y0;
                endPage = nextHeading.page;
            }
            
            // Collect all elements in this section
            const sectionElements = [];
            
            // Define the column boundaries
            const minX = column === "left" ? 0 : columnDivider;
            const maxX = column === "left" ? columnDivider : Infinity;
            
            // Process elements page by page
            const maxPageNum = Math.max(...Object.keys(elementsByPage).map(k => parseInt(k, 10)));
            for (let pageNum = headingPage; pageNum <= Math.min(endPage, maxPageNum); pageNum++) {
                if (!elementsByPage[pageNum]) {
                    continue;
                }
                
                elementsByPage[pageNum].forEach(elem => {
                    // Check if the element is in the correct column
                    const elemInColumn = minX <= elem.x0 && elem.x0 < maxX;
                    
                    // Skip if not in this column
                    if (!elemInColumn) {
                        return;
                    }
                    
                    // Skip if this is a heading element
                    let isHeadingElem = false;
                    for (const h of headings) {
                        if (elem.page === h.page && 
                            Math.abs(elem.y0 - h.y0) < 2 && 
                            Math.abs(elem.x0 - h.x0) < 2) {
                            isHeadingElem = true;
                            break;
                        }
                    }
                    
                    if (isHeadingElem) {
                        return;
                    }
                    
                    // If on heading page, only include elements after the heading
                    if (pageNum === headingPage && elem.y0 < headingY) {
                        return;
                    }
                    
                    // If on end page, only include elements before the next heading
                    if (pageNum === endPage && elem.y0 >= endY) {
                        return;
                    }
                    
                    sectionElements.push(elem);
                });
            }
            
            // Generate a unique section key
            const sectionKey = `${heading.type}_${heading.page}_${column}_${i}`;
            
            // Store the section
            sections[sectionKey] = {
                heading: heading,
                elements: sectionElements,
                column: column,
                confidence: heading.confidence
            };
        });
    });
    
    return sections;
}

// Extract section text
function extractSectionText(sections) {
    const sectionTexts = {};
    
    for (const [sectionKey, sectionData] of Object.entries(sections)) {
        const elements = sectionData.elements;
        const sectionType = sectionData.heading.type;
        const column = sectionData.column;
        const confidence = sectionData.confidence;
        
        if (!elements || elements.length === 0) {
            continue;
        }
        
        // Group elements by page
        const elementsByPage = {};
        elements.forEach(elem => {
            if (!elementsByPage[elem.page]) {
                elementsByPage[elem.page] = [];
            }
            elementsByPage[elem.page].push(elem);
        });
        
        // Process each page separately
        const textLinesByPage = [];
        
        for (const pageNum of Object.keys(elementsByPage).sort((a, b) => parseInt(a) - parseInt(b))) {
            const pageElements = elementsByPage[pageNum];
            
            // Calculate the average font size for this page's elements
            const fontSizes = pageElements
                .filter(e => e.fontSize > 0)
                .map(e => e.fontSize);
            
            const avgFontSize = fontSizes.length > 0 ? 
                math.mean(fontSizes) : 10;  // Default if no valid font sizes
            
            // Use the average font size to determine line spacing tolerance
            const lineSpacing = Math.max(LINE_SPACING_TOLERANCE, avgFontSize * 0.3);
            
            // Sort elements by y position
            const sortedY = [...pageElements].sort((a, b) => a.y0 - b.y0);
            
            // Group elements into lines using dynamic tolerance
            const lines = [];
            if (sortedY.length > 0) {
                let currentLine = [sortedY[0]];
                let currentY = sortedY[0].y0;
                
                for (let i = 1; i < sortedY.length; i++) {
                    const elem = sortedY[i];
                    // If this element is within the tolerance of the current line, add it
                    if (Math.abs(elem.y0 - currentY) <= lineSpacing) {
                        currentLine.push(elem);
                    } else {
                        // Sort the current line elements by x position
                        currentLine.sort((a, b) => a.x0 - b.x0);
                        lines.push(currentLine);
                        
                        // Start a new line
                        currentLine = [elem];
                        currentY = elem.y0;
                    }
                }
                
                // Add the last line
                if (currentLine.length > 0) {
                    currentLine.sort((a, b) => a.x0 - b.x0);
                    lines.push(currentLine);
                }
            }
            
            // Build text lines for this page
            const pageTextLines = [];
            for (const line of lines) {
                const lineText = line.map(elem => elem.text).join(' ');
                // Clean up extra spaces
                const cleanLineText = lineText.replace(/\s+/g, ' ').trim();
                if (cleanLineText) {  // Only add non-empty lines
                    pageTextLines.push(cleanLineText);
                }
            }
            
            // Add page number if there are multiple pages
            if (Object.keys(elementsByPage).length > 1) {
                pageTextLines.unshift(`[Page ${parseInt(pageNum, 10) + 1}]`);
            }
            
            textLinesByPage.push(...pageTextLines);
        }
        
        // Join all lines with newlines
        // Use a simple section key for better readability
        const simpleKey = sectionType;
        const sectionContent = textLinesByPage.join('\n');
        
        // Add column information
        const sectionHeader = `--- ${column.toUpperCase()} COLUMN ---\n`;
        
        if (sectionTexts[simpleKey]) {
            // If we already have this section type, append new content
            sectionTexts[simpleKey] += "\n\n" + sectionHeader + sectionContent;
        } else {
            sectionTexts[simpleKey] = sectionHeader + sectionContent;
        }
    }
    
    return sectionTexts;
}

// Main function to parse resume
async function parseResume(pdfPath) {
    try {
        console.log(`Processing resume: ${pdfPath}`);
        const startTime = Date.now();
        
        // Extract text elements with style information
        const elements = await extractTextWithStyle(pdfPath);
        if (!elements || elements.length === 0) {
            console.error(`Failed to extract text elements from ${pdfPath}`);
            console.log(`Trying fallback parser due to text extraction failure.`);
            return await fallbackParser.parseResumeTextBased(pdfPath);
        }
        
        console.log(`Extracted ${elements.length} text elements`);
        
        // Analyze font statistics
        const fontStats = analyzeFontStatistics(elements);
        console.log(`Font statistics: mean=${fontStats.mean.toFixed(2)}, median=${fontStats.median.toFixed(2)}, ` +
                    `max=${fontStats.max.toFixed(2)}, threshold=${fontStats.headingThreshold.toFixed(2)}`);
        
        // Identify potential headings with confidence scores
        const enhancedElements = identifyPotentialHeadings(elements, fontStats);
        const potentialHeadings = enhancedElements.filter(e => e.isLikelyHeading);
        console.log(`Identified ${potentialHeadings.length} potential headings`);
        
        // Analyze the layout to detect columns
        const columnDivider = await detectColumnWhitespace(elements, pdfPath);
        console.log(`Detected column divider at x-coordinate: ${columnDivider.toFixed(2)}`);
        
        // Identify section headings with confidence scores
        const headings = identifySectionHeadings(enhancedElements, columnDivider);
        console.log(`Identified ${headings.length} section headings`);

        // Early check for fallback - if we don't have enough headings, switch to fallback
        if (shouldUseFallback(pdfPath, elements, headings)) {
            console.log(`Using fallback parser for ${pdfPath} due to insufficient heading detection.`);
            return await fallbackParser.parseResumeTextBased(pdfPath);
            
        }
        
        // Count headings in each column
        const leftHeadings = headings.filter(h => h.column === "left");
        const rightHeadings = headings.filter(h => h.column === "right");
        console.log(`  - Left column: ${leftHeadings.length} headings`);
        console.log(`  - Right column: ${rightHeadings.length} headings`);
        
        // Log each heading with confidence
        headings.forEach(heading => {
            console.log(`  - ${heading.text} (${heading.type}, ${heading.column} column, ` +
                        `confidence: ${heading.confidence.toFixed(2)})`);
        });
        
        // Extract sections
        const sections = extractSections(headings, enhancedElements, columnDivider);
        console.log(`Extracted ${Object.keys(sections).length} sections`);

       let emptySections = 0;
       let totalSections = Object.keys(sections).length;


        // Allow up to 2 empty sections before switching to fallback
        if (headings.length > totalSections + 1) {
            console.log(`Using fallback parser for ${pdfPath} due to ${totalSections} sections having insufficient content.`);
            return await fallbackParser.parseResumeTextBased(pdfPath);
        }
        
        // Extract text from each section
        const sectionTexts = extractSectionText(sections);
        
        // Calculate processing time
        const processingTime = (Date.now() - startTime) / 1000;
        
        // Prepare the result
        const result = {
            pdfPath: pdfPath,
            columnDivider: columnDivider,
            fontStatistics: fontStats,
            headings: headings,
            elements: enhancedElements, // Add elements to the result for visualization
            sectionTexts: sectionTexts,
            processingTime: processingTime,
            usedFallback: false
        };
        
        return result;
    } catch (e) {
        console.error(`Error processing resume ${pdfPath}: ${e.message}`);
        console.error(e.stack);
        
        // Final fallback - if anything failed, try the text-based approach
        try {
            console.log(`Trying fallback parser after error in main parser for ${pdfPath}.`);
            return await fallbackParser.parseResumeTextBased(pdfPath);
        } catch (fallbackError) {
            console.error(`Both main and fallback parsers failed for ${pdfPath}`);
            return null;
        }
    }
}

// Visualize sections on the PDF
async function visualizeSections(pdfPath, result, outputFolder = null) {
    try {
        // Check if this is a fallback result
        if (result.usedFallback) {
            try {
                // Create a text-based visualization instead
                const visualizationContent = Object.entries(result.sectionTexts)
                    .map(([type, content]) => `=== ${type} ===\n${content}\n`)
                    .join('\n');
                
                // Determine output path
                let outPath;
                if (outputFolder) {
                    if (!fs.existsSync(outputFolder)) {
                        fs.mkdirSync(outputFolder, { recursive: true });
                    }
                    const baseName = path.basename(pdfPath);
                    outPath = path.join(outputFolder, path.parse(baseName).name + "_fallback_sections.txt");
                } else {
                    outPath = path.join(
                        path.dirname(pdfPath), 
                        path.parse(path.basename(pdfPath)).name + "_fallback_sections.txt"
                    );
                }
                
                // Write the text visualization
                fs.writeFileSync(outPath, visualizationContent);
                console.log(`Fallback text visualization saved to: ${outPath}`);
                
                return outPath;
            } catch (e) {
                console.error(`Error creating fallback visualization: ${e.message}`);
                return null;
            }
        } else {
            // For non-fallback results, we need to extract sections and do visualization
            const sections = extractSections(result.headings, result.elements, result.columnDivider);
            
            // Define colors for different section types (RGB, values from 0-1)
            const colors = {
                "EDUCATION": rgb(0, 0, 1),      // Blue
                "EXPERIENCE": rgb(0, 0.6, 0),   // Green
                "SKILLS": rgb(1, 0.5, 0),       // Orange
                "PROJECTS": rgb(0.5, 0, 0.5),   // Purple
                "CONTACT": rgb(0, 0.6, 0.6),    // Teal
                "SUMMARY": rgb(0.6, 0, 0),      // Dark Red
                "CERTIFICATIONS": rgb(0.7, 0.7, 0), // Olive
                "AWARDS": rgb(0.8, 0.4, 0),     // Brown
                "ACHIEVEMENTS": rgb(0, 0.4, 0.8), // Light Blue
                "LANGUAGES": rgb(0.5, 0.5, 0),  // Olive Green
                "PUBLICATIONS": rgb(0.8, 0.2, 0.2), // Red
                "ACTIVITIES": rgb(0.2, 0.5, 0.8), // Light Blue
                "REFERENCES": rgb(0.4, 0.4, 0.8), // Purple Blue
                "STRENGTH": rgb(0.6, 0.4, 0.2), // Brown
                "INTERESTS": rgb(0.3, 0.7, 0.3), // Light Green
                "OTHER": rgb(0.3, 0.3, 0.3)     // Dark Gray
            };
            
            // Read the original PDF file
            const pdfBytes = fs.readFileSync(pdfPath);
            const pdfDoc = await PDFDocument.load(pdfBytes);
            const pages = pdfDoc.getPages();
            
            // Load a font
            const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
            const boldFont = await pdfDoc.embedFont(StandardFonts.HelveticaBold);
            
            // Get PDF dimensions from the first page
            const { width, height } = pages[0].getSize();
            
            // Draw column divider on each page
            pages.forEach(page => {
                // Draw a dashed vertical line for the column divider
                page.drawLine({
                    start: { x: result.columnDivider, y: 0 },
                    end: { x: result.columnDivider, y: height },
                    thickness: 0.5,
                    color: rgb(0.5, 0.5, 0.5),
                    dashArray: [3, 3] // Create a dashed line
                });
            });
            
            // Draw heading labels and rectangles
            result.headings.forEach(heading => {
                if (heading.page >= pages.length) return; // Skip if page doesn't exist
                
                const page = pages[heading.page];
                const headingType = heading.type;
                const color = colors[headingType] || colors["OTHER"];
                
                // Convert y-coordinate (PDF uses bottom-left origin)
                const pdfY = height - heading.y0;
                
                // Draw a label above the heading
                page.drawText(`${headingType}`, {
                    x: heading.x0,
                    y: pdfY + 10, // Position above the heading
                    size: 8,
                    font: font,
                    color: color
                });
                
                // Draw a rectangle around the heading
                page.drawRectangle({
                    x: heading.x0 - 2,
                    y: pdfY - (heading.y1 - heading.y0), // Account for height
                    width: heading.x1 - heading.x0 + 4,
                    height: heading.y1 - heading.y0 + 4,
                    borderColor: color,
                    borderWidth: 1,
                    opacity: 0.9
                });
            });
            
            // Draw section content boxes
            Object.entries(sections).forEach(([sectionKey, sectionData]) => {
                const sectionType = sectionData.heading.type;
                const pageElements = {};
                
                // Group elements by page
                sectionData.elements.forEach(elem => {
                    if (!pageElements[elem.page]) {
                        pageElements[elem.page] = [];
                    }
                    pageElements[elem.page].push(elem);
                });
                
                // Process each page
                Object.entries(pageElements).forEach(([pageNum, elements]) => {
                    if (parseInt(pageNum) >= pages.length) return; // Skip if page doesn't exist
                    
                    const page = pages[parseInt(pageNum)];
                    
                    // Find the bounds of all elements in this section on this page
                    const x0 = Math.min(...elements.map(e => e.x0));
                    const y0 = Math.min(...elements.map(e => e.y0));
                    const x1 = Math.max(...elements.map(e => e.x1));
                    const y1 = Math.max(...elements.map(e => e.y1));
                    
                    // Get the color for this section
                    const color = colors[sectionType] || colors["OTHER"];
                    
                    // Draw a rectangle around the section content (with PDF coordinates)
                    page.drawRectangle({
                        x: x0 - 5,
                        y: height - y1 - 5, // Bottom of content
                        width: x1 - x0 + 10,
                        height: y1 - y0 + 10,
                        borderColor: color,
                        borderWidth: 1.5,
                        opacity: 0.2,
                        borderOpacity: 0.8,
                        borderDashArray: [5, 5] // Dashed border
                    });
                });
            });
            
            // Determine output path
            let outPath;
            if (outputFolder) {
                // Create output folder if it doesn't exist
                if (!fs.existsSync(outputFolder)) {
                    fs.mkdirSync(outputFolder, { recursive: true });
                }
                const baseName = path.basename(pdfPath);
                outPath = path.join(outputFolder, path.parse(baseName).name + "_highlighted.pdf");
            } else {
                outPath = path.join(
                    path.dirname(pdfPath), 
                    path.parse(path.basename(pdfPath)).name + "_highlighted.pdf"
                );
            }
            
            // Save the annotated PDF
            const pdfBytesModified = await pdfDoc.save();
            fs.writeFileSync(outPath, pdfBytesModified);
            console.log(`Annotated PDF saved to: ${outPath}`);
            
            return outPath;
        }
    } catch (e) {
        console.error(`Error creating visualization for ${pdfPath}: ${e.message}`);
        console.error(e.stack);
        return null;
    }
}

// Process a folder of resumes
async function processResumeFolder(folderPath, outputFolder = null, visualize = true) {
    // Setup output folder
    if (!outputFolder) {
        outputFolder = `${folderPath}_output`;
    }
    
    // Create output folder if it doesn't exist
    if (!fs.existsSync(outputFolder)) {
        fs.mkdirSync(outputFolder, { recursive: true });
    }
    
    // Get all PDF files in the folder
    const pdfFiles = fs.readdirSync(folderPath)
        .filter(file => file.toLowerCase().endsWith('.pdf'))
        .map(file => path.join(folderPath, file));
    
    if (pdfFiles.length === 0) {
        console.warn(`No PDF files found in ${folderPath}`);
        return [];
    }
    
    console.log(`Found ${pdfFiles.length} PDF files in ${folderPath}`);
    
    // Process files sequentially
    const results = [];
    const fallbackResults = [];
    
    for (let i = 0; i < pdfFiles.length; i++) {
        const pdfPath = pdfFiles[i];
        try {
            console.log(`\nProcessing file ${i+1}/${pdfFiles.length}: ${path.basename(pdfPath)}`);
            
            const result = await parseResume(pdfPath);
            if (result) {
                // Track whether fallback was used
                if (result.usedFallback) {
                    fallbackResults.push(pdfPath);
                    console.log(`Used fallback parser for: ${path.basename(pdfPath)}`);
                }
                
                // Create visualization if requested
                if (visualize) {
                    const outPath = await visualizeSections(pdfPath, result, outputFolder);
                    result.outPath = outPath;
                }
                
                // Save the full result as JSON
                const baseName = path.basename(pdfPath, '.pdf');
                const jsonPath = path.join(outputFolder, `${baseName}_parsed.json`);
                
                // Convert result to a serializable format
                const serializableResult = {
                    pdfPath: pdfPath,
                    outPath: result.outPath,
                    usedFallback: result.usedFallback,
                    columnDivider: result.columnDivider,
                    sectionTexts: result.sectionTexts,
                    headings: result.headings.map(h => ({
                        ...h,
                        confidence: parseFloat(h.confidence.toFixed(4))
                    })),
                    fontStatistics: result.fontStatistics ? Object.entries(result.fontStatistics).reduce((obj, [key, value]) => {
                        obj[key] = typeof value === 'number' ? parseFloat(value.toFixed(4)) : value;
                        return obj;
                    }, {}) : {},
                    processingTime: result.processingTime
                };
                
                fs.writeFileSync(jsonPath, JSON.stringify(serializableResult, null, 2));
                
                results.push(result);
                console.log(`Successfully processed ${path.basename(pdfPath)} - ` + 
                            `${result.usedFallback ? 'Using fallback parser' : `Found ${result.headings.length} headings`}`);
            } else {
                console.warn(`Failed to process ${path.basename(pdfPath)}`);
            }
        } catch (e) {
            console.error(`Error processing ${path.basename(pdfPath)}: ${e.message}`);
            console.error(e.stack);
        }
    }
    
    // Generate summary report with fallback information
    if (results.length > 0) {
        const summaryPath = path.join(outputFolder, "processing_summary.json");
        const summary = {
            totalFiles: pdfFiles.length,
            successfulFiles: results.length,
            fallbackFiles: fallbackResults.length,
            fallbackFilesList: fallbackResults.map(p => path.basename(p)),
            averageProcessingTime: results.reduce((sum, r) => sum + r.processingTime, 0) / results.length,
            averageSectionsFound: results.filter(r => !r.usedFallback).reduce((sum, r) => sum + r.headings.length, 0) / 
                                 (results.filter(r => !r.usedFallback).length || 1)
        };
        
        fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
        console.log(`\nProcessing complete. Summary saved to ${summaryPath}`);
        console.log(`Fallback parser used for ${fallbackResults.length} of ${pdfFiles.length} files.`);
    }
    
    return results;
}

// Example usage
async function main() {
    if (process.argv.length < 3) {
        console.log('Usage: node resumeParser.js <path_to_pdf_or_folder> [output_folder]');
        process.exit(1);
    }
    
    const inputPath = process.argv[2];
    const outputFolder = process.argv.length >= 4 ? process.argv[3] : null;
    
    try {
        const stats = fs.statSync(inputPath);
        
        if (stats.isDirectory()) {
            // Process all PDFs in the folder
            const results = await processResumeFolder(inputPath, outputFolder);
            if (results.length > 0) {
                console.log(`Processing complete. Processed ${results.length} of ${fs.readdirSync(inputPath).filter(f => f.toLowerCase().endsWith('.pdf')).length} files.`);
            } else {
                console.log("No resumes were successfully processed.");
            }
        } else if (inputPath.toLowerCase().endsWith('.pdf')) {
            // Process a single PDF
            const result = await parseResume(inputPath);
            if (result) {
                console.log(`Resume successfully processed: ${path.basename(inputPath)}`);
                console.log(`Found ${result.headings.length} sections: ${result.headings.map(h => h.type).join(', ')}`);
                
                // Save output if output folder specified
                if (outputFolder) {
                    if (!fs.existsSync(outputFolder)) {
                        fs.mkdirSync(outputFolder, { recursive: true });
                    }
                    
                    const baseName = path.basename(inputPath, '.pdf');
                    const jsonPath = path.join(outputFolder, `${baseName}_parsed.json`);
                    
                    fs.writeFileSync(jsonPath, JSON.stringify({
                        pdfPath: inputPath,
                        columnDivider: result.columnDivider,
                        sectionTexts: result.sectionTexts,
                        headings: result.headings.map(h => ({
                            ...h,
                            confidence: parseFloat(h.confidence.toFixed(4))
                        })),
                        fontStatistics: Object.entries(result.fontStatistics).reduce((obj, [key, value]) => {
                            obj[key] = typeof value === 'number' ? parseFloat(value.toFixed(4)) : value;
                            return obj;
                        }, {}),
                        processingTime: result.processingTime
                    }, null, 2));
                    
                    console.log(`Result saved to: ${jsonPath}`);
                }
            } else {
                console.error(`Failed to process resume: ${inputPath}`);
            }
        } else {
            console.error(`Error: ${inputPath} is not a PDF file or a directory`);
        }
    } catch (e) {
        console.error(`Error: ${e.message}`);
        console.error(e.stack);
    }
}

// Export functions for module usage
module.exports = {
    parseResume,
    processResumeFolder,
    extractTextWithStyle,
    identifyPotentialHeadings,
    identifySectionHeadings,
    extractSections,
    extractSectionText
};

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}