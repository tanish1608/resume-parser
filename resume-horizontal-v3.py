import pdfplumber
import fitz  # PyMuPDF
import os
import re
from collections import defaultdict
import time
import logging
import numpy as np
import json  
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants for section detection
MIN_HEADING_LENGTH = 2  # Minimum length for heading text
LINE_SPACING_TOLERANCE = 2  # Tolerance for grouping elements into lines (pixels)
# Adjust this constant in your code
CONFIDENCE_THRESHOLD = 0.75 # Previously was likely 0.8

# Enhanced Section keywords - expanded with international and abbreviated variations
SECTION_KEYWORDS = {
    "OBJECTIVE": [
        "objective", "career objective", "professional objective", "job objective",
        "employment objective", "career goal", "professional goal", "career aim",
        "professional summary", "personal statement", "career summary"
    ],
    # Standard sections
    "EDUCATION": [
        # English variations
        "education", "academic", "academics", "qualification", "qualifications", "educational", "edu", "schooling", "school", "studies", "study", "learning", "degree", "degrees", "university", 
        "universities", "college", "colleges", "educational background", "academic background", "educational qualifications", "academic record", "academic credentials",
        "academic history", "scholastic record", "academic profile", "university education", "college education", "school education", "tertiary education", "higher education"
    ],
    
    "EXPERIENCE": [
        # English variations
        "experience", "experiences", "work", "workexperience", "work-experience", "work/experience", "employment", "career", "job", "jobs", "workhistory", "work history", "professional", "prof", 
        "professionalexperience", "professional background", "employment history", "internship", "internships", "exp", "wrk", "work exp", "prof exp", "occupation", "occupations", "positions", "Work Experience"
    ],
    
    "SKILLS": [
        # English variations
        "skills", "skill", "abilities", "ability", "expertise", "competencies", "competences", "capabilities", "capability", "proficiencies", "proficiency", "talents", "talent", "aptitudes", 
        "technicalskills", "technical skills", "softskill", "soft skills", "hardskills", "hard skills", "core competencies", "skls", "proficiencies", "competency", "tech skills"
    ],
    
    "PROJECTS": [
        # English variations
        "projects", "project", "works", "portfolio", "showcase", "achievements", "initiatives", "implementations", "developments", "applications", "software projects", "development projects", 
        "proj", "key projects", "personal projects", "academic projects", "research projects"
    ],
    
    "CONTACT": [
        # English variations
        "contact", "contacts", "contact information", "contact details", "contact info", "personal","personal information", "personal details", "personal info", "personal data", "personaldetails", 
        "personal-details", "info", "details", "get in touch", "reach me", "contact me", "connect"
    ],
    
    "SUMMARY": [
        # English variations
        "summary", "professional summary", "career summary", "profile", "professional profile", "overview", "about", "about me", "introduction", "bio", "biography", "background", "brief",
        "executive summary", "career objective", "career goal", "careerobjective", "objective", "objective", "summary of qualifications", "qualifications summary", "career profile", "summ",
        "curriculum", "vitae", "cv", "resume summary", "professional background", "candidate summary"
    ],
    
    "CERTIFICATIONS": [
        # English variations
        "certifications", "certification", "certificates", "certificate","CERTIFICATES", "credentials", "licenses", "license", "accreditation", "accreditations", "qualification", "qualifications", "certs", 
        "certified", "diplomas", "diploma", "professional certifications", "professional development", 
    ],
    
    "TECHNICAL_SKILLS": [
        "technical skills", "technical", "tech skills", "technical expertise", "technical competencies",
        "hard skills", "programming skills", "computer skills", "development skills", "engineering skills",
        "tech stack", "technical proficiencies", "technical capabilities"
    ],
    
    "SOFT_SKILLS": [
        "soft skills", "interpersonal skills", "people skills", "communication skills", "personal skills",
        "soft abilities", "professional skills", "transferable skills", "personal attributes", 
        "personal qualities", "behavioral skills"
    ],
    
    "LEADERSHIP": [
        "leadership", "leadership & event participation", "leadership and event participation",
        "leadership experience", "leadership roles", "leadership positions", "event participation",
        "leadership activities", "leadership & activities", "leadership and activities",
        "leadership skills", "leadership qualities"
    ],
    
    "HACKATHONS": [
        "hackathons", "hackathon", "hackathon experience", "hackathon participation", 
        "hack-a-thons", "coding competitions", "coding contests", "programming competitions",
        "tech competitions", "development contests"
    ],
    
    "CAREER_OBJECTIVE": [
        "career objective", "objective", "career goal", "professional objective", "job objective",
        "employment objective", "career aim", "professional goal", "career summary", "professional summary",
        "career aspiration", "professional aspiration", "objective statement"
    ],
    
    "INTERNSHIPS": [
        "internships", "internship", "intern", "internship experience", "professional internships",
        "internship program", "industrial training", "summer internship", "winter internship",
        "internships and professional experience", "professional experience", "internship and professional experience"
    ],
    
    "EXTRACURRICULAR": [
        "extracurricular", "extracurricular activities", "extra-curricular", "extra curricular",
        "extraciruluar", "co-curricular", "co curricular", "after-school activities",
        "out-of-class activities", "outside activities", "campus involvement", "student organizations",
        "clubs and organizations", "campus activities", "student activities"
    ],
    
    "ACADEMIC_ACHIEVEMENTS": [
        "academic achievements", "academic accomplishments", "academic honors",
        "academic awards", "educational achievements", "academic distinctions",
        "academic and extracurricular achievements", "academic and extracurricular accomplishments",
        "academic recognitions", "scholastic achievements", "educational honors",
        "academic performance", "academic excellence", "academic merit"
    ],
    
    "AWARDS": [  # Expand existing AWARDS category
        "awards", "award", "honors", "honour", "achievements", "recognition", "accolades",  "prizes", "prize", "distinctions", "accomplishments", "scholarships", "fellowship", "grant", "grants", "scholarship", "awrds", "achievements", "recognitions", "accolades", "honors and awards",
        "awards and achievements", "awards & achievements", "awards and recognitions", "recognitions and awards", "honors & awards", "distinctions and honors"
    ],

    
    "ACHIEVEMENTS": [
        # English variations
        "achievements", "achievement", "accomplishments", "accomplishment", "successes", "success", "milestones", "highlights", "key achievements", "significant achievements", "major accomplishments",
        "achvmts", "success stories", "key accomplishments", "professional achievements", "notable achievements"
    ],
    
    "LANGUAGES": [
        # English variations
        "languages", "language", "language skills", "language proficiency", "language proficiencies", "language abilities", "spoken languages", "linguistic skills", "idiomas", "foreign languages", "lang"
    ],
    
    "PUBLICATIONS": [
        # English variations
        "publications", "publication", "papers", "paper", "articles", "article", "journal articles", "research papers", "published works", "books", "book", "conference proceedings", "conf. proc.", 
        "pubs", "published papers", "research publications", "academic publications", "scientific publications"
    ],
    
    "ACTIVITIES": [
        # English variations
        "activities", "activity", "extracurricular", "extra-curricular", "extracurricular activities", "co-curricular", "cocurricular", "volunteer", "volunteering", "community service", "leadership","involvement", "interests", "hobbies", "hobby", "activities & interests", "activities and interests",
        "extra", "volunteer work", "community involvement", "participation","leadership & activities", "campus involvement"
    ],
    
    "REFERENCES": [
        # English variations
        "references", "reference", "referees", "referee", "recommendations", "recommendation", 
        "recommenders", "recommender", "professional references", "refs", "testimonials", "endorsements"],
    
    "STRENGTH": [
        # English variations
        "strength", "strengths", "key strengths", "personal strengths", "professional strengths", 
        "core strengths", "strength areas", "strong points", "strong suits", "forte", "strong areas"
    ],
    
    "INTERESTS": [
        # English variations
        "interests", "interest", "hobbies", "hobby", "passions", "passion", "personal interests", 
        "leisure activities", "recreational activities", "pastimes", "pastime", "interests & hobbies",
        "interests and hobbies", "intrsts", "personal activities", "recreational pursuits"
    ],
    
    "DECLARATION": [
        # English variations
        "declaration", "decleration", "verification", "authentication", "confirmation", "affirmation", 
        "authorization", "declaration of authenticity", "decl", "statement", "personal declaration"
    ],
    
    "OTHER": [
        # Limit to very specific section header terms only
        "miscellaneous", "additional information", "additional", "more information",
        "addendum", "appendix", "supplementary"
        # Remove generic terms like "other", "others", "plus", "extra" etc.
    ]
}

# Flatten the section keywords for easier lookup
FLAT_SECTION_KEYWORDS = {}
for section_type, keywords in SECTION_KEYWORDS.items():
    for keyword in keywords:
        FLAT_SECTION_KEYWORDS[keyword] = section_type

def normalize(text):
    """Normalize text by removing whitespace and converting to lowercase."""
    # Handle None or empty text
    if not text:
        return ""
    # Convert to lowercase and remove all whitespace
    return re.sub(r'\s+', '', text).lower()

def preprocess_text(text):
    """Preprocess text to handle special characters and unicode."""
    if not text:
        return ""
    # Replace common unicode characters
    replacements = {
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2022': '*',  # bullet
        '\u2026': '...',  # ellipsis
        '\u00a0': ' ',  # non-breaking space
        '\u00ad': '-',  # soft hyphen
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    return text.strip()

def is_heading_text(text, section_keywords=None):
    """
    Improved heading detection with better confidence scoring.
    
    Args:
        text (str): The text to check
        section_keywords (dict, optional): Dictionary of section keywords
    
    Returns:
        tuple: (is_heading (bool), confidence_score (float), section_type (str))
    """
    if not section_keywords:
        section_keywords = FLAT_SECTION_KEYWORDS
    
    # Normalize and preprocess the text for comparison
    clean = normalize(preprocess_text(text))
    original_text = text.strip().lower()
    
    if not clean or len(clean) < MIN_HEADING_LENGTH:
        return False, 0.0, None
    
    # Direct match first (highest confidence)
    if clean in section_keywords:
        return True, 1.0, section_keywords[clean]
    
    # Check for full matches with common patterns
    # Match common patterns like "Technical Skills:" or "EDUCATION"
    section_patterns = {
        r"\b(education|academic|degree|qualification)s?\b": "EDUCATION",
        r"\b(experience|work|employment|job|career|professional)s?\b": "EXPERIENCE",
        r"\b(skill|ability|proficiency|competency|expertise)s?\b": "SKILLS",
        r"\b(technical|hard|programming|computer|software|coding)\s+skill": "SKILLS",
        r"\b(soft|personal|interpersonal|communication)\s+skill": "SKILLS",
        r"\b(project|portfolio|work|assignment|application)s?\b": "PROJECTS",
        r"\b(certification|certificate|credential|qualification)s?\b": "CERTIFICATIONS",
        r"\b(award|honor|achievement|recognition|scholarship)s?\b": "AWARDS",
        r"\b(language|linguistic|idioma|tongue)\b": "LANGUAGES",
        r"\b(objective|goal|summary|profile|about)s?\b": "SUMMARY",
        r"\b(leadership|participation|involvement|activity)s?\b": "ACTIVITIES",
        r"\b(hackathon|competition|contest|challenge|event)s?\b": "ACTIVITIES",
        r"\b(reference|recommendation|endorsement|referee)s?\b": "REFERENCES",
        r"\b(interest|hobby|pastime|leisure|extracurricular)s?\b": "INTERESTS",
        r"\b(strength|forte|capability|talent|power)s?\b": "STRENGTH",
        r"\b(contact|detail|information|reach|connect)\b": "CONTACT",
        r"\b(publication|paper|article|journal|research)s?\b": "PUBLICATIONS",
        r"\b(internship|training|practical)s?\b": "EXPERIENCE"
    }
    
    # Check patterns
    for pattern, section_type in section_patterns.items():
        if re.search(pattern, original_text, re.IGNORECASE):
            # Return high confidence if it's a strong match
            match_strength = len(re.findall(pattern, original_text, re.IGNORECASE))
            if match_strength > 0:
                return True, min(0.95, 0.75 + match_strength * 0.1), section_type
    
    # Check for partial matches with standard approach
    matches = []
    
    # Check if any keyword is contained in the text
    for keyword, section_type in section_keywords.items():
        if keyword in clean:
            # Calculate confidence based on keyword length and position
            # Exact match at start of text gets highest confidence
            if clean.startswith(keyword):
                confidence = 0.9
            else:
                # Otherwise, calculate based on keyword coverage
                confidence = len(keyword) / len(clean) * 0.8
            matches.append((confidence, section_type))
    
    # Check if the text is contained in any keyword (abbreviated form)
    if len(clean) >= MIN_HEADING_LENGTH + 1:
        for keyword, section_type in section_keywords.items():
            if clean in keyword and len(clean) >= len(keyword) * 0.5:
                # Confidence based on how much of the keyword is covered
                confidence = len(clean) / len(keyword) * 0.85
                matches.append((confidence, section_type))
    
    # Special check for title formats with colons 
    # e.g., "Skills:" or "Technical Experience:"
    if ":" in original_text:
        prefix = original_text.split(":")[0].strip().lower()
        if len(prefix) >= MIN_HEADING_LENGTH:
            # Check if this prefix matches any keywords
            for keyword, section_type in section_keywords.items():
                if prefix.endswith(keyword) or keyword.endswith(prefix):
                    # Confidence based on length match and position
                    confidence = min(0.9, 0.7 + len(prefix) / 20)
                    matches.append((confidence, section_type))
    
    # Find the best match
    if matches:
        best_match = max(matches, key=lambda x: x[0])
        # Lower the threshold for detection to catch more headings
        if best_match[0] >= CONFIDENCE_THRESHOLD - 0.1:
            return True, best_match[0], best_match[1]
    
    return False, 0.0, None

def extract_text_with_style(pdf_path):
    """Extract text with style information using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(pdf_path)
        all_elements = []
        try:
            for page_num, page in enumerate(doc):
                try:
                    blocks = page.get_text("dict")["blocks"]
                    
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                    if "spans" in line:
                                        for span in line["spans"]:
                                            text = span.get("text", "").strip()
                                            if not text:
                                                continue
                                            
                                            # Clean and preprocess the text
                                            text = preprocess_text(text)                                    
                                            font_name = span.get("font", "")
                                            font_size = span.get("size", 0)
                                            is_bold = (
                                                "bold" in font_name.lower() or 
                                                "heavy" in font_name.lower() or
                                                "black" in font_name.lower() or
                                                "extrabold" in font_name.lower()
                                            )
                                            is_capital = text.isupper() and len(text) > 2  # Check if text is all uppercase
                                            is_spaced = " " in text and all(len(part) == 1 for part in text.split())  # S P A C E D text
                                            
                                            all_elements.append({
                                                "page": page_num,
                                                "text": text,
                                                "x0": span["bbox"][0],
                                                "y0": span["bbox"][1],
                                                "x1": span["bbox"][2],
                                                "y1": span["bbox"][3],
                                                "font_name": font_name,
                                                "font_size": font_size,
                                                "is_bold": is_bold,
                                                "is_capital": is_capital,
                                                "is_spaced": is_spaced,
                                                "is_likely_heading": False,  # Will be determined after analyzing font sizes
                                                "confidence": 0.0,  # Will store confidence in heading detection
                                                "section_type": None  # Will store detected section type
                                            })
                except Exception as e:
                    logger.warning(f"Error processing page {page_num} in {pdf_path}: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to extract with 'dict' method: {str(e)}")
        
        doc.close()
        return all_elements
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        return []

def analyze_font_statistics(elements):
    """
    Analyze font size distribution and determine statistical thresholds.
    
    Args:
        elements (list): List of text elements with font information
        
    Returns:
        dict: Font statistics including median, mean, percentiles, etc.
    """
    if not elements:
        return {
            "mean": 0,
            "median": 0,
            "std": 0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "min": 0,
            "max": 0,
            "most_common": 0,
            "heading_threshold": 0
        }
    
    # Extract font sizes
    font_sizes = [elem["font_size"] for elem in elements if elem["font_size"] > 0]
    
    if not font_sizes:
        return {
            "mean": 0,
            "median": 0,
            "std": 0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "min": 0,
            "max": 0,
            "most_common": 0,
            "heading_threshold": 0
        }
    
    # Calculate statistics
    mean_size = np.mean(font_sizes)
    median_size = np.median(font_sizes)
    std_size = np.std(font_sizes)
    
    # Calculate percentiles
    p25 = np.percentile(font_sizes, 25)
    p50 = np.percentile(font_sizes, 50)
    p75 = np.percentile(font_sizes, 75)
    p90 = np.percentile(font_sizes, 90)
    p95 = np.percentile(font_sizes, 95)
    
    # Find min and max
    min_size = np.min(font_sizes)
    max_size = np.max(font_sizes)
    
    # Find most common font size (mode)
    unique_sizes, counts = np.unique(font_sizes, return_counts=True)
    most_common = unique_sizes[np.argmax(counts)]
    
    # Calculate a good threshold for heading detection
    # We use a percentile-based approach: headings are usually in the top 25% of font sizes
    # But we make sure it's at least 10% bigger than the most common font size
    heading_threshold = max(p75, most_common * 1.1)
    
    return {
        "mean": mean_size,
        "median": median_size,
        "std": std_size,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "min": min_size,
        "max": max_size,
        "most_common": most_common,
        "heading_threshold": heading_threshold
    }

def detect_column_whitespace(elements, pdf_path):
    """
    Detect columns by finding vertical whitespace gaps in the document.
    
    Args:
        elements (list): List of text elements
        pdf_path (str): Path to the PDF file
        
    Returns:
        float: X-coordinate that separates left and right columns
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_width = pdf.pages[0].width
            page_height = pdf.pages[0].height
    except Exception as e:
        logger.error(f"Failed to open PDF with pdfplumber: {str(e)}")
        page_width = 612  # Default
        page_height = 792  # Default
    
    # Skip header/footer by focusing on the middle 80% of the page
    top_margin = page_height * 0.15
    bottom_margin = page_height * 0.8
    
    # Only include elements in the main content area (exclude header/footer)
    content_elements = [e for e in elements if top_margin <= e["y0"] <= bottom_margin]
    
    if len(content_elements) < 10:
        return page_width  # Default if not enough elements
    
    # Divide the page width into bins (e.g., 50 bins for more granularity)
    num_bins = 50
    bin_width = page_width / num_bins
    
    # Create a density map for the x-axis
    x_density = [0] * num_bins
    
    # Calculate density
    for elem in content_elements:
        # Mark bins that this element covers
        start_bin = max(0, min(num_bins - 1, int(elem["x0"] / bin_width)))
        end_bin = max(0, min(num_bins - 1, int(elem["x1"] / bin_width)))
        
        for bin_idx in range(start_bin, end_bin + 1):
            x_density[bin_idx] += 1
    
    # Find the longest continuous whitespace gap in the middle 60% of the page
    middle_start = int(num_bins * 0.3)  # Start at 20% of page width
    middle_end = int(num_bins * 0.9)    # End at 80% of page width
    
    max_gap_length = 0
    max_gap_center = num_bins // 2  # Default to middle
    
    current_gap_start = None
    
    for i in range(middle_start, middle_end):
        # If bin is empty (or nearly empty), it's part of a gap
        if x_density[i] <= len(content_elements) * 0.02:  # Allow some noise (2%)
            if current_gap_start is None:
                current_gap_start = i
        else:
            # End of a gap
            if current_gap_start is not None:
                gap_length = i - current_gap_start
                if gap_length > max_gap_length:
                    max_gap_length = gap_length
                    max_gap_center = (current_gap_start + i) / 2
                current_gap_start = None
    
    # If we ended with an open gap, close it
    if current_gap_start is not None:
        gap_length = middle_end - current_gap_start
        if gap_length > max_gap_length:
            max_gap_length = gap_length
            max_gap_center = (current_gap_start + middle_end) / 2
    
    # If we found a significant gap (at least 5% of page width), use it
    if max_gap_length >= num_bins * 0.01:
        return max_gap_center * bin_width
    
    # Fallback to previous method if no clear gap is found
    return page_width 

def fallback_rank_based_heading_detection(elements, column_divider):
    """
    Fallback function for heading detection based on font size ranking.
    This function is used when standard methods detect 2 or fewer headings.
    
    Args:
        elements (list): List of text elements
        column_divider (float): X-coordinate dividing the columns
        
    Returns:
        list: List of identified section headings
    """
    logger.info("Using fallback rank-based heading detection")
    
    # Filter valid elements (non-empty with positive font size)
    valid_elements = [
        e for e in elements 
        if e["font_size"] > 0 and len(e["text"].strip()) >= MIN_HEADING_LENGTH
    ]
    
    if not valid_elements:
        logger.warning("No valid elements found for fallback detection")
        return []
    
    # Create a list of unique font sizes in descending order
    font_sizes = sorted(set(e["font_size"] for e in valid_elements), reverse=True)
    
    # Log font size distribution
    logger.info(f"Font sizes in document (descending): {[round(fs, 2) for fs in font_sizes]}")
    
    # Determine key ranks for heading detection
    # Typically, the largest font (rank 0) is the document title
    # Second largest (rank 1) and third largest (rank 2) are often section headers
    key_ranks = [1, 2]  # Focus on these ranks
    
    # For very simple documents, we might only have 2-3 font sizes
    if len(font_sizes) <= 3:
        # Adjust to include more ranks for simple documents
        key_ranks = list(range(min(len(font_sizes), 3)))
    
    # Map each font size to its rank (0 = largest)
    size_to_rank = {size: rank for rank, size in enumerate(font_sizes)}
    
    # Find elements with key ranks
    candidate_headings = []
    
    for elem in valid_elements:
        rank = size_to_rank.get(elem["font_size"], -1)
        
        # Calculate a base confidence score
        if rank in key_ranks:
            # Higher confidence for preferred ranks
            base_confidence = 0.8 if rank == 1 else 0.75
        elif rank == 0:  # Largest font - likely document title
            base_confidence = 0.7
        elif rank < 5:  # Other large fonts
            base_confidence = 0.6
        else:  # Smaller fonts
            continue  # Skip smaller fonts entirely
        
        # Adjust confidence based on styling
        style_boost = 0.0
        
        # Bold text
        if elem["is_bold"]:
            style_boost += 0.15
            
        # ALL CAPS
        if elem["is_capital"] and len(elem["text"].strip()) >= 3:
            style_boost += 0.1
            
        # Ends with colon (e.g., "Skills:")
        if elem["text"].strip().endswith(":"):
            style_boost += 0.1
            
        # Short text (likely a heading)
        text_length = len(elem["text"].strip())
        if text_length < 30:
            length_factor = max(0, 0.1 - (text_length / 300))
            style_boost += length_factor
            
        # Left-aligned
        if elem["x0"] < 70:  # Near left margin
            style_boost += 0.05
        
        # Check for common section keywords
        common_keywords = [
            "education", "experience", "skills", "project", "certification",
            "award", "achievement", "language", "objective", "summary", 
            "profile", "contact", "reference", "interest", "publication",
            "qualification", "training", "leadership", "activity", "strength",
            "expertise", "technical"
        ]
        
        text_lower = elem["text"].lower()
        keyword_boost = 0.0
        
        if any(keyword in text_lower for keyword in common_keywords):
            keyword_boost = 0.15
            
        # Try formal section type detection
        is_heading, keyword_confidence, section_type = is_heading_text(elem["text"])
        if not section_type:
            print(f"Section type not detected for text: {elem['text']}")
            # If no formal section type detected, use fallback logic            
            # Try to infer section type from text
            if "education" in text_lower or "academic" in text_lower:
                section_type = "EDUCATION"
            elif "experience" in text_lower or "work" in text_lower or "employment" in text_lower:
                section_type = "EXPERIENCE"
            elif "skill" in text_lower or "technical" in text_lower or "expertise" in text_lower:
                section_type = "SKILLS"
            elif "project" in text_lower:
                section_type = "PROJECTS"
            elif "certification" in text_lower or "certificate" in text_lower:
                section_type = "CERTIFICATIONS"
            elif "award" in text_lower or "achievement" in text_lower or "honor" in text_lower:
                section_type = "AWARDS"
            elif "language" in text_lower:
                section_type = "LANGUAGES"
            elif "objective" in text_lower or "summary" in text_lower or "profile" in text_lower:
                section_type = "SUMMARY"
            elif "contact" in text_lower or "personal" in text_lower:
                section_type = "CONTACT"
        
        # Final confidence calculation
        final_confidence = min(1.0, base_confidence + style_boost + keyword_boost)
        
        # Determine which column this heading belongs to
        column = "left" if elem["x0"] < column_divider else "right"
        
        # Add to candidates if confidence is sufficient
        if final_confidence >= 0.95:  # Lower threshold for fallback detection
            candidate_headings.append({
                "page": elem["page"],
                "text": elem["text"],
                "x0": elem["x0"],
                "y0": elem["y0"],
                "x1": elem["x1"],
                "y1": elem["y1"],
                "type": section_type,
                "column": column,
                "confidence": final_confidence,
                "rank": rank
            })
    
    # If we have too many candidates, filter to keep only the most confident ones
    if len(candidate_headings) > 15:  # Cap at a reasonable number
        # Sort by confidence and take the top ones
        candidate_headings.sort(key=lambda h: h["confidence"], reverse=True)
        candidate_headings = candidate_headings[:15]
    
    # Sort headings by page and y-coordinate for the final output
    headings = sorted(candidate_headings, key=lambda h: (h["page"], h["y0"]))
    
    logger.info(f"Fallback detection found {len(headings)} potential section headings")
    
    return headings

def identify_potential_headings(elements, font_stats):
    """
    Identify potential headings with improved confidence scoring.
    
    Args:
        elements (list): List of text elements
        font_stats (dict): Font statistics from analyze_font_statistics
        
    Returns:
        list: Updated elements with heading likelihood scores
    """
    if not elements or not font_stats:
        return elements
    
    # Calculate the document-wide font statistics
    avg_font_size = font_stats["mean"]
    max_font_size = font_stats["max"]
    most_common_size = font_stats["most_common"]
    median_size = font_stats["median"]
    
    # Adjust the heading threshold to be more inclusive
    # Instead of using a fixed threshold above the mean, use a more adaptive approach
    heading_base_threshold = max(most_common_size * 1.05, median_size * 1.1)
    
    # Find all bold text elements - we'll analyze their sizes to better understand heading patterns
    bold_elements = [e for e in elements if e["is_bold"]]
    bold_font_sizes = [e["font_size"] for e in bold_elements if e["font_size"] > 0]
    
    # If there are enough bold elements, calculate their own statistics
    if len(bold_font_sizes) >= 5:
        bold_median_size = np.median(bold_font_sizes)
        # Lower the threshold if bold text is used strategically
        heading_threshold = min(heading_base_threshold, bold_median_size * 0.95)
    else:
        heading_threshold = heading_base_threshold
    
    # First pass: Reset all elements
    for elem in elements:
        elem["is_likely_heading"] = False
        elem["confidence"] = 0.0
        elem["section_type"] = None
    
    # Second pass: Calculate confidence scores with improved algorithm
    for elem in elements:
        # Skip very short text (likely not headings)
        if len(elem["text"].strip()) < MIN_HEADING_LENGTH:
            continue
            
        # Base confidence starts at 0
        confidence_score = 0.0
        
        # 1. Font size factor: How much larger is this than the document's typical text?
        # Using a smoother scaling function
        if elem["font_size"] > 0:
            size_ratio = elem["font_size"] / most_common_size
            # Exponential scaling with smoothing
            font_size_confidence = min(1.0, max(0, (size_ratio - 1) * 1.5))
        else:
            font_size_confidence = 0.0
            
        # 2. Styling factor: Bold, all caps, and positioning
        style_confidence = 0.0
        
        # Bold text is very indicative of headings
        if elem["is_bold"]:
            style_confidence += 0.4
            
        # All caps are often used for headings
        if elem["is_capital"] and len(elem["text"]) >= 3:
            style_confidence += 0.25
            
        # Check if text appears to be a single line (may be a heading)
        if "\n" not in elem["text"] and len(elem["text"].split()) <= 5:
            style_confidence += 0.15
            
        # Position factor - headings often start at the left margin or page edge
        # This doesn't require column_divider - we just check if it's near either edge
        page_start = 50  # Assume page margin is within 50 pixels
        if elem["x0"] <= page_start:
            style_confidence += 0.1
            
        # 3. Keyword matching - is this text similar to common section headers?
        is_heading, keyword_confidence, section_type = is_heading_text(elem["text"])
        
        # 4. Length factor - headings are typically short
        # Use an exponential decay function - shorter text gets higher scores
        text_length = len(elem["text"].strip())
        if text_length <= 30:
            length_factor = 1.0  # Perfect length
        else:
            # Exponential decay for longer text
            length_factor = max(0.4, math.exp(-0.02 * (text_length - 30)))
            
        # Combined confidence score with weighted factors
        # The weights sum to 1.0
        confidence_score = (
            keyword_confidence * 0.35 +  # Keywords are strong indicators
            font_size_confidence * 0.3 +  # Font size is very important
            style_confidence * 0.25 +     # Styling matters a lot
            length_factor * 0.1           # Length is a secondary factor
        )
        
        # Apply a non-linear scaling to boost mid-to-high confidence scores
        # This helps prevent the "low confidence" issue
        if confidence_score > 0.5:
            # Apply a positive bias to scores above 0.5
            confidence_score = 0.5 + (confidence_score - 0.5) * 1.5
            
        # Ensure we don't exceed 1.0
        confidence_score = min(1.0, confidence_score)
        
        # Apply threshold with reduced stringency
        if confidence_score >= CONFIDENCE_THRESHOLD - 0.1:  # More inclusive
            elem["is_likely_heading"] = True
            elem["confidence"] = confidence_score
            elem["section_type"] = section_type
            
            # Extra boost for extremely strong signals
            if (elem["is_bold"] and elem["is_capital"] and 
                font_size_confidence > 0.4 and keyword_confidence > 0.6):
                elem["confidence"] = min(1.0, confidence_score * 1.2)
    
    return elements
      

def identify_section_headings(elements, column_divider):
    """
    Identify section headings with improved confidence handling.
    
    Args:
        elements (list): List of text elements with heading likelihood scores
        column_divider (float): X-coordinate dividing the columns
        
    Returns:
        list: List of identified section headings
    """
    headings = []
    
    # Sort elements first by confidence score (descending)
    sorted_elements = sorted(elements, key=lambda e: e["confidence"], reverse=True)
    
    # Track which text positions we've already marked as headings
    heading_positions = set()
    
    # First pass - collect all high-confidence headings
    high_confidence_threshold = 0.75
    for elem in sorted_elements:
        # Only consider high-confidence elements in this pass
        if elem["confidence"] < high_confidence_threshold or not elem["is_likely_heading"]:
            continue
        
        # Skip if we've already included this position as a heading
        position_key = (elem["page"], round(elem["y0"]), round(elem["x0"]))
        if position_key in heading_positions:
            continue
        
        # Skip very short text
        if len(elem["text"].strip()) < MIN_HEADING_LENGTH:
            continue
        
        # Determine which column this heading belongs to
        column = "left" if elem["x0"] < column_divider else "right"
        
        # Add to headings list
        headings.append({
            "page": elem["page"],
            "text": elem["text"],
            "x0": elem["x0"],
            "y0": elem["y0"],
            "x1": elem["x1"],
            "y1": elem["y1"],
            "type": elem["section_type"] or "OTHER",
            "column": column,
            "confidence": elem["confidence"]
        })
        
        # Mark this position as used
        heading_positions.add(position_key)
    
    # Second pass - collect medium confidence headings
    for elem in sorted_elements:
        # Skip if not likely a heading or high confidence (already processed)
        if not elem["is_likely_heading"] or elem["confidence"] >= high_confidence_threshold:
            continue
        
        # Skip if we've already included this position as a heading
        position_key = (elem["page"], round(elem["y0"]), round(elem["x0"]))
        if position_key in heading_positions:
            continue
        
        # Skip very short text
        if len(elem["text"].strip()) < MIN_HEADING_LENGTH:
            continue
        
        # Apply formatting-based confidence boost
        boosted_confidence = elem["confidence"]
        
        # Boost confidence if this follows typical heading patterns
        
        # 1. Boost for all caps text
        if elem["is_capital"] and len(elem["text"]) >= 3:
            boosted_confidence = min(0.9, boosted_confidence + 0.1)
            
        # 2. Boost for text with colon ending (e.g., "Skills:")
        if elem["text"].strip().endswith(":"):
            boosted_confidence = min(0.9, boosted_confidence + 0.15)
            
        # 3. Boost for short text (likely a heading)
        if len(elem["text"].strip()) < 20:
            boost_factor = 0.05 + max(0, (20 - len(elem["text"].strip())) / 100)
            boosted_confidence = min(0.9, boosted_confidence + boost_factor)
            
        # 4. Boost for text at the left margin
        column_margin = column_divider * 0.1 if elem["x0"] < column_divider else column_divider * 1.1
        if abs(elem["x0"] - column_margin) < 20:
            boosted_confidence = min(0.9, boosted_confidence + 0.05)
        
        # Determine which column this heading belongs to
        column = "left" if elem["x0"] < column_divider else "right"
        
        # Add to headings list
        headings.append({
            "page": elem["page"],
            "text": elem["text"],
            "x0": elem["x0"],
            "y0": elem["y0"],
            "x1": elem["x1"],
            "y1": elem["y1"],
            "type": elem["section_type"] or "OTHER",
            "column": column,
            "confidence": boosted_confidence  # Use the boosted confidence
        })
        
        # Mark this position as used
        heading_positions.add(position_key)
    
    # Third pass - boost confidence for any headings that follow heading patterns
    # Look for structural patterns in the document
    if headings:
        # Sort headings by page and y-coordinate 
        sorted_headings = sorted(headings, key=lambda h: (h["page"], h["y0"]))
        
        # Find common x-coordinates for headings
        x_coordinates = [h["x0"] for h in sorted_headings]
        common_x = None
        if len(x_coordinates) >= 3:
            # Use a simple clustering approach
            x_clusters = {}
            tolerance = 10  # pixels
            for x in x_coordinates:
                matched = False
                for center in x_clusters:
                    if abs(x - center) <= tolerance:
                        x_clusters[center] += 1
                        matched = True
                        break
                if not matched:
                    x_clusters[x] = 1
            
            # Find the most common x-coordinate
            if x_clusters:
                common_x = max(x_clusters.items(), key=lambda x: x[1])[0]
        
        # Boost confidence for headings at common x-coordinates
        if common_x is not None:
            for heading in headings:
                if abs(heading["x0"] - common_x) <= 10 and heading["confidence"] < 0.9:
                    heading["confidence"] = min(0.9, heading["confidence"] + 0.15)
    
    # Sort headings by page and y-coordinate for final output
    return sorted(headings, key=lambda h: (h["page"], h["y0"]))



def post_process_headings(headings, elements):
    """
    Apply post-processing to ensure consistency in heading detection.
    
    Args:
        headings (list): List of identified section headings
        elements (list): List of all text elements
        
    Returns:
        list: Updated headings with improved confidence
    """
    if not headings or len(headings) < 2:
        return headings
    
    # Group headings by page for analysis
    headings_by_page = {}
    for h in headings:
        page = h["page"]
        if page not in headings_by_page:
            headings_by_page[page] = []
        headings_by_page[page].append(h)
    
    # Check for consistent formatting within each page
    for page, page_headings in headings_by_page.items():
        if len(page_headings) < 2:
            continue
        
        # Analyze format consistency
        is_bold_counts = defaultdict(int)
        is_caps_counts = defaultdict(int)
        font_sizes = []
        
        for heading in page_headings:
            # Find the corresponding element
            for elem in elements:
                if (elem["page"] == heading["page"] and 
                    abs(elem["y0"] - heading["y0"]) < 2 and 
                    abs(elem["x0"] - heading["x0"]) < 2):
                    
                    is_bold_counts[elem["is_bold"]] += 1
                    is_caps_counts[elem["is_capital"]] += 1
                    if elem["font_size"] > 0:
                        font_sizes.append(elem["font_size"])
                    break
        
        # Determine the dominant formatting
        dominant_bold = max(is_bold_counts.items(), key=lambda x: x[1])[0] if is_bold_counts else None
        dominant_caps = max(is_caps_counts.items(), key=lambda x: x[1])[0] if is_caps_counts else None
        avg_font_size = np.mean(font_sizes) if font_sizes else None
        
        # Boost confidence for headings that match the dominant formatting
        for heading in page_headings:
            # Find the corresponding element again
            for elem in elements:
                if (elem["page"] == heading["page"] and 
                    abs(elem["y0"] - heading["y0"]) < 2 and 
                    abs(elem["x0"] - heading["x0"]) < 2):
                    
                    format_matches = 0
                    if dominant_bold is not None and elem["is_bold"] == dominant_bold:
                        format_matches += 1
                    if dominant_caps is not None and elem["is_capital"] == dominant_caps:
                        format_matches += 1
                    if avg_font_size is not None and abs(elem["font_size"] - avg_font_size) < 2:
                        format_matches += 1
                    
                    # Boost confidence based on format consistency
                    if format_matches >= 2 and heading["confidence"] < 0.9:
                        heading["confidence"] = min(0.95, heading["confidence"] + 0.2)
                    break
    
    return headings

def extract_sections(pdf_path, headings, elements, column_divider):
    """
    Extract sections from the PDF, handling the multi-column layout.
    Enhanced with improved error handling and confidence scores.
    
    Args:
        pdf_path (str): Path to the PDF file
        headings (list): List of identified section headings
        elements (list): List of text elements
        column_divider (float): X-coordinate dividing the columns
        
    Returns:
        dict: Dictionary of extracted sections
    """
    sections = {}
    
    if not headings:
        logger.warning(f"No headings found in {pdf_path}")
        return sections
    
    # Separate headings by column
    left_headings = [h for h in headings if h["column"] == "left"]
    right_headings = [h for h in headings if h["column"] == "right"]
    
    # Sort headings in each column by page and vertical position
    left_headings = sorted(left_headings, key=lambda h: (h["page"], h["y0"]))
    right_headings = sorted(right_headings, key=lambda h: (h["page"], h["y0"]))
    
    # Group elements by page for faster access
    elements_by_page = defaultdict(list)
    for elem in elements:
        elements_by_page[elem["page"]].append(elem)
    
    # Process each column separately
    for column, col_headings in [("left", left_headings), ("right", right_headings)]:
        for i, heading in enumerate(col_headings):
            heading_page = heading["page"]
            heading_y = heading["y1"]  # Bottom of the heading
            
            # Find the next heading in the same column
            end_y = float('inf')
            end_page = float('inf')
            
            if i < len(col_headings) - 1:
                next_heading = col_headings[i + 1]
                end_y = next_heading["y0"]
                end_page = next_heading["page"]
            
            # Collect all elements in this section
            section_elements = []
            
            # Define the column boundaries
            min_x = 0 if column == "left" else column_divider
            max_x = column_divider if column == "left" else float('inf')
            
            # Process elements page by page
            for page_num in range(heading_page, min(end_page + 1, max(elements_by_page.keys()) + 1)):
                if page_num not in elements_by_page:
                    continue
                
                for elem in elements_by_page[page_num]:
                    # Check if the element is in the correct column
                    elem_in_column = min_x <= elem["x0"] < max_x
                    
                    # Skip if not in this column
                    if not elem_in_column:
                        continue
                    
                    # Skip if this is a heading element
                    is_heading_elem = False
                    for h in headings:
                        if (elem["page"] == h["page"] and 
                            abs(elem["y0"] - h["y0"]) < 2 and 
                            abs(elem["x0"] - h["x0"]) < 2):
                            is_heading_elem = True
                            break
                    
                    if is_heading_elem:
                        continue
                    
                    # If on heading page, only include elements after the heading
                    if page_num == heading_page and elem["y0"] < heading_y:
                        continue
                    
                    # If on end page, only include elements before the next heading
                    if page_num == end_page and elem["y0"] >= end_y:
                        continue
                    
                    section_elements.append(elem)
            
            # Generate a unique section key
            section_key = f"{heading['type']}_{heading['page']}_{column}_{i}"
            
            # Store the section
            sections[section_key] = {
                "heading": heading,
                "elements": section_elements,
                "column": column,
                "confidence": heading["confidence"]
            }
    
    return sections

def extract_section_text(sections):
    """
    Convert the section elements into readable text using improved line detection.
    
    Args:
        sections (dict): Dictionary of extracted sections
        
    Returns:
        dict: Dictionary mapping section types to extracted text
    """
    section_texts = {}
    
    for section_key, section_data in sections.items():
        elements = section_data["elements"]
        section_type = section_data["heading"]["type"]
        column = section_data["column"]
        confidence = section_data["confidence"]
        
        if not elements:
            continue
        
        # Group elements by page
        elements_by_page = defaultdict(list)
        for elem in elements:
            elements_by_page[elem["page"]].append(elem)
        
        # Process each page separately
        text_lines_by_page = []
        
        for page_num in sorted(elements_by_page.keys()):
            page_elements = elements_by_page[page_num]
            
            # Calculate the average font size for this page's elements
            font_sizes = [e["font_size"] for e in page_elements if e["font_size"] > 0]
            avg_font_size = np.mean(font_sizes) if font_sizes else 10  # Default if no valid font sizes
            
            # Use the average font size to determine line spacing tolerance
            # Larger font sizes typically have larger line spacing
            line_spacing = max(LINE_SPACING_TOLERANCE, avg_font_size * 0.3)
            
            # Sort elements by y position
            sorted_y = sorted(page_elements, key=lambda e: e["y0"])
            
            # Group elements into lines using dynamic tolerance
            lines = []
            if sorted_y:  # Check if there are any elements
                current_line = [sorted_y[0]]
                current_y = sorted_y[0]["y0"]
                
                for elem in sorted_y[1:]:
                    # If this element is within the tolerance of the current line, add it
                    if abs(elem["y0"] - current_y) <= line_spacing:
                        current_line.append(elem)
                    else:
                        # Sort the current line elements by x position
                        current_line.sort(key=lambda e: e["x0"])
                        lines.append(current_line)
                        
                        # Start a new line
                        current_line = [elem]
                        current_y = elem["y0"]
                
                # Add the last line
                if current_line:
                    current_line.sort(key=lambda e: e["x0"])
                    lines.append(current_line)
            
            # Build text lines for this page
            page_text_lines = []
            for line in lines:
                line_text = ' '.join(elem["text"] for elem in line)
                # Clean up extra spaces
                line_text = re.sub(r'\s+', ' ', line_text).strip()
                if line_text:  # Only add non-empty lines
                    page_text_lines.append(line_text)
            
            # Add page number if there are multiple pages
            if len(elements_by_page) > 1:
                page_text_lines.insert(0, f"[Page {page_num+1}]")
            
            text_lines_by_page.extend(page_text_lines)
        
        # Join all lines with newlines
        # Use a simple section key for better readability
        simple_key = section_type
        section_content = '\n'.join(text_lines_by_page)
        
        # Add column information
        section_header = f"--- {column.upper()} COLUMN ---\n"
        
        if simple_key in section_texts:
            # If we already have this section type, append new content
            section_texts[simple_key] += "\n\n" + section_header + section_content
        else:
            section_texts[simple_key] = section_header + section_content
    
    return section_texts

def visualize_sections(pdf_path, sections, headings, column_divider, output_folder=None):
    """
    Create a visualized PDF with highlighted sections, showing the column structure.
    
    Args:
        pdf_path (str): Path to the PDF file
        sections (dict): Dictionary of extracted sections
        headings (list): List of identified section headings
        column_divider (float): X-coordinate dividing the columns
        output_folder (str, optional): Output folder for the visualized PDF
        
    Returns:
        str: Path to the visualized PDF
    """
    # Colors for different section types (RGB, values from 0-1)
    colors = {
        "EDUCATION": (0, 0, 1),      # Blue
        "EXPERIENCE": (0, 0.6, 0),   # Green
        "SKILLS": (1, 0.5, 0),       # Orange
        "PROJECTS": (0.5, 0, 0.5),   # Purple
        "CONTACT": (0, 0.6, 0.6),    # Teal
        "SUMMARY": (0.6, 0, 0),      # Dark Red
        "CERTIFICATIONS": (0.7, 0.7, 0), # Olive
        "AWARDS": (0.8, 0.4, 0),     # Brown
        "ACHIEVEMENTS": (0, 0.4, 0.8), # Light Blue
        "LANGUAGES": (0.5, 0.5, 0),  # Olive Green
        "PUBLICATIONS": (0.8, 0.2, 0.2), # Red
        "ACTIVITIES": (0.2, 0.5, 0.8), # Light Blue
        "REFERENCES": (0.4, 0.4, 0.8), # Purple Blue
        "STRENGTH": (0.6, 0.4, 0.2), # Brown
        "INTERESTS": (0.3, 0.7, 0.3), # Light Green
        "DECLARATION": (0.4, 0.4, 0.4), # Gray
        "OTHER": (0.3, 0.3, 0.3)     # Dark Gray
    }
    
    try:
        doc = fitz.open(pdf_path)
        output = fitz.open()
        
        # Create a new PDF with highlighted sections
        for page_num in range(len(doc)):
            # Create a new page
            orig_page = doc[page_num]
            new_page = output.new_page(width=orig_page.rect.width, height=orig_page.rect.height)
            new_page.show_pdf_page(orig_page.rect, doc, page_num)
            
            # Draw column divider line
            new_page.draw_line((column_divider, 0), (column_divider, orig_page.rect.height), 
                               color=(0.5, 0.5, 0.5), width=0.5, dashes="[2 2]")
            
            # Draw heading labels
            for heading in [h for h in headings if h["page"] == page_num]:
                heading_type = heading["type"]
                color = colors.get(heading_type, (0.3, 0.3, 0.3))
                
                # Draw a label above the heading
                label_text = f"{heading_type}"
                label_point = fitz.Point(heading["x0"], heading["y0"] - 5)
                new_page.insert_text(label_point, label_text, fontsize=8, color=color)
                
                # Draw a rectangle around the heading
                rect = fitz.Rect(heading["x0"] - 2, heading["y0"] - 2, 
                               heading["x1"] + 2, heading["y1"] + 2)
                new_page.draw_rect(rect, color=color, width=1)
            
            # Draw section content boxes
            for section_key, section_data in sections.items():
                section_type = section_data["heading"]["type"]
                
                # Filter elements to this page
                page_elements = [e for e in section_data["elements"] if e["page"] == page_num]
                if not page_elements:
                    continue
                    
                # Find the bounds of all elements in this section on this page
                x0 = min(e["x0"] for e in page_elements)
                y0 = min(e["y0"] for e in page_elements)
                x1 = max(e["x1"] for e in page_elements)
                y1 = max(e["y1"] for e in page_elements)
                
                # Get the color for this section
                color = colors.get(section_type, (0.3, 0.3, 0.3))
                
                # Draw a rectangle around the section content
                rect = fitz.Rect(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
                new_page.draw_rect(rect, color=color, width=1.5, dashes="[2 2]")
        
        # Determine output path
        if output_folder:
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            base_name = os.path.basename(pdf_path)
            out_path = os.path.join(output_folder, os.path.splitext(base_name)[0] + "_highlighted.pdf")
        else:
            out_path = os.path.splitext(pdf_path)[0] + "_highlighted.pdf"
        
        # Save the output PDF
        output.save(out_path)
        logger.info(f"Annotated PDF saved to: {out_path}")
        
        # Clean up
        output.close()
        doc.close()
        
        return out_path
    
    except Exception as e:
        logger.error(f"Error creating visualization for {pdf_path}: {str(e)}")
        return None

def parse_resume(pdf_path, output_folder=None, visualize=True):
    """
    Parse a resume PDF file, extract sections from multi-column layout, and return structured data.
    Enhanced with confidence scoring and better font analysis.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str, optional): Output folder for visualized PDF
        visualize (bool): Whether to create visualization
        
    Returns:
        dict: Parsed resume data
    """
    try:
        start_time = time.time()
        logger.info(f"Processing resume: {pdf_path}")
        
        # Extract text elements with style information
        elements = extract_text_with_style(pdf_path)
        if not elements:
            logger.error(f"Failed to extract text elements from {pdf_path}")
            return None
        
        logger.info(f"Extracted {len(elements)} text elements")
        
        # Analyze font statistics
        font_stats = analyze_font_statistics(elements)
        logger.info(f"Font statistics: mean={font_stats['mean']:.2f}, median={font_stats['median']:.2f}, "
                    f"max={font_stats['max']:.2f}, threshold={font_stats['heading_threshold']:.2f}")
        
        # Identify potential headings with confidence scores
        elements = identify_potential_headings(elements, font_stats)
        potential_headings = [e for e in elements if e["is_likely_heading"]]
        logger.info(f"Identified {len(potential_headings)} potential headings")
        
        # Analyze the layout to detect columns
        column_divider = detect_column_whitespace(elements, pdf_path)
        logger.info(f"Detected column divider at x-coordinate: {column_divider:.2f}")
        
        # Identify section headings with confidence scores
        headings = identify_section_headings(elements, column_divider)
        logger.info(f"Identified {len(headings)} section headings")

        headings = post_process_headings(headings, elements)
        logger.info(f"After post-processing: {len(headings)} section headings")
        
        # NEW: If 2 or fewer headings found, apply fallback detection
        if len(headings) <= 2:
            logger.warning(f"Only {len(headings)} headings found, applying fallback detection")
            fallback_headings = fallback_rank_based_heading_detection(elements, column_divider)
            
            if fallback_headings and len(fallback_headings) > len(headings):
                headings = fallback_headings
                logger.info(f"Using {len(headings)} headings from fallback detection")
        
        
        # Count headings in each column
        left_headings = [h for h in headings if h["column"] == "left"]
        right_headings = [h for h in headings if h["column"] == "right"]
        logger.info(f"  - Left column: {len(left_headings)} headings")
        logger.info(f"  - Right column: {len(right_headings)} headings")
        
        # Log each heading with confidence
        for heading in headings:
            rank_info = f", rank: {heading.get('rank', 'N/A')}" if "rank" in heading else ""
            logger.info(f"  - {heading['text']} ({heading['type']}, {heading['column']} column, "
                       f"confidence: {heading.get('confidence', 0.0):.2f}{rank_info})")
        
        # Extract sections with improved error handling
        sections = extract_sections(pdf_path, headings, elements, column_divider)
        logger.info(f"Extracted {len(sections)} sections")
        
        # Extract text from each section with improved line grouping
        section_texts = extract_section_text(sections)
        
        # Create visualization if requested
        out_path = None
        if visualize:
            out_path = visualize_sections(pdf_path, sections, headings, column_divider, output_folder)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare the result
        result = {
            "pdf_path": pdf_path,
            "out_path": out_path,
            "column_divider": column_divider,
            "font_statistics": font_stats,
            "headings": headings,
            "section_texts": section_texts,
            "processing_time": processing_time
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing resume {pdf_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_resume_folder(folder_path, output_folder=None, visualize=True):
    """
    Process all PDF files in a folder to identify and extract resume sections.
    Enhanced with better error handling and progress reporting.
    
    Args:
        folder_path (str): Path to the folder containing resume PDFs
        output_folder (str, optional): Path to save output files. If None, creates folder_path + "_output"
        visualize (bool): Whether to create visualization PDFs
        
    Returns:
        list: List of processed file paths and their extracted sections
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Error: {folder_path} is not a valid directory")
        return []
    
    # Setup output folder
    if output_folder is None:
        output_folder = folder_path + "_output"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PDF files in the folder
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
    
    # Process files sequentially
    results = []
    
    for i, pdf_path in enumerate(pdf_files):
        try:
            logger.info(f"Processing file {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
            
            result = parse_resume(pdf_path, output_folder, visualize)
            if result:
                # Save the full result as JSON
                base_name = os.path.basename(pdf_path)
                json_path = os.path.join(output_folder, os.path.splitext(base_name)[0] + "_parsed.json")
                
                # Prepare serializable version of the result
                serializable_result = {
                    'pdf_path': pdf_path,
                    'out_path': result.get('out_path'),
                    'column_divider': float(result.get('column_divider', 0)),
                    'section_texts': result.get('section_texts', {}),
                }
                
                # Convert headings to serializable format (handle numpy values)
                headings = []
                for h in result.get('headings', []):
                    serializable_heading = {}
                    for k, v in h.items():
                        if isinstance(v, np.number):
                            serializable_heading[k] = float(v)
                        else:
                            serializable_heading[k] = v
                    headings.append(serializable_heading)
                
                serializable_result['headings'] = headings
                
                # Convert font statistics to serializable format
                font_stats = {}
                for k, v in result.get('font_statistics', {}).items():
                    if isinstance(v, np.number):
                        font_stats[k] = float(v)
                    else:
                        font_stats[k] = v
                
                serializable_result['font_statistics'] = font_stats
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_result, f, indent=2, ensure_ascii=False)
                
                results.append(result)
                logger.info(f"Successfully processed {os.path.basename(pdf_path)} - "
                           f"Found {len(result.get('headings', []))} headings")
            else:
                logger.warning(f"Failed to process {os.path.basename(pdf_path)}")
        
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Generate summary report
    if results:
        summary_path = os.path.join(output_folder, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(pdf_files),
                'successful_files': len(results),
                'average_processing_time': sum(r['processing_time'] for r in results) / len(results) if results else 0,
                'average_sections_found': sum(len(r.get('headings', [])) for r in results) / len(results) if results else 0,
            }, f, indent=2)
        
        logger.info(f"Processing complete. Summary saved to {summary_path}")
    
    return results

def main():
    """Main function to run the resume parser from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Resume PDF Parser')
    parser.add_argument('input_path', help='Path to a resume PDF file or a folder containing PDFs')
    parser.add_argument('--output', '-o', help='Output folder for results (default: input_path + "_output")')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization step')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    input_path = args.input_path
    output_folder = args.output
    visualize = not args.no_visualize
    
    if os.path.isdir(input_path):
        # Process all PDFs in the folder
        results = process_resume_folder(input_path, output_folder, visualize)
        if results:
            print(f"Processing complete. Processed {len(results)} of {len([f for f in os.listdir(input_path) if f.lower().endswith('.pdf')])} files.")
            print(f"Check the output folder for details: {output_folder or (input_path + '_output')}")
        else:
            print("No resumes were successfully processed.")
    else:
        # Process a single PDF
        result = parse_resume(input_path, output_folder, visualize)
        if result:
            print(f"Resume successfully processed: {os.path.basename(input_path)}")
            print(f"Found {len(result.get('headings', []))} sections: {', '.join(h['type'] for h in result.get('headings', []))}")
            if result['out_path']:
                print(f"Annotated PDF saved to: {result['out_path']}")
        else:
            print(f"Failed to process resume: {input_path}")

if __name__ == "__main__":
    main()