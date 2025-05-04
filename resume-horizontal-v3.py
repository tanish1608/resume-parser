import pdfplumber
import fitz  # PyMuPDF
import os
import re
import logging
import time
import json
from collections import defaultdict
import sys
import numpy as np
import concurrent.futures
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MIN_HEADING_LENGTH = 3  # Minimum length for h in headings if h["column"] == "left"]
COLUMN_DETECTION_THRESHOLD = 0.2  # Percentage of page width to determine column separation
FONT_SIZE_THRESHOLD_FACTOR = 1.2  # Multiplier for average font size to identify headings
LINE_SPACING_TOLERANCE = 2  # Tolerance for grouping elements into lines (pixels)
MAX_WORKER_THREADS = 4  # Maximum number of threads for parallel processing

# Section keywords - expanded with international and industry-specific variations
HEADING_KEYWORDS = [
    # Standard sections
    "education", "skills", "projects", "certifications", "contact", "languages",
    "objective", "experience", "workexperience", "work-experience", "work/experience",
    "accomplishments", "awards", "careergoals", "careerobjective", "declaration",
    "decleration", "profile", "summary", "achievements", "references", "referee",
    # International variations
    "curriculum", "vitae", "cv", "personalinformation", "personaldetails", "personal-details",
    "qualifications", "professional", "formation", "academic", "bildung", "educación",
    # Industry-specific sections
    "publications", "research", "patents", "volunteering", "community", "service",
    "leadership", "activities", "extracurricular", "interests", "hobbies", "portfolio",
    "technical", "softskill", "competencies", "expertise", "strengths", "tools",
    "technologies", "software", "programming", "frameworks", "methodologies",
    # Additional variations
    "workhistory", "employment", "professionalexperience", "internship", "training",
    "scholarship", "honors", "fellowship", "conference", "presentation", "speaker", 
    "exhibition", "laboratory", "clinical", "fieldwork", "specializations"
]

def normalize(text):
    """Normalize text by removing whitespace and converting to lowercase."""
    # Handle None or empty text
    if not text:
        return ""
    return re.sub(r'\s+', '', text).lower()

def preprocess_text(text):
    """Preprocess text to handle special characters and unicode."""
    if not text:
        return ""
    # Replace common unicode characters
    text = text.replace('\u2013', '-')  # en dash
    text = text.replace('\u2014', '-')  # em dash
    text = text.replace('\u2018', "'")  # left single quote
    text = text.replace('\u2019', "'")  # right single quote
    text = text.replace('\u201c', '"')  # left double quote
    text = text.replace('\u201d', '"')  # right double quote
    text = text.replace('\u2022', '*')  # bullet
    text = text.replace('\u2026', '...')  # ellipsis
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    return text.strip()

def is_heading_text(text):
    """Check if the text is a common resume section heading."""
    # Normalize and preprocess the text for comparison
    clean = normalize(preprocess_text(text))
    
    # Check if the normalized text contains any heading keyword
    return any(kw in clean for kw in HEADING_KEYWORDS)

def get_heading_type(text):
    """Determine which type of heading the text represents."""
    clean = normalize(preprocess_text(text))
    
    # Map section types to their keywords - using 'in' for partial matching
    section_types = {
        "EDUCATION": ["education", "academic", "bildung", "educación", "formation", "qualifications"],
        "EXPERIENCE": ["experience", "workexperience", "career", "employment", "workhistory", 
                      "professionalexperience", "work-experience", "work/experience", "internship"],
        "SKILLS": ["skills", "technicalskills", "softskill", "competencies", "expertise", 
                  "strengths", "tools", "technologies", "software", "programming", "frameworks"],
        "PROJECTS": ["projects", "portfolio", "research"],
        "CONTACT": ["contact", "personalinformation", "personaldetails", "personal-details"],
        "SUMMARY": ["summary", "profile", "objective", "careerobjective", "professional", "curriculum", "vitae", "cv"],
        "CERTIFICATIONS": ["certifications", "certification", "credentials", "qualifications"],
        "AWARDS": ["awards", "honors", "scholarship", "fellowship"],
        "ACHIEVEMENTS": ["achievements", "accomplishments"],
        "LANGUAGES": ["languages"],
        "PUBLICATIONS": ["publications", "conference", "presentation", "speaker"],
        "ACTIVITIES": ["activities", "extracurricular", "interests", "hobbies", "volunteering", "community", "service", "leadership"],
        "REFERENCES": ["references", "referee"],
        "DECLARATION": ["declaration", "decleration"]
    }
    
    for section_name, keywords in section_types.items():
        if any(kw in clean for kw in keywords):
            return section_name
    
    return "OTHER"

def extract_text_with_style(pdf_path):
    """Extract text with style information using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(pdf_path)
        all_elements = []
        
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
                                    is_bold = "bold" in font_name.lower() or "heavy" in font_name.lower()
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
                                        "is_likely_heading": False  # Will be determined after analyzing font sizes
                                    })
            except Exception as e:
                logger.warning(f"Error processing page {page_num} in {pdf_path}: {str(e)}")
                continue
        
        doc.close()
        return all_elements
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        return []

def analyze_font_sizes(elements):
    """
    Analyze font sizes in the document to determine relative thresholds for headings.
    Updates the is_likely_heading property of each element.
    """
    if not elements:
        return elements
    
    # Extract all font sizes
    font_sizes = [elem["font_size"] for elem in elements if elem["font_size"] > 0]
    
    if not font_sizes:
        return elements
    
    # Calculate statistics
    avg_font_size = np.mean(font_sizes)
    std_font_size = np.std(font_sizes)
    
    # Define dynamic threshold (larger than average)
    threshold = avg_font_size * FONT_SIZE_THRESHOLD_FACTOR
    
    # Update elements with heading likelihood
    for elem in elements:
        # Detect if this might be a heading based on styling
        is_likely_heading = (
            (elem["is_bold"] and is_heading_text(elem["text"])) or 
            (elem["is_capital"] and is_heading_text(elem["text"])) or
            (elem["font_size"] > threshold and is_heading_text(elem["text"])) or
            (elem["is_spaced"] and len(elem["text"]) > 5)  # Spaced out text like "E D U C A T I O N"
        )
        elem["is_likely_heading"] = is_likely_heading
    
    return elements

def analyze_layout(elements, pdf_path):
    """
    Analyze the resume layout to detect columns using histogram-based analysis.
    This method is more robust than the original and doesn't require external libraries.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_width = pdf.pages[0].width
    except Exception as e:
        logger.error(f"Failed to open PDF with pdfplumber: {str(e)}")
        # Default to 612 (standard letter width in points)
        page_width = 612
    
    # Get all x-coordinates
    x_coords = [elem["x0"] for elem in elements]
    
    if len(x_coords) < 10:  # Not enough data points for analysis
        return page_width * 0.6
    
    # Create histogram to find columns
    # Divide page width into bins (e.g., 20 bins)
    num_bins = 20
    bin_width = page_width / num_bins
    
    # Create histogram
    histogram = [0] * num_bins
    for x in x_coords:
        bin_idx = min(int(x / bin_width), num_bins - 1)
        histogram[bin_idx] += 1
    
    # Find peaks in histogram (potential column starts)
    peaks = []
    for i in range(1, num_bins - 1):
        if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1] and histogram[i] > len(elements) / 50:
            peaks.append((i, histogram[i]))
    
    # Sort peaks by frequency (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # If we have at least two significant peaks, they could represent columns
    if len(peaks) >= 2:
        # Get the two most significant peaks
        top_peaks = peaks[:2]
        top_peaks.sort(key=lambda x: x[0])  # Sort by bin index
        
        # Calculate the midpoint between the two most significant columns
        col1_center = (top_peaks[0][0] + 0.5) * bin_width
        col2_center = (top_peaks[1][0] + 0.5) * bin_width
        
        # Check if the columns are far enough apart
        if col2_center - col1_center > page_width * COLUMN_DETECTION_THRESHOLD:
            # Return midpoint between columns as divider
            return (col1_center + col2_center) / 2
    
    # Default to 60% of page width if we can't clearly detect columns
    return page_width * 0.6

def identify_section_headings(elements, column_divider):
    """
    Identify section headings based on text styling and patterns.
    Takes into account the multi-column layout.
    """
    headings = []
    
    # First pass - identify clear headings
    for i, elem in enumerate(elements):
        text = elem["text"].strip()
        
        # Skip short text
        if len(text) < MIN_HEADING_LENGTH:
            continue
        
        # Check if this element is likely a heading
        if elem["is_likely_heading"]:
            heading_type = get_heading_type(text)
            
            # Determine which column this heading belongs to
            column = "left" if elem["x0"] < column_divider else "right"
            
            headings.append({
                "page": elem["page"],
                "text": text,
                "x0": elem["x0"],
                "y0": elem["y0"],
                "x1": elem["x1"],
                "y1": elem["y1"],
                "type": heading_type,
                "column": column
            })
    
    return headings

def extract_sections(pdf_path, headings, elements, column_divider):
    """
    Extract sections from the PDF, handling the multi-column layout.
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
                    if any(elem["x0"] == h["x0"] and elem["y0"] == h["y0"] for h in headings):
                        continue
                    
                    # If on heading page, only include elements after the heading
                    if page_num == heading_page and elem["y0"] < heading_y:
                        continue
                    
                    # If on end page, only include elements before the next heading
                    if page_num == end_page and elem["y0"] >= end_y:
                        continue
                    
                    section_elements.append(elem)
            
            # Store the section
            sections[f"{heading['type']}_{column}"] = {
                "heading": heading,
                "elements": section_elements,
                "column": column
            }
    
    return sections

def extract_section_text(sections):
    """
    Convert the section elements into readable text.
    Uses improved line detection based on y-coordinates and font size.
    """
    section_texts = {}
    
    for section_key, section_data in sections.items():
        elements = section_data["elements"]
        section_type = section_data["heading"]["type"]
        column = section_data["column"]
        
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
                page_text_lines.append(line_text)
            
            # Add page number if there are multiple pages
            if len(elements_by_page) > 1:
                page_text_lines.insert(0, f"[Page {page_num+1}]")
            
            text_lines_by_page.extend(page_text_lines)
        
        # Join all lines with newlines
        # Use a simple section key for better readability
        simple_key = section_type
        section_content = '\n'.join(text_lines_by_page)
        
        if simple_key in section_texts:
            # If we already have this section type, append new content
            section_texts[simple_key] += "\n\n--- " + column.upper() + " COLUMN ---\n" + section_content
        else:
            section_texts[simple_key] = "--- " + column.upper() + " COLUMN ---\n" + section_content
    
    return section_texts

def extract_section_structured_content(section_texts):
    """
    Process section texts into structured content based on section type.
    Performs specialized extraction based on the section.
    """
    structured_content = {}
    
    for section_type, text in section_texts.items():
        # Skip empty sections
        if not text:
            continue
        
        # Store basic content
        structured_content[section_type] = {
            "raw_text": text,
            "extracted_data": {}
        }
        
        # Add section-specific extraction
        if section_type == "EDUCATION":
            # Extract education details (degrees, institutions, dates)
            degrees = re.findall(r'(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?|M\.?B\.?A\.?|Bachelor|Master|Doctor|Diploma)', text)
            institutions = re.findall(r'(?:University|College|Institute|School) of [A-Za-z\s]+', text)
            dates = re.findall(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4} - (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\d{4} - \d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4} - (?:Present|Current|Now)', text)
            
            structured_content[section_type]["extracted_data"] = {
                "degrees": degrees,
                "institutions": institutions,
                "dates": dates
            }
            
        elif section_type == "EXPERIENCE":
            # Extract job titles, companies, and dates
            job_titles = re.findall(r'(?:Senior|Junior|Lead|Principal|Chief|Head)? ?(?:Developer|Engineer|Analyst|Manager|Director|Consultant|Designer|Architect|Administrator|Coordinator|Specialist|Associate)', text)
            companies = re.findall(r'[A-Z][a-z]+ (?:Inc|LLC|Ltd|Limited|Corporation|Corp|Company|Co)\.?', text)
            dates = re.findall(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4} - (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\d{4} - \d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4} - (?:Present|Current|Now)', text)
            
            structured_content[section_type]["extracted_data"] = {
                "job_titles": job_titles,
                "companies": companies,
                "dates": dates
            }
            
        elif section_type == "SKILLS":
            # Extract technical skills, programming languages, etc.
            programming_languages = re.findall(r'(?:Python|Java|JavaScript|C\+\+|C#|SQL|PHP|Ruby|Swift|Kotlin|Go|Rust|TypeScript|HTML|CSS|R|MATLAB|Scala|Perl|Shell|Bash|PowerShell)', text)
            frameworks = re.findall(r'(?:React|Angular|Vue|Django|Flask|Spring|Hibernate|Laravel|Ruby on Rails|Express|Node\.js|TensorFlow|PyTorch|Keras|Pandas|NumPy|Scikit-learn)', text)
            tools = re.findall(r'(?:Git|Docker|Kubernetes|AWS|Azure|GCP|Jenkins|Travis CI|CircleCI|Jira|Confluence|Tableau|Power BI|Excel|Photoshop|Illustrator|Figma|Sketch)', text)
            
            structured_content[section_type]["extracted_data"] = {
                "programming_languages": programming_languages,
                "frameworks": frameworks,
                "tools": tools
            }
    
    return structured_content

def visualize_sections(pdf_path, sections, headings, column_divider, output_folder=None):
    """
    Create a visualized PDF with highlighted sections, showing the column structure.
    Now with configurable output folder.
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
                label_point = fitz.Point(heading["x0"], heading["y0"] - 5)
                new_page.insert_text(label_point, heading_type, fontsize=8, color=color)
                
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

def process_resume(pdf_path, output_folder=None, visualize=True):
    """
    Parse a resume PDF file, extract sections from multi-column layout, and return structured data.
    Now with optional visualization.
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
        
        # Analyze font sizes and update heading likelihood
        elements = analyze_font_sizes(elements)
        
        # Analyze the layout to detect columns using improved methods
        column_divider = analyze_layout(elements, pdf_path)
        
        logger.info(f"Detected column divider at x-coordinate: {column_divider}")
        
        # Identify section headings, taking into account the column structure
        headings = identify_section_headings(elements, column_divider)
        logger.info(f"Identified {len(headings)} section headings:")
        
        # Count headings in each column
        left_headings = [h for h in headings if h["column"] == "left"]   
        right_headings = [h for h in headings if h["column"] == "right"]
        logger.info(f"  - Left column: {len(left_headings)} headings")
        logger.info(f"  - Right column: {len(right_headings)} headings")
        
        for heading in headings:
            logger.info(f"  - {heading['text']} ({heading['type']}, {heading['column']} column)")
        
        # Extract sections, handling the multi-column layout
        sections = extract_sections(pdf_path, headings, elements, column_divider)
        logger.info(f"Extracted {len(sections)} sections")
        
        # Extract text from each section
        section_texts = extract_section_text(sections)
        
        # Extract structured content from sections
        structured_content = extract_section_structured_content(section_texts)
        
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
            "headings": headings,
            "section_texts": section_texts,
            "structured_content": structured_content,
            "processing_time": processing_time
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing resume {pdf_path}: {str(e)}")
        return None

def process_resume_folder(folder_path, output_folder=None, visualize=True, max_workers=None):
    """
    Process all PDF files in a folder to identify and extract resume sections.
    Now with parallel processing and progress reporting.
    
    Args:
        folder_path (str): Path to the folder containing resume PDFs
        output_folder (str, optional): Path to save output files. If None, creates folder_path + "_output"
        visualize (bool): Whether to create visualization PDFs
        max_workers (int, optional): Maximum number of worker threads for parallel processing
    
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
    
    # Use the minimum of MAX_WORKER_THREADS, the number of CPUs, and the requested max_workers
    if max_workers is None:
        max_workers = min(MAX_WORKER_THREADS, os.cpu_count() or 4)
    else:
        max_workers = min(max_workers, MAX_WORKER_THREADS, os.cpu_count() or 4)
    
    # Process files in parallel
    results = []
    
    if max_workers > 1:
        logger.info(f"Processing with {max_workers} worker threads")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {executor.submit(process_resume, pdf_path, output_folder, visualize): pdf_path 
                            for pdf_path in pdf_files}
            
            # Process as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_pdf)):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    if result:
                        results.append({
                            'pdf_path': pdf_path,
                            'out_path': result.get('out_path'),
                            'sections': list(result.get('section_texts', {}).keys()),
                            'processing_time': result.get('processing_time', 0)
                        })
                        
                        # Save the full result as JSON
                        base_name = os.path.basename(pdf_path)
                        json_path = os.path.join(output_folder, os.path.splitext(base_name)[0] + "_parsed.json")
                        
                        # Prepare serializable version of the result
                        serializable_result = {
                            'pdf_path': pdf_path,
                            'out_path': result.get('out_path'),
                            'column_divider': float(result.get('column_divider')),  # Convert numpy float to Python float
                            'section_texts': result.get('section_texts'),
                            'structured_content': result.get('structured_content')
                        }
                        
                        # Convert headings to serializable format (handle numpy values)
                        headings = []
                        for h in result.get('headings', []):
                            serializable_heading = {k: (float(v) if isinstance(v, np.number) else v) for k, v in h.items()}
                            headings.append(serializable_heading)
                        
                        serializable_result['headings'] = headings
                        
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"Processed {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)} - Found {len(result.get('section_texts', {}))} sections")
                    else:
                        logger.warning(f"Failed to process {os.path.basename(pdf_path)}")
                
                except Exception as e:
                    logger.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
    else:
        # Process sequentially if max_workers is 1
        logger.info("Processing files sequentially")
        for i, pdf_path in enumerate(pdf_files):
            try:
                result = process_resume(pdf_path, output_folder, visualize)
                if result:
                    results.append({
                        'pdf_path': pdf_path,
                        'out_path': result.get('out_path'),
                        'sections': list(result.get('section_texts', {}).keys()),
                        'processing_time': result.get('processing_time', 0)
                    })
                    
                    # Save the full result as JSON
                    base_name = os.path.basename(pdf_path)
                    json_path = os.path.join(output_folder, os.path.splitext(base_name)[0] + "_parsed.json")
                    
                    # Prepare serializable version of the result (handle numpy values)
                    serializable_result = {
                        'pdf_path': pdf_path,
                        'out_path': result.get('out_path'),
                        'column_divider': float(result.get('column_divider')),
                        'section_texts': result.get('section_texts'),
                        'structured_content': result.get('structured_content')
                    }
                    
                    # Convert headings to serializable format
                    headings = []
                    for h in result.get('headings', []):
                        serializable_heading = {k: (float(v) if isinstance(v, np.number) else v) for k, v in h.items()}
                        headings.append(serializable_heading)
                    
                    serializable_result['headings'] = headings
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Processed {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)} - Found {len(result.get('section_texts', {}))} sections")
                else:
                    logger.warning(f"Failed to process {os.path.basename(pdf_path)}")
            
            except Exception as e:
                logger.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
    
    # Generate summary report
    summary_path = os.path.join(output_folder, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(pdf_files),
            'successful_files': len(results),
            'average_processing_time': sum(r['processing_time'] for r in results) / len(results) if results else 0,
            'results': results
        }, f, indent=2)
    
    # Generate HTML report
    
    logger.info(f"Processing complete. Summary saved to {summary_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume PDF Parser')
    parser.add_argument('input_path', help='Path to a resume PDF file or a folder containing PDFs')
    parser.add_argument('--output', '-o', help='Output folder for results (default: input_path + "_output")')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization step (faster)')
    parser.add_argument('--threads', '-t', type=int, help=f'Maximum number of parallel threads (default: {MAX_WORKER_THREADS})')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    input_path = args.input_path
    output_folder = args.output
    visualize = not args.no_visualize
    max_workers = args.threads
    
    if os.path.isdir(input_path):
        # Process all PDFs in the folder
        results = process_resume_folder(input_path, output_folder, visualize, max_workers)

    else:
        # Process a single PDF
        result = process_resume(input_path, output_folder, visualize)
        if result:
            print(f"Resume successfully processed: {os.path.basename(input_path)}")
            print(f"Found sections: {', '.join(result['section_texts'].keys())}")
            if result['out_path']:
                print(f"Annotated PDF saved to: {result['out_path']}")
        else:
            print(f"Failed to process resume: {input_path}")

                         