import pdfplumber
import fitz  # PyMuPDF
import os
import re
from collections import defaultdict
import sys
import time
import logging
import statistics


# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resume_parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def normalize(text):
    """Normalize text by removing whitespace and converting to lowercase."""
    return re.sub(r'\s+', '', text).lower()

def is_heading_text(text):
    """Check if the text is a common resume section heading."""
    heading_keywords = [
        "education", "skills", "projects", "certifications", "contact",
        "languages", "objective", "experience", "workexperience", 
        "accomplishments", "awards", "careergoals", "careerobjective",
        "declaration", "decleration", "profile", "summary", "achievements",
        "interests", "activities", "leadership", "volunteer", "references",
        "research", "publications", "patents", "affiliations", "training",
        "courses", "hobbies", "personaldetails", "personal", "strength", 
        "portfolio", "qualification", "academic", "extracurricular"
    ]
    # Normalize the text for comparison
    clean = normalize(text)
    
    # Check if the normalized text matches any heading keyword
    return any(kw in clean for kw in heading_keywords)


def get_heading_type(text):
    """Determine which type of heading the text represents."""
    clean = normalize(text)
    
    # Map section types to their keywords - using 'in' for partial matching
    section_types = {
        "EDUCATION": ["education", "academic", "qualification", "degree"],
        "EXPERIENCE": ["experience", "workexperience", "career", "employment", "work history"],
        "SKILLS": ["skills", "technicalskills", "softskill", "competencies", "expertise"],
        "PROJECTS": ["projects", "portfolio", "assignments", "works"],
        "CONTACT": ["contact", "personal", "profile", "details"],
        "SUMMARY": ["summary", "profile", "objective", "careerobjective", "about", "introduction"],
        "CERTIFICATIONS": ["certifications", "certification", "credentials", "courses"],
        "AWARDS": ["awards", "honors", "achievements", "recognition"],
        "ACHIEVEMENTS": ["achievements", "accomplishments", "highlights"],
        "LANGUAGES": ["languages", "proficiency"],
        "DECLARATION": ["declaration", "decleration", "statement"],
        "INTERESTS": ["interests", "hobbies", "activities"],
        "VOLUNTEER": ["volunteer", "volunteering", "community"],
        "LEADERSHIP": ["leadership", "management"]
    }
    
    best_match = None
    best_score = 0
    
    for section_name, keywords in section_types.items():
        score = 0
        for kw in keywords:
            if kw in clean:
                # Exact match gets higher score
                if kw == clean:
                    score += 3
                # Partial match gets proportional score based on keyword length
                else:
                    score += len(kw) / len(clean) if len(clean) > 0 else 0
        
        if score > best_score:
            best_score = score
            best_match = section_name
    
    return best_match if best_score > 0.3 else None

def extract_text_with_style(pdf_path):
    """Extract text with style information using PyMuPDF (fitz)."""
    doc = fitz.open(pdf_path)
    all_elements = []
    
    # First, calculate average font size across document for better headings detection
    font_sizes = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            if span.get("size", 0) > 0:  # Avoid zero sizes
                                font_sizes.append(span.get("size", 0))
    
    # Calculate average font size
    # avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 11  # Default if no valid sizes
    # heading_font_threshold = avg_font_size * 1.1  # 10% larger than average
    
    # Now extract elements with style information
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                                
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
                            })
    
    doc.close()
    return all_elements

def analyze_font_metrics(elements):
    """Analyze font metrics to determine relative size thresholds."""
    # Extract font sizes
    font_sizes = [elem["font_size"] for elem in elements if elem["font_size"] > 0]
    
    # If no valid font sizes found, return default values
    if not font_sizes:
        return {
            "median_size": 11,
            "heading_size_threshold": 12,
            "small_text_threshold": 8
        }
    
    # Calculate font size statistics
    try:
        median_size = statistics.median(font_sizes)
        # Get the most common font sizes (may have multiple modes)
        size_counts = {}
        for size in font_sizes:
            size_rounded = round(size, 1)  # Round to nearest decimal to group similar sizes
            size_counts[size_rounded] = size_counts.get(size_rounded, 0) + 1
        
        # Get the most frequent font size
        mode_size = max(size_counts.items(), key=lambda x: x[1])[0]
    except statistics.StatisticsError:
        # Handle cases where statistics can't be calculated
        median_size = sum(font_sizes) / len(font_sizes)
        mode_size = median_size
    
    # Determine more adaptive thresholds based on document analysis
    # Look for natural breaks in font sizes by finding the largest gap
    sorted_sizes = sorted(list(set([round(s, 1) for s in font_sizes])))
    
    # If there are multiple font sizes, look for natural breaks
    if len(sorted_sizes) > 1:
        gaps = [(sorted_sizes[i+1] - sorted_sizes[i], i) for i in range(len(sorted_sizes)-1)]
        if gaps:
            max_gap = max(gaps)
            if max_gap[0] > 0.5:  # If there's a significant gap
                # Use the size just above the largest gap as heading threshold
                heading_size_threshold = sorted_sizes[max_gap[1] + 1]
            else:
                # Use a relative threshold if no clear gap exists
                heading_size_threshold = max(mode_size * 1.15, median_size * 1.1)
        else:
            heading_size_threshold = max(mode_size * 1.15, median_size * 1.1)
    else:
        # If only one font size, use a default relative increase
        heading_size_threshold = mode_size * 1.2
    
    small_text_threshold = min(mode_size * 0.9, median_size * 0.85)
    
    return {
        "median_size": median_size,
        "mode_size": mode_size,
        "heading_size_threshold": heading_size_threshold,
        "small_text_threshold": small_text_threshold,
    }

def detect_layout_orientation(elements, pdf_path):
    """Detect if the resume has vertical sections or horizontal layout."""
    with pdfplumber.open(pdf_path) as pdf:
        page_width = pdf.pages[0].width
    
    # Group elements by Y position to detect lines
    y_positions = defaultdict(list)
    for elem in elements:
        # Round y-position to nearest 5 points to group elements on roughly the same line
        y_bin = round(elem["y0"] / 5) * 5
        y_positions[y_bin].append(elem)
    
    # Count lines with multiple elements horizontally
    lines_with_multiple_elements = sum(1 for elems in y_positions.values() if len(elems) > 1)
    total_lines = len(y_positions)
    
    # Calculate percentage of lines with multiple elements
    if total_lines > 0:
        horizontal_ratio = lines_with_multiple_elements / total_lines
    else:
        horizontal_ratio = 0
    
    # Check distribution of text across page width
    x_positions = defaultdict(int)
    for elem in elements:
        # Group x-positions into bins (10% of page width)
        bin_width = page_width / 10
        bin_index = int(elem["x0"] / bin_width)
        x_positions[bin_index] += 1
    
    # Count filled bins (ones with significant number of elements)
    filled_bins = sum(1 for count in x_positions.values() if count > len(elements) * 0.05)
    
    # If text is distributed across many x-positions and many lines have multiple elements,
    # it's likely a horizontal layout. Otherwise, it's more likely vertical.
    is_vertical_layout = filled_bins <= 3 or horizontal_ratio < 0.3
    
    return is_vertical_layout

def analyze_layout(elements, pdf_path):
    """
    Analyze the resume layout to detect columns.
    Returns the x-coordinate that separates left and right columns and whether multi-column layout exists.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page_width = pdf.pages[0].width
    
    # Get all x-coordinates
    x_coords = [elem["x0"] for elem in elements]
    
    # Create a histogram of x-coordinates to find columns
    x_hist = defaultdict(int)
    for x in x_coords:
        # Round to nearest 10 to create bins
        bin_x = round(x / 10) * 10
        x_hist[bin_x] += 1
    
    # Sort bins by frequency
    sorted_bins = sorted(x_hist.items(), key=lambda x: x[1], reverse=True)
    
    # Look for significant gaps in the middle of the page
    left_region = sum(x_hist.get(x, 0) for x in range(0, int(page_width * 0.4), 10))
    middle_region = sum(x_hist.get(x, 0) for x in range(int(page_width * 0.4), int(page_width * 0.6), 10))
    right_region = sum(x_hist.get(x, 0) for x in range(int(page_width * 0.6), int(page_width), 10))
    
    total_elements = left_region + middle_region + right_region
    if total_elements == 0:
        return page_width * 0.6, False
    
    left_ratio = left_region / total_elements
    middle_ratio = middle_region / total_elements
    right_ratio = right_region / total_elements
    
    # If we have a clear two-column layout, we'll have significant content on both sides
    # and relatively little in the middle
    is_multi_column = (left_ratio > 0.25 and right_ratio > 0.25 and middle_ratio < 0.2)
    
    # If we have a clear two-column layout, we'll have distinct clusters of x-coordinates
    if is_multi_column and len(sorted_bins) >= 2:
        # Get the most common x-positions
        left_col_x = sorted_bins[0][0]
        
        # Find the next most common x-position that's significantly different
        right_col_x = None
        for bin_x, count in sorted_bins:
            if abs(bin_x - left_col_x) > page_width * 0.2:  # At least 20% of page width apart
                right_col_x = bin_x
                break
        
        if right_col_x:
            # Return the midpoint between the columns as a divider
            return (left_col_x + right_col_x) / 2, is_multi_column
    
    # If we detect a multi-column layout but can't find clear column positions,
    # use the middle of the page as the divider
    if is_multi_column:
        return page_width * 0.5, True
    
    # Default to 60% of page width if we can't clearly detect columns
    return page_width * 0.6, False

def is_at_line_start(element, all_elements):
    """Check if an element is at the start of a line (no elements to its left)."""
    for other in all_elements:
        if (other["page"] == element["page"] and 
            abs(other["y0"] - element["y0"]) < 5 and  # Same line (within 5 points)
            other["x0"] < element["x0"]):  # To the left
            return False
    return True

def has_space_above(element, all_elements):
    """Check if there's significant whitespace above this element."""
    if element["page"] == 0 and element["y0"] < 50:
        return True  # Near top of first page
        
    # Find the closest element above this one
    elements_above = [e for e in all_elements if 
                     e["page"] == element["page"] and 
                     e["y1"] < element["y0"]]
    
    if not elements_above:
        return True  # Nothing above
    
    closest_above = max(elements_above, key=lambda e: e["y1"])
    gap = element["y0"] - closest_above["y1"]
    
    # Estimate typical line height
    line_height = element["y1"] - element["y0"]
    
    # Is gap significantly larger than line height?
    return gap > 1.5 * line_height

def is_indented_less_than_following(element, all_elements):
    """Check if this element is less indented than the following text (section content)."""
    elements_below = [e for e in all_elements if 
                     e["page"] == element["page"] and 
                     e["y0"] > element["y1"] and
                     e["y0"] < element["y1"] + 50]  # Within reasonable distance
    
    if not elements_below:
        return False
    
    # Get the most common x0 of elements below
    x0_values = [e["x0"] for e in elements_below]
    if not x0_values:
        return False
        
    # Check if most following elements are indented more than this one
    return sum(1 for x in x0_values if x > element["x0"]) > len(x0_values) / 2

def get_following_text(element, all_elements, num_elements=5):
    """Get text from elements following the given element."""
    elements_below = [e for e in all_elements if 
                     (e["page"] > element["page"] or 
                     (e["page"] == element["page"] and e["y0"] > element["y1"]))]
    
    # Sort by page and position
    elements_below.sort(key=lambda e: (e["page"], e["y0"]))
    
    # Get the next few elements
    next_elements = elements_below[:num_elements]
    
    # Combine their text
    return " ".join([e["text"] for e in next_elements])

def section_matches_content(section_type, content_text):
    """Check if content matches the expected pattern for a section type."""
    content_lower = content_text.lower()
    
    patterns = {
        "EDUCATION": r'(university|college|school|degree|gpa|cgpa|bachelor|master|phd|20\d\d|19\d\d)',
        "EXPERIENCE": r'(company|position|job|role|responsibilities|manager|intern|20\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
        "SKILLS": r'(proficient|familiar|expert|knowledge|experience|advanced|intermediate|beginner|tools|technologies|languages)',
        "PROJECTS": r'(developed|created|built|implemented|designed|project|application|system|website|software)',
        "CERTIFICATIONS": r'(certified|certificate|certification|course|training|issued|completed)',
        "LANGUAGES": r'(native|fluent|proficient|beginner|intermediate|advanced|speak|written|verbal)'
    }
    
    if section_type in patterns:
        pattern = patterns[section_type]
        matches = re.findall(pattern, content_lower)
        return len(matches) > 1
    
    return False

def calculate_heading_confidence(element, all_elements, font_metrics):
    """Calculate a more accurate confidence score for potential headings."""
    confidence = 0
    text = element["text"].strip()
    
    # Styling cues
    if element["is_bold"]:
        confidence += 2
    if element["is_capital"]:
        confidence += 2
    if element["is_spaced"]:
        confidence += 2
    
    # Font size comparison using relative metrics
    if font_metrics["heading_size_threshold"] > 0:
        size_ratio = element["font_size"] / font_metrics["heading_size_threshold"]
        if size_ratio > 1:
            confidence += 2 * min(size_ratio, 1.5)  # Cap at 3 points for size
    
    # Position cues
    if is_at_line_start(element, all_elements):
        confidence += 1.5
    if has_space_above(element, all_elements):
        confidence += 1.5
    if is_indented_less_than_following(element, all_elements):
        confidence += 1
        
    # Content cues
    if len(text) < 25:  # Section headings are usually short
        confidence += 1

    
    # Check if the text matches common section headings
    heading_type = get_heading_type(text)
    if heading_type != "OTHER":
        confidence += 1.5
    
    # Contextual cues - check if following content matches section expectation
    following_text = get_following_text(element, all_elements, 5)
    if section_matches_content(heading_type, following_text):
        confidence += 2
    
    return confidence


def identify_section_headings(elements, column_divider, font_metrics, is_multi_column):
    """
    Identify section headings based on text styling and patterns.
    Takes into account the multi-column layout and relative font sizes.
    """
    headings = []
    min_confidence_threshold = 3.0  # Minimum confidence to consider as heading
    
    # First pass - identify potential headings
    for i, elem in enumerate(elements):
        text = elem["text"].strip()
        
        # Skip very short text unless it's a known heading
        if len(text) < 2 and not is_heading_text(text):
            continue
        
        # Calculate confidence score
        heading_confidence = calculate_heading_confidence(elem, elements, font_metrics)
        heading_type = get_heading_type(text)
        
        # Only consider elements with sufficient confidence
        if heading_confidence >= min_confidence_threshold:
            # Determine which column this heading belongs to in multi-column layout
            if is_multi_column:
                column = "left" if elem["x0"] < column_divider else "right"
            else:
                column = "single"
            
            # Ensure font_size is included in the heading information
            headings.append({
                "page": elem["page"],
                "text": text,
                "x0": elem["x0"],
                "y0": elem["y0"],
                "x1": elem["x1"],
                "y1": elem["y1"],
                "type": heading_type,
                "column": column,
                "confidence": heading_confidence,
                "font_size": elem["font_size"]  # Include font_size
            })
    
    # Sort headings by confidence (higher score first)
    headings.sort(key=lambda h: h["confidence"], reverse=True)
    
    # Filter out overlapping or duplicate headings
    filtered_headings = []
    for heading in headings:
        # Check if we already have a similar heading
        duplicate = False
        for existing in filtered_headings:
            # Same type in same area
            if (existing["type"] == heading["type"] and 
                existing["page"] == heading["page"] and
                existing["column"] == heading["column"] and
                abs(existing["y0"] - heading["y0"]) < 50):  # Within 50 points vertically
                duplicate = True
                break
        
        if not duplicate:
            filtered_headings.append(heading)
    
    # Final pass - ensure canonical section names have highest confidence
    # If we find a section like "education" with low confidence but strong contextual
    # evidence, boost its confidence
    for heading in filtered_headings:
        if heading["confidence"] < 5 and heading["type"] != "OTHER":
            following_text = get_following_text(heading, elements, 10)
            if section_matches_content(heading["type"], following_text):
                heading["confidence"] += 1.5
                logger.info(f"Boosted confidence for {heading['text']} based on content match")
    
    return filtered_headings


def extract_sections(headings, elements, column_divider, is_multi_column, is_vertical_layout):
    """
    Extract sections from the PDF, handling multi-column or single-column layout,
    and respecting vertical or horizontal section arrangement.
    """
    sections = {}
    
    if not headings:
        return sections
    
    # Group elements by page for efficiency
    elements_by_page = defaultdict(list)
    for elem in elements:
        elements_by_page[elem["page"]].append(elem)
    
    if is_vertical_layout or not is_multi_column:
        # Vertical layout processing - all headings sorted top to bottom
        sorted_headings = sorted(headings, key=lambda h: (h["page"], h["y0"]))
        
        # Process headings sequentially
        for i, heading in enumerate(sorted_headings):
            heading_page = heading["page"]
            heading_y = heading["y1"]  # Bottom of the heading
            
            # Find the next heading (for vertical layout, consider any heading)
            end_y = float('inf')
            end_page = float('inf')
            
            if i < len(sorted_headings) - 1:
                next_heading = sorted_headings[i + 1]
                end_y = next_heading["y0"]
                end_page = next_heading["page"]
            
            # Collect all elements in this section
            section_elements = []
            
            # Process elements page by page
            for page_num in range(heading_page, min(end_page + 1, max(elements_by_page.keys()) + 1)):
                if page_num not in elements_by_page:
                    continue
                
                for elem in elements_by_page[page_num]:
                    # Skip if this is a heading element
                    if any(elem["x0"] == h["x0"] and elem["y0"] == h["y0"] for h in headings):
                        continue
                    
                    # If on heading page, only include elements after the heading
                    if page_num == heading_page and elem["y0"] < heading_y:
                        continue
                    
                    # If on end page, only include elements before the next heading
                    if page_num == end_page and elem["y0"] >= end_y:
                        continue
                    
                    # For vertical layouts, check horizontal alignment with more tolerance
                    if is_vertical_layout:
                        # Calculate horizontal alignment
                        heading_left = heading["x0"]
                        heading_right = heading["x1"]
                        elem_left = elem["x0"]
                        
                        # Allow for wider horizontal tolerance
                        horizontal_alignment = (
                            abs(elem_left - heading_left) < 150 or  # Within 150 points of heading start
                            (elem_left >= heading_left - 20 and elem_left <= heading_right + 100)  # Within reasonable bounds
                        )
                        
                        if not horizontal_alignment:
                            continue
                    
                    section_elements.append(elem)
            
            # Only add section if it has content
            if section_elements:
                section_key = f"{heading['type']}"
                if is_multi_column:
                    section_key += f"_{heading['column']}"
                
                sections[section_key] = {
                    "heading": heading,
                    "elements": section_elements,
                    "column": heading["column"]
                }
    else:
        # Multi-column layout processing
        # Separate headings by column
        left_headings = [h for h in headings if h["column"] == "left"]
        right_headings = [h for h in headings if h["column"] == "right"]
        
        # Sort headings in each column by page and vertical position
        left_headings = sorted(left_headings, key=lambda h: (h["page"], h["y0"]))
        right_headings = sorted(right_headings, key=lambda h: (h["page"], h["y0"]))
        
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
                
                # Define the column boundaries with some overlap tolerance
                min_x = 0 if column == "left" else column_divider - 30  # Allow 30pt overlap
                max_x = column_divider + 30 if column == "left" else float('inf')  # Allow 30pt overlap
                
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
                
                # Only add section if it has content
                if section_elements:
                    section_key = f"{heading['type']}_{column}"
                    sections[section_key] = {
                        "heading": heading,
                        "elements": section_elements,
                        "column": column
                    }
    
    return sections

def extract_section_text(sections):
    """Convert the section elements into readable text."""
    section_texts = {}
    
    for section_key, section_data in sections.items():
        elements = section_data["elements"]
        section_type = section_data["heading"]["type"]
        column = section_data["column"]
        
        # Sort elements by page, then by y position, then by x position
        sorted_elements = sorted(elements, key=lambda e: (e["page"], e["y0"], e["x0"]))
        
        # Group elements by line (similar y positions)
        lines = defaultdict(list)
        for elem in sorted_elements:
            # Create a key based on page and rounded y-position to group elements on the same line
            line_key = (elem["page"], round(elem["y0"]))
            lines[line_key].append(elem)
        
        # Build the text line by line
        text_lines = []
        for line_key in sorted(lines.keys()):
            line_elements = sorted(lines[line_key], key=lambda e: e["x0"])
            line_text = ' '.join(elem["text"] for elem in line_elements)
            text_lines.append(line_text)
        
        # Join all lines with newlines
        # Use a simple section key for better readability
        simple_key = section_type
        if simple_key in section_texts:
            # If we already have this section type, append new content
            section_texts[simple_key] += "\n\n--- " + column.upper() + " COLUMN ---\n" + '\n'.join(text_lines)
        else:
            section_texts[simple_key] = "--- " + column.upper() + " COLUMN ---\n" + '\n'.join(text_lines)
    
    return section_texts

def visualize_sections(pdf_path, sections, headings, column_divider, is_multi_column, is_vertical_layout):
    """Create a visualized PDF with highlighted sections, showing the column structure."""
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
        "DECLARATION": (0.4, 0.4, 0.4), # Gray
        "INTERESTS": (0.2, 0.6, 0.8), # Sky Blue
        "VOLUNTEER": (0.8, 0.2, 0.6), # Pink
        "REFERENCES": (0.3, 0.7, 0.3), # Light Green
        "LEADERSHIP": (0.3, 0.3, 0.7), # Lavender
        "OTHER": (0.3, 0.3, 0.3)     # Dark Gray
    }
    
    doc = fitz.open(pdf_path)
    output = fitz.open()
    
    # Create a new PDF with highlighted sections
    for page_num in range(len(doc)):
        # Create a new page
        orig_page = doc[page_num]
        new_page = output.new_page(width=orig_page.rect.width, height=orig_page.rect.height)
        new_page.show_pdf_page(orig_page.rect, doc, page_num)
        
        # Draw layout information on first page
        if page_num == 0:
            layout_info = []
            if is_multi_column:
                layout_info.append("Multi-column Layout")
            else:
                layout_info.append("Single-column Layout")
                
            if is_vertical_layout:
                layout_info.append("Vertical Section Arrangement")
            else:
                layout_info.append("Horizontal Section Arrangement")
                
            info_text = " | ".join(layout_info)
            new_page.insert_text(fitz.Point(50, 30), info_text, fontsize=12, color=(0,0,0))
        
        # Draw column divider line if it's a multi-column layout
        if is_multi_column:
            new_page.draw_line((column_divider, 0), (column_divider, orig_page.rect.height), 
                               color=(0.5, 0.5, 0.5), width=0.5, dashes="[2 2]")
            
            # Add labels for columns
            new_page.insert_text(fitz.Point(column_divider/2, 50), "LEFT COLUMN", 
                                fontsize=8, color=(0.5, 0.5, 0.5))
            new_page.insert_text(fitz.Point(column_divider + (orig_page.rect.width - column_divider)/2, 50), 
                                "RIGHT COLUMN", fontsize=8, color=(0.5, 0.5, 0.5))
        
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
            x0 = min(e["x0"] for e in page_elements) - 5  # Add margin
            y0 = min(e["y0"] for e in page_elements) - 5  # Add margin
            x1 = max(e["x1"] for e in page_elements) + 5  # Add margin
            y1 = max(e["y1"] for e in page_elements) + 5  # Add margin
            
            # Find the nearest heading below to constrain the box
            current_heading = section_data["heading"]
            
            if current_heading["page"] == page_num:
                # Find headings on this page that are below the current heading
                headings_below = [
                    h for h in headings if 
                    h["page"] == page_num and 
                    h["y0"] > current_heading["y1"]
                ]
                
                # Sort by vertical position
                headings_below.sort(key=lambda h: h["y0"])
                
                # If there's a heading below, limit box height
                if headings_below and y1 >= headings_below[0]["y0"]:
                    y1 = headings_below[0]["y0"] - 10  # Leave some space
            
            # Prevent excessively wide boxes (potential "blue box" issue)
            if x1 - x0 > 1000:  # Unreasonably wide box
                # Recalculate using the median x position to avoid outliers
                x_coords = [e["x0"] for e in page_elements]
                median_x = sorted(x_coords)[len(x_coords) // 2]
                
                # Reset bounds using elements closer to the median
                close_elements = [e for e in page_elements if abs(e["x0"] - median_x) < 250]
                if close_elements:
                    x0 = min(e["x0"] for e in close_elements) - 5
                    x1 = max(e["x1"] for e in close_elements) + 5
            
            # Get the color for this section
            color = colors.get(section_type, (0.3, 0.3, 0.3))
            
            # Draw a rectangle around the section content
            rect = fitz.Rect(x0, y0, x1, y1)
            new_page.draw_rect(rect, color=color, width=1.5, dashes="[2 2]")
            
            # Add a small label at the bottom of the section for clarity
            label_point = fitz.Point(x0, y1 + 5)
            label_text = f"{section_type} ({section_data['column']} column)"
            new_page.insert_text(label_point, label_text, fontsize=7, color=color)
    
    # Save the output PDF
    out_path = os.path.splitext(pdf_path)[0] + "_sections_highlighted.pdf"
    output.save(out_path)
    logger.info(f"Annotated PDF saved to: {out_path}")
    
    # Clean up
    output.close()
    doc.close()
    
    return out_path

def parse_resume(pdf_path):
    """Parse a resume PDF file, extract sections from multi-column layout, and visualize them."""
    logger.info(f"Processing resume: {pdf_path}")
    
    try:
        # Extract text elements with style information
        elements = extract_text_with_style(pdf_path)
        logger.info(f"Extracted {len(elements)} text elements")
        
        if not elements:
            logger.error(f"No text elements found in {pdf_path}")
            return None, {}, False, False
        font_metrics = analyze_font_metrics(elements)
        logger.info(f"Font metrics analysis:")
        logger.info(f"  - Median size: {font_metrics['median_size']:.2f}")
        logger.info(f"  - Heading size threshold: {font_metrics['heading_size_threshold']:.2f}")
        
        # Detect layout orientation (vertical vs horizontal)
        is_vertical_layout = detect_layout_orientation(elements, pdf_path)
        logger.info(f"Layout orientation: {'Vertical' if is_vertical_layout else 'Horizontal'}")
        
        # Analyze the layout to detect columns
        column_divider, is_multi_column = analyze_layout(elements, pdf_path)
        logger.info(f"Layout analysis: {'Multi-column' if is_multi_column else 'Single-column'}")
        logger.info(f"Column divider at x-coordinate: {column_divider}")
        
        # Identify section headings, taking into account the column structure
        headings = identify_section_headings(elements, column_divider,font_metrics, is_multi_column)
        logger.info(f"Identified {len(headings)} section headings:")
        
        # Count headings in each column
        if is_multi_column:
            left_headings = [h for h in headings if h["column"] == "left"]
            right_headings = [h for h in headings if h["column"] == "right"]
            logger.info(f"  - Left column: {len(left_headings)} headings")
            logger.info(f"  - Right column: {len(right_headings)} headings")
        
        for heading in headings:
            logger.info(f"  - {heading['text']} ({heading['type']}, {heading['column']} column)")
        
        # Extract sections, handling the layout structure
        sections = extract_sections(headings, elements, column_divider, is_multi_column, is_vertical_layout)
        logger.info(f"Extracted {len(sections)} sections")
        
        # Refine sections to avoid the "blue box" issue
        refined_sections = refine_section_boundaries(sections, headings)
        
        # Extract text from each section
        section_texts = extract_section_text(refined_sections)
        
        # Visualize the sections and layout structure
        out_path = visualize_sections(pdf_path, refined_sections, headings, column_divider, 
                                      is_multi_column, is_vertical_layout)
        
        # Print the extracted text
        logger.info("\nExtracted Section Content:")
        logger.info("=========================")
        for section_type, text in section_texts.items():
            logger.info(f"\n{section_type}:")
            logger.info("-" * len(section_type))
            if len(text) > 500:
                logger.info(text[:500] + "...")
            else:
                logger.info(text)
        
        return out_path, section_texts, is_multi_column, is_vertical_layout
        
    except Exception as e:
        logger.error(f"Error parsing resume {pdf_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}, False, False

def refine_section_boundaries(sections, headings):
    """Refine section boundaries to avoid issues like the 'blue box' problem."""
    refined_sections = {}
    
    # Sort headings by page and vertical position
    sorted_headings = sorted(headings, key=lambda h: (h["page"], h["y0"]))
    
    for section_key, section_data in sections.items():
        current_heading = section_data["heading"]
        column = section_data["column"]
        elements = section_data["elements"]
        
        # Find the next heading in the same column
        next_heading = None
        for h in sorted_headings:
            if (h["page"] > current_heading["page"] or 
                (h["page"] == current_heading["page"] and h["y0"] > current_heading["y1"])):
                if column == "single" or h["column"] == column:
                    next_heading = h
                    break
        
        # Filter elements to respect next heading boundary
        if next_heading:
            filtered_elements = [
                e for e in elements if
                e["page"] < next_heading["page"] or 
                (e["page"] == next_heading["page"] and e["y0"] < next_heading["y0"])
            ]
        else:
            filtered_elements = elements
        
        # Fix horizontal spread to prevent "blue box" issue
        if filtered_elements:
            # Group elements by x-position
            x_positions = defaultdict(int)
            for elem in filtered_elements:
                bin_x = round(elem["x0"] / 10) * 10  # Round to nearest 10
                x_positions[bin_x] += 1
            
            # Find the most common x-positions
            sorted_x_bins = sorted(x_positions.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_x_bins) > 1:
                main_x = sorted_x_bins[0][0]
                
                # Check if there are outlier elements that are very far from main x position
                far_outliers = [
                    e for e in filtered_elements if 
                    abs(round(e["x0"] / 10) * 10 - main_x) > 200  # More than 200 points away
                ]
                
                # If there are very few outliers compared to total elements, remove them
                if far_outliers and len(far_outliers) < len(filtered_elements) * 0.1:  # Less than 10%
                    filtered_elements = [e for e in filtered_elements if e not in far_outliers]
        
        # Store the refined section
        refined_sections[section_key] = {
            "heading": current_heading,
            "elements": filtered_elements,
            "column": column
        }
    
    return refined_sections

def process_resume_folder(folder_path, output_folder=None):
    """
    Process all PDF files in a folder to identify and extract resume sections.
    
    Args:
        folder_path (str): Path to the folder containing resume PDFs
        output_folder (str, optional): Path to save output files. If None, uses folder_path + "_output"
    
    Returns:
        list: List of processed file paths and their extracted sections
    """
    if output_folder is None:
        output_folder = folder_path + "_output"
    
    if not os.path.isdir(folder_path):
        logger.error(f"Error: {folder_path} is not a valid directory")
        return []
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PDF files in the folder
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.error(f"No PDF files found in {folder_path}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
    
    # Process each PDF file
    results = []
    for i, pdf_path in enumerate(pdf_files):
        logger.info(f"\nProcessing file {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
        
        start_time = time.time()
        out_path, sections, is_multi_column, is_vertical = parse_resume(pdf_path)
        elapsed_time = time.time() - start_time
        
        if out_path:
            # Save each section to a separate text file if needed
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Create a subfolder for this resume
            resume_folder = os.path.join(output_folder, base_name)
            os.makedirs(resume_folder, exist_ok=True)
            
            # Save summary info
            summary_file = os.path.join(resume_folder, "summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Resume: {pdf_path}\n")
                f.write(f"Processing time: {elapsed_time:.2f} seconds\n")
                f.write(f"Layout: {'Multi-column' if is_multi_column else 'Single-column'}\n")
                f.write(f"Section Arrangement: {'Vertical' if is_vertical else 'Horizontal'}\n")
                f.write(f"Sections found: {', '.join(sections.keys())}\n")
            
            # Copy the highlighted PDF
            import shutil
            highlighted_pdf = os.path.join(resume_folder, f"{base_name}_highlighted.pdf")
            shutil.copy2(out_path, highlighted_pdf)
            
            results.append({
                'pdf_path': pdf_path,
                'out_path': out_path,
                'sections': list(sections.keys()),
                'processing_time': elapsed_time,
                'is_multi_column': is_multi_column,
                'is_vertical': is_vertical
            })
    
    # Create overall summary
    summary_path = os.path.join(output_folder, "processing_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Resume Processing Summary\n")
        f.write(f"=======================\n")
        f.write(f"Total PDFs processed: {len(results)}\n")
        if results:
            f.write(f"Average processing time: {sum(r['processing_time'] for r in results) / len(results):.2f} seconds\n\n")
            
            for r in results:
                f.write(f"File: {os.path.basename(r['pdf_path'])}\n")
                f.write(f"  Layout: {'Multi-column' if r.get('is_multi_column', False) else 'Single-column'}, ")
                f.write(f"{'Vertical' if r.get('is_vertical', False) else 'Horizontal'} arrangement\n")
                f.write(f"  Sections: {', '.join(r['sections'])}\n")
                f.write(f"  Processing time: {r['processing_time']:.2f} seconds\n\n")
    
    logger.info(f"\nProcessing complete. Results saved to {output_folder}")
    logger.info(f"Summary file created at {summary_path}")
    
    return results

def main():
    """Main function to run the resume parser on a single PDF or directory."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume Parser with Section Detection')
    parser.add_argument('input_path', help='Path to a resume PDF file or a directory of PDFs')
    parser.add_argument('-o', '--output_dir', help='Output directory (optional)')
    
    args = parser.parse_args()
    input_path = args.input_path
    output_dir = args.output_dir
    
    if os.path.isdir(input_path):
        # Process all PDFs in the folder
        process_resume_folder(input_path, output_dir)
    else:
        # Process a single PDF
        out_path, sections, is_multi_column, is_vertical = parse_resume(input_path)
        if out_path:
            logger.info(f"Successfully processed: {input_path}")
            logger.info(f"Output: {out_path}")
            logger.info(f"Sections detected: {len(sections)}")
            logger.info(f"Layout: {'Multi-column' if is_multi_column else 'Single-column'}, "
                       f"{'Vertical' if is_vertical else 'Horizontal'} arrangement")
        else:
            logger.error(f"Failed to process: {input_path}")

if __name__ == "__main__":
    main()