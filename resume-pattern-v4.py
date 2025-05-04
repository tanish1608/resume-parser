import pdfplumber
import fitz  # PyMuPDF
import os
import re
import logging
import time
import statistics
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resume_parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Minimum confidence threshold for sections
MIN_CONFIDENCE_THRESHOLD = 8.0

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

def contains_sentence_ending_punctuation(text):
    """Check if text contains sentence-ending punctuation."""
    return bool(re.search(r'[.!?]$', text.strip()))

def get_heading_type(text):
    """Determine which type of heading the text represents with fuzzy matching."""
    clean = normalize(text)
    
    # Expanded section types with more variations and synonyms
    section_types = {
        "EDUCATION": ["education", "academic", "qualification", "degree", "schooling", "university", "college", "school"],
        "EXPERIENCE": ["experience", "workexperience", "employment", "career", "work", "professional", "job", "internship"],
        "SKILLS": ["skills", "technicalskills", "softskill", "competencies", "expertise", "proficiencies", "abilities"],
        "PROJECTS": ["projects", "portfolio", "assignment", "casestudy", "works"],
        "CONTACT": ["contact", "personalinfo", "personaldetails", "personal", "profile"],
        "SUMMARY": ["summary", "profile", "objective", "careerobjective", "about", "introduction", "professional summary"],
        "CERTIFICATIONS": ["certifications", "certification", "credentials", "courses", "training", "qualification"],
        "AWARDS": ["awards", "honors", "achievements", "recognition", "accomplishments"],
        "LANGUAGES": ["languages", "languageskills", "languageproficiency", "spoken"],
        "INTERESTS": ["interests", "hobbies", "activities", "extracurricular", "passions"],
        "LEADERSHIP": ["leadership", "management", "initiative", "positions", "roles"],
        "STRENGTHS": ["strength", "coreskills", "keystrengths"],
        "VOLUNTEER": ["volunteer", "volunteering", "community", "social work"],
        "ACHIEVEMENTS": ["achievements", "accomplishments", "highlights", "keyachievements"],
        "DECLARATION": ["declaration", "decleration", "statement"],
    }
    
    # Check for best match among section types
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
    
    return best_match if best_score > 0.3 else "OTHER"

def extract_text_with_style(pdf_path):
    """Extract text with style information using PyMuPDF (fitz)."""
    doc = fitz.open(pdf_path)
    all_elements = []
    
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
                            
                            # Don't determine if it's a heading here - just store the elements
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
                                "is_spaced": is_spaced
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

def analyze_layout(elements, pdf_path):
    """
    Enhanced layout analysis to detect complex column structures.
    Returns column dividers and whether it's a multi-column layout.
    """
    # with pdfplumber.open(pdf_path) as pdf:
    #     page_width = pdf.pages[0].width
    
    # # Create a visual "heat map" of text positions
    # x_positions = {}
    # for elem in elements:
    #     x_bin = round(elem["x0"] / 5) * 5  # 5pt bins
    #     x_positions[x_bin] = x_positions.get(x_bin, 0) + 1
    
    # # Check if we have a multi-column layout
    # left_region = sum(x_positions.get(x, 0) for x in range(0, int(page_width * 0.4), 5))
    # middle_region = sum(x_positions.get(x, 0) for x in range(int(page_width * 0.4), int(page_width * 0.6), 5))
    # right_region = sum(x_positions.get(x, 0) for x in range(int(page_width * 0.6), int(page_width), 5))
    
    # total_elements = left_region + middle_region + right_region
    # if total_elements == 0:
    #     return page_width * 0.6, False  # Default, no multi-column
    
    # left_ratio = left_region / total_elements
    # middle_ratio = middle_region / total_elements
    # right_ratio = right_region / total_elements
    
    # # Generate density profile for visualization
    # x_vals = sorted(x_positions.keys())
    # density = [x_positions[x] for x in x_vals]
    
    # # Check for significant gaps in x-distribution
    # is_multi_column = (left_ratio > 0.25 and right_ratio > 0.25 and middle_ratio < 0.15)
    
    # # If we have too few points, fall back to simple approach
    # if len(x_vals) < 10:
    #     if is_multi_column:
    #         return page_width * 0.5, True
    #     else:
    #         return page_width * 0.6, False
    
    # # Use peak detection to find columns
    # try:
    #     # Convert density to numpy array for find_peaks
    #     density_array = np.array(density)
    #     peaks, _ = find_peaks(density_array, height=max(density_array)*0.3, distance=page_width*0.1/5)
    #     peak_positions = [x_vals[p] for p in peaks if p < len(x_vals)]
        
    #     if len(peak_positions) <= 1:
    #         # Only one column detected
    #         return page_width * 0.6, False
        
    #     # Find valleys between peaks to identify column dividers
    #     valleys = []
    #     for i in range(len(peaks)-1):
    #         start_idx = peaks[i]
    #         end_idx = peaks[i+1]
    #         if start_idx >= len(density) or end_idx >= len(density):
    #             continue
                
    #         between_segment = density[start_idx:end_idx+1]
    #         if between_segment:
    #             min_val = min(between_segment)
    #             if min_val < max(density) * 0.3:  # Significant valley
    #                 min_idx = between_segment.index(min_val) + start_idx
    #                 if min_idx < len(x_vals):
    #                     valley_position = x_vals[min_idx]
    #                     valleys.append(valley_position)
        
    #     # Use the most significant valley as column divider
    #     if valleys and is_multi_column:
    #         # Find the valley closest to the middle
    #         middle_x = page_width / 2
    #         divider = min(valleys, key=lambda v: abs(v - middle_x))
    #         return divider, True
    # except Exception as e:
    #     logger.error(f"Error in peak detection: {str(e)}")
    #     # Fall back to simple method
    
    # # Improved heuristic for column detection - analyze text distribution
    # if is_multi_column:
    #     # Try to find a clear gap in the middle of the page
    #     middle_start = int(page_width * 0.4)
    #     middle_end = int(page_width * 0.6)
        
    #     # Count elements in each position in the middle area
    #     middle_counts = [x_positions.get(x, 0) for x in range(middle_start, middle_end, 5)]
        
    #     # If there's a significant empty area in the middle, use it to determine column divider
    #     if middle_counts and max(middle_counts) < max(density) * 0.3:
    #         # Find the position with minimum elements in the middle
    #         min_pos = middle_start + middle_counts.index(min(middle_counts)) * 5
    #         return min_pos, True
    #     else:
    #         # Default to middle of page
    #         return page_width * 0.5, True
    # else:
    #     return page_width * 0.6, False
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
    
    # If we have a clear two-column layout, we'll have distinct clusters of x-coordinates
    if len(sorted_bins) >= 2:
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
            return (left_col_x + right_col_x) / 2
    
    # Default to 60% of page width if we can't clearly detect columns
    return page_width * 0.6

def detect_layout_orientation(elements, pdf_path):
    """
    Detect if the resume has a vertical or horizontal section layout.
    Returns True for vertical sections, False for horizontal/mixed layout.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page_width = pdf.pages[0].width
        page_height = pdf.pages[0].height
    
    # Group elements by Y position (lines)
    y_positions = {}
    for elem in elements:
        # Round y-position to nearest 5 units to group elements on roughly the same line
        y_bin = round(elem["y0"] / 5) * 5
        if y_bin not in y_positions:
            y_positions[y_bin] = []
        y_positions[y_bin].append(elem)
    
    # Check if there are multiple elements on the same line frequently
    lines_with_multiple_elements = sum(1 for elems in y_positions.values() if len(elems) > 1)
    total_lines = len(y_positions)
    
    # If most lines have multiple elements horizontally aligned, it's likely a horizontal layout
    # Otherwise, it's more likely to be a vertical layout
    horizontal_layout_ratio = lines_with_multiple_elements / total_lines if total_lines > 0 else 0
    
    # Also check for distribution of text across the page width
    x_positions = [elem["x0"] for elem in elements]
    x_distribution = {}
    
    # Create bins of x-positions (10 bins across page width)
    bin_width = page_width / 10
    for x in x_positions:
        bin_index = int(x / bin_width)
        x_distribution[bin_index] = x_distribution.get(bin_index, 0) + 1
    
    # Calculate if text is distributed evenly across the page (horizontal layout)
    # or concentrated in fewer columns (vertical layout)
    filled_bins = sum(1 for count in x_distribution.values() if count > len(elements) * 0.05)
    
    # If text is spread across many x-positions and many lines have multiple elements,
    # it's likely a horizontal layout
    is_vertical_layout = filled_bins <= 3 or horizontal_layout_ratio < 0.3
    
    return is_vertical_layout

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
    if not contains_sentence_ending_punctuation(text):  # Headings rarely end with periods
        confidence += 0.5
    
    # Check if the text matches common section headings
    heading_type = get_heading_type(text)
    if heading_type != "OTHER":
        confidence += 1.5
    
    # Contextual cues - check if following content matches section expectation
    following_text = get_following_text(element, all_elements, 5)
    if section_matches_content(heading_type, following_text):
        confidence += 2
    
    return confidence

def identify_section_headings(elements, column_divider, font_metrics, is_multi_column, is_vertical_layout):
    """
    Identify section headings based on text styling and patterns.
    Takes into account the multi-column layout and relative font sizes.
    """
    headings = []
    min_confidence_threshold = MIN_CONFIDENCE_THRESHOLD
    
    # First pass - identify potential headings
    for elem in elements:
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
                "font_size": elem["font_size"]
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
    for heading in filtered_headings:
        if heading["confidence"] < 10 and heading["type"] != "OTHER":
            following_text = get_following_text(heading, elements, 10)
            if section_matches_content(heading["type"], following_text):
                heading["confidence"] += 1.5
                logger.info(f"Boosted confidence for {heading['text']} based on content match")
    
    # Final filtering - only keep headings with sufficient confidence
    final_headings = [h for h in filtered_headings if h["confidence"] >= min_confidence_threshold]
    
    # For vertical layouts, sort headings top to bottom
    if is_vertical_layout:
        final_headings.sort(key=lambda h: (h["page"], h["y0"]))
    
    return final_headings
def extract_sections(headings, elements, column_divider, is_multi_column, is_vertical_layout):
    """
    Extract sections from the PDF, handling multi-column or single-column layout,
    and respecting section boundaries.
    """
    sections = {}
    
    if not headings:
        return sections
    
    # Group elements by page for efficiency
    elements_by_page = {}
    for elem in elements:
        page = elem["page"]
        if page not in elements_by_page:
            elements_by_page[page] = []
        elements_by_page[page].append(elem)
    
    # If vertical layout, use a top-to-bottom approach for all layouts
    if is_vertical_layout or not is_multi_column:
        # Sort all headings by page and vertical position
        sorted_headings = sorted(headings, key=lambda h: (h["page"], h["y0"]))
        
        # Process headings sequentially
        for i, heading in enumerate(sorted_headings):
            heading_page = heading["page"]
            heading_y = heading["y1"]  # Bottom of the heading
            
            # Find the next heading
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
                    
                    # In vertical layout, also check if element is horizontally aligned with the heading
                    if is_vertical_layout:
                        # Calculate horizontal overlap or proximity with a wider margin
                        heading_left = heading["x0"]
                        heading_right = heading["x1"]
                        element_left = elem["x0"]
                        element_right = elem["x1"]
                        
                        # For single column layouts, be more permissive with horizontal alignment
                        horizontal_overlap = (
                            (element_left <= heading_right + 150 and element_right >= heading_left - 150) or  # Wider margin
                            abs(element_left - heading_left) < 150  # Increased tolerance
                        )
                        
                        if not horizontal_overlap:
                            continue
                    
                    section_elements.append(elem)
            
            # Only add section if it has content
            if section_elements:
                section_key = f"{heading['type']}"
                sections[section_key] = {
                    "heading": heading,
                    "elements": section_elements,
                    "column": "single",
                    "page_span": [heading_page, end_page]  # Store page range for visualization
                }
    else:
        # Multi-column layout processing - improved to handle more complex layouts
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
                
                # Define the column boundaries with overlapping tolerance
                min_x = 0 if column == "left" else column_divider - 30  # Allow more overlap
                max_x = column_divider + 30 if column == "left" else float('inf')  # Allow more overlap
                
                # Process elements page by page
                for page_num in range(heading_page, min(end_page + 1, max(elements_by_page.keys()) + 1)):
                    if page_num not in elements_by_page:
                        continue
                    
                    for elem in elements_by_page[page_num]:
                        # Check if the element is in the correct column with more flexibility
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
                        "column": column,
                        "column_divider": column_divider,  # Store for later use
                        "page_span": [heading_page, end_page]  # Store page range for visualization
                    }
    
    return sections

def assign_orphaned_elements(sections, elements, headings):
    """Assign elements that don't belong to any section to the most appropriate section."""
    if not sections:
        return {}
        
    # Get all elements already assigned to sections
    assigned_elements = set()
    for section_key, section_data in sections.items():
        for elem in section_data["elements"]:
            # Create a unique identifier for the element
            elem_id = (elem["page"], elem["x0"], elem["y0"], elem["text"])
            assigned_elements.add(elem_id)
    
    # Find orphaned elements
    orphaned = []
    for elem in elements:
        elem_id = (elem["page"], elem["x0"], elem["y0"], elem["text"])
        if elem_id not in assigned_elements:
            # Skip elements that are section headings
            is_heading = False
            for section_data in sections.values():
                heading = section_data["heading"]
                if (elem["page"] == heading["page"] and 
                    elem["x0"] == heading["x0"] and 
                    elem["y0"] == heading["y0"]):
                    is_heading = True
                    break
            
            if not is_heading:
                orphaned.append(elem)
    
    # If no orphaned elements, return unchanged
    if not orphaned:
        return sections
    
    # Sort sections by page and position for assignment
    sorted_section_items = sorted(
        sections.items(),
        key=lambda x: (x[1]["heading"]["page"], x[1]["heading"]["y0"])
    )
    
    # Assign each orphaned element to the nearest section
    for elem in orphaned:
        # Find the nearest section
        best_section = None
        min_distance = float('inf')
        
        for section_key, section_data in sorted_section_items:
            heading = section_data["heading"]
            
            # Skip sections in different columns for multi-column layouts
            if section_data["column"] != "single":
                is_left_col = elem["x0"] < section_data.get("column_divider", float('inf'))
                if (is_left_col and section_data["column"] != "left") or \
                   (not is_left_col and section_data["column"] != "right"):
                    continue
            
            # Calculate distance (prioritize same page)
            if elem["page"] < heading["page"]:
                # Element is on earlier page
                distance = (heading["page"] - elem["page"]) * 1000
            elif elem["page"] > heading["page"]:
                # Element is on later page
                distance = (elem["page"] - heading["page"]) * 1000
            else:
                # Same page - calculate distance with more weight to vertical proximity
                if elem["y0"] < heading["y0"]:
                    # Element is above heading - likely belongs to previous section
                    distance = (heading["y0"] - elem["y0"]) + 500  # Penalty for being above
                else:
                    # Element is below heading - more likely to belong to this section
                    distance = elem["y0"] - heading["y0"]
            
            if distance < min_distance:
                min_distance = distance
                best_section = section_key
        
        # Add the element to the best section if found
        if best_section:
            sections[best_section]["elements"].append(elem)
    
    # Re-sort elements in each section
    for section_data in sections.values():
        section_data["elements"].sort(key=lambda e: (e["page"], e["y0"], e["x0"]))
    
    return sections

def clean_section_boundaries(sections, headings):
    """
    Improve section boundary detection by ensuring sections don't overlap
    and content doesn't extend beyond the next heading.
    """
    # First, sort headings by page and position
    sorted_headings = sorted(headings, key=lambda h: (h["page"], h["y0"]))
    
    # For each section, ensure content doesn't extend beyond the next logical heading
    for section_key, section_data in sections.items():
        heading = section_data["heading"]
        column = section_data["column"]
        
        # Find the next heading (in the same column for multi-column layouts)
        next_heading_page = float('inf')
        next_heading_y0 = float('inf')
        
        for h in sorted_headings:
            if column != "single" and h["column"] != column:
                continue  # Skip headings in different columns for multi-column layouts
                
            if (h["page"] > heading["page"] or 
                (h["page"] == heading["page"] and h["y0"] > heading["y1"])):
                if h["page"] < next_heading_page or (h["page"] == next_heading_page and h["y0"] < next_heading_y0):
                    next_heading_page = h["page"]
                    next_heading_y0 = h["y0"]
        
        # Filter elements to respect the next heading boundary
        filtered_elements = []
        for elem in section_data["elements"]:
            if (elem["page"] < next_heading_page or 
                (elem["page"] == next_heading_page and elem["y0"] < next_heading_y0 - 5)):  # 5pt buffer
                filtered_elements.append(elem)
        
        # Update the section with filtered elements
        section_data["elements"] = filtered_elements
    
    return sections

def post_process_sections(sections, elements, headings):
    """Apply post-processing to improve section identification."""
    
    # Step 1: Remove "OTHER" sections
    sections = {k: v for k, v in sections.items() if v["heading"]["type"] != "OTHER"}
    
    # Step 2: Fix section boundary issues
    sections = clean_section_boundaries(sections, headings)
    
    # Step 3: Fill in gaps (elements not assigned to any section)
    sections = assign_orphaned_elements(sections, elements, headings)
    
    # Step 4: Additional refinement for blue box issue - ensure each section has proper dimensions
    for section_key, section_data in sections.items():
        section_elements = section_data["elements"]
        
        # Skip empty sections
        if not section_elements:
            continue
        
        # Check if section spans too many pages (possible error)
        pages = set(elem["page"] for elem in section_elements)
        if len(pages) > 3:  # If section spans more than 3 pages, it might be misidentified
            # Try to limit to elements on the same or adjacent pages as the heading
            heading_page = section_data["heading"]["page"]
            filtered_elements = [
                elem for elem in section_elements 
                if abs(elem["page"] - heading_page) <= 1  # Same page or adjacent
            ]
            
            # Only update if we still have elements left
            if filtered_elements:
                section_data["elements"] = filtered_elements
        
        # Check for extreme horizontal spread (another potential cause of the blue box issue)
        if section_elements:
            x_min = min(elem["x0"] for elem in section_elements)
            x_max = max(elem["x1"] for elem in section_elements)
            
            # If horizontal spread is too wide for a single column, filter out potential outliers
            if section_data["column"] != "single" and (x_max - x_min) > 400:
                # Calculate average x-position
                avg_x = sum(elem["x0"] for elem in section_elements) / len(section_elements)
                
                # Filter out elements that are too far from the average
                filtered_elements = [
                    elem for elem in section_elements 
                    if abs(elem["x0"] - avg_x) < 200  # Max 200pt deviation
                ]
                
                # Only update if we still have elements left
                if filtered_elements:
                    section_data["elements"] = filtered_elements
    
    return sections

def visualize_sections(pdf_path, sections, headings, column_divider, is_multi_column):
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
        "PUBLICATIONS": (0.7, 0.3, 0.3), # Rust
        "LEADERSHIP": (0.3, 0.3, 0.7), # Lavender
        "STRENGTHS": (0.8, 0.5, 0.3), # Copper
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
        
        # Draw column divider line if it's a multi-column layout
        if is_multi_column:
            new_page.draw_line((column_divider, 0), (column_divider, orig_page.rect.height), 
                               color=(0.5, 0.5, 0.5), width=0.5, dashes="[2 2]")
            
            # Add a label for columns
            new_page.insert_text(fitz.Point(column_divider/2, 20), "LEFT COLUMN", 
                                fontsize=8, color=(0.5, 0.5, 0.5))
            new_page.insert_text(fitz.Point(column_divider + (orig_page.rect.width - column_divider)/2, 20), 
                                "RIGHT COLUMN", fontsize=8, color=(0.5, 0.5, 0.5))
        else:
            new_page.insert_text(fitz.Point(orig_page.rect.width / 2, 20), "SINGLE COLUMN", 
                                fontsize=8, color=(0.5, 0.5, 0.5))
        
        # Only draw headings with confidence >= MIN_CONFIDENCE_THRESHOLD
        high_confidence_headings = [h for h in headings if h.get("confidence", 10) >= MIN_CONFIDENCE_THRESHOLD]
        
        # Draw heading labels
        for heading in [h for h in high_confidence_headings if h["page"] == page_num]:
            heading_type = heading["type"]
            # Skip OTHER headings for visualization
            if heading_type == "OTHER":
                continue
                
            color = colors.get(heading_type, (0.3, 0.3, 0.3))
            
            # Draw a label above the heading
            label_point = fitz.Point(heading["x0"], heading["y0"] - 5)
            label_text = f"{heading_type}" + (f" (Confidence: {heading.get('confidence', 0):.1f})" if 'confidence' in heading else "")
            new_page.insert_text(label_point, label_text, fontsize=8, color=color)
            
            # Draw a rectangle around the heading
            rect = fitz.Rect(heading["x0"] - 2, heading["y0"] - 2, 
                            heading["x1"] + 2, heading["y1"] + 2)
            new_page.draw_rect(rect, color=color, width=1)
        
        # Draw section content boxes - only for non-OTHER sections
        for section_key, section_data in sections.items():
            section_type = section_data["heading"]["type"]
            # Skip OTHER sections for visualization
            if section_type == "OTHER":
                continue
                
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
            
            # Find next heading for boundary constraint
            next_heading_page, next_heading_y0 = float('inf'), float('inf')
            for h in high_confidence_headings:
                if (h["page"] > section_data["heading"]["page"] or 
                    (h["page"] == section_data["heading"]["page"] and h["y0"] > section_data["heading"]["y1"])):
                    if h["page"] < next_heading_page or (h["page"] == next_heading_page and h["y0"] < next_heading_y0):
                        next_heading_page = h["page"]
                        next_heading_y0 = h["y0"]
            
            # Constrain section box to not exceed next heading
            if page_num == next_heading_page and y1 >= next_heading_y0:
                y1 = next_heading_y0 - 5
            
            # Draw a rectangle around the section content
            rect = fitz.Rect(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
            new_page.draw_rect(rect, color=color, width=1.5, dashes="[2 2]")
            
            # Add a small label at the bottom of the section
            label_point = fitz.Point(x0, y1 + 15)
            label_text = f"{section_type}"
            new_page.insert_text(label_point, label_text, fontsize=7, color=color)
    
    # Add layout information to first page
    first_page = output[0]
    layout_info = f"Layout: {'Multi-column' if is_multi_column else 'Single-column'}"
    first_page.insert_text(fitz.Point(50, 50), layout_info, fontsize=10, color=(0, 0, 0))
    
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
            return None, {}, False
        
        # Analyze font metrics
        font_metrics = analyze_font_metrics(elements)
        logger.info(f"Font metrics analysis:")
        logger.info(f"  - Median size: {font_metrics['median_size']:.2f}")
        logger.info(f"  - Heading size threshold: {font_metrics['heading_size_threshold']:.2f}")
        
        # Determine if the document has a multi-column layout
        column_divider = analyze_layout(elements, pdf_path)
        logger.info(f"Layout analysis:")
        logger.info(f"  - Column divider at x-coordinate: {column_divider:.1f}")
        is_multi_column = False
        # Check if the layout is vertical
        is_vertical_layout = detect_layout_orientation(elements, pdf_path)
        logger.info(f"  - Vertical layout: {is_vertical_layout}")
        
        # Identify section headings, taking into account the column structure
        headings = identify_section_headings(elements, column_divider, font_metrics,is_multi_column, is_vertical_layout)
        logger.info(f"Identified {len(headings)} section headings")
        
        # Count headings in each column
        # if is_multi_column:
        #     left_headings = [h for h in headings if h["column"] == "left"]
        #     right_headings = [h for h in headings if h["column"] == "right"]
        #     logger.info(f"  - Left column: {len(left_headings)} headings")
        #     logger.info(f"  - Right column: {len(right_headings)} headings")
        
        for heading in headings:
            logger.info(f"  - {heading['text']} ({heading['type']}, {heading['column']} column, confidence: {heading['confidence']:.1f})")
        
        # Extract sections, handling the multi-column layout
        sections = extract_sections(headings, elements, column_divider, is_multi_column, is_vertical_layout)
        logger.info(f"Extracted {len(sections)} sections")
        
        # Apply post-processing to improve results
        sections = post_process_sections(sections, elements, headings)
        logger.info(f"After post-processing: {len(sections)} sections")
        
        # Visualize the sections and column structure
        out_path = visualize_sections(pdf_path, sections, headings, column_divider, is_multi_column)
        logger.info(f"Visualization saved to: {out_path}")
        
        return out_path, sections
        
    except Exception as e:
        logger.error(f"Error parsing resume {pdf_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}, False

def main():
    """Main function to run the resume parser on a single PDF or directory."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume Parser with Section Detection')
    parser.add_argument('input_path', help='Path to a resume PDF file or a directory of PDFs')
    parser.add_argument('-o', '--output_dir', help='Output directory (optional)')
    
    args = parser.parse_args()
    input_path = args.input_path
    
    if os.path.isdir(input_path):
        # Process a directory of PDFs
        pdf_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                    if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {input_path}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            out_path, sections = parse_resume(pdf_path)
            logger.info(f"Processed: {pdf_path}")
            logger.info(f"Output: {out_path}")
            logger.info(f"Sections detected: {len(sections)}")
            # logger.info(f"Layout: {'Multi-column' if is_multi_column else 'Single-column'}")
            logger.info("-" * 40)
    else:
        # Process a single PDF
        out_path, sections = parse_resume(input_path)
        if out_path:
            logger.info(f"Successfully processed: {input_path}")
            logger.info(f"Output: {out_path}")
            logger.info(f"Sections detected: {len(sections)}")
            # logger.info(f"Layout: {'Multi-column' if is_multi_column else 'Single-column'}")
        else:
            logger.error(f"Failed to process: {input_path}")

if __name__ == "__main__":
    main()