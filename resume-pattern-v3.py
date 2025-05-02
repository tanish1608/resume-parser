import pdfplumber
import fitz  # PyMuPDF
import os
import re
import json
from collections import defaultdict
import sys
import time
import statistics
import numpy as np
from scipy.signal import find_peaks
import logging

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

def normalize(text):
    """Normalize text by removing whitespace and converting to lowercase."""
    return re.sub(r'\s+', '', text).lower()

def contains_sentence_ending_punctuation(text):
    """Check if text contains sentence-ending punctuation."""
    return bool(re.search(r'[.!?]$', text.strip()))

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
    
    # Determine thresholds
    heading_size_threshold = max(mode_size * 1.15, median_size * 1.1)  # 15% larger than mode or 10% larger than median
    small_text_threshold = min(mode_size * 0.9, median_size * 0.85)    # 10% smaller than mode or 15% smaller than median
    
    # Find distinct size clusters
    sorted_sizes = sorted(list(set([round(s, 1) for s in font_sizes])))
    size_clusters = []
    
    if len(sorted_sizes) > 1:
        # Look for gaps in font sizes
        for i in range(1, len(sorted_sizes)):
            if sorted_sizes[i] - sorted_sizes[i-1] > 1.0:  # Significant gap
                size_clusters.append((sorted_sizes[i-1], sorted_sizes[i]))
    
    return {
        "median_size": median_size,
        "mode_size": mode_size,
        "heading_size_threshold": heading_size_threshold,
        "small_text_threshold": small_text_threshold,
        "size_clusters": size_clusters,
        "font_size_distribution": size_counts
    }

def analyze_layout(elements, pdf_path):
    """
    Enhanced layout analysis to detect complex column structures.
    Returns column dividers and whether it's a multi-column layout.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page_width = pdf.pages[0].width
        page_height = pdf.pages[0].height
    
    # Create a visual "heat map" of text positions
    x_positions = defaultdict(int)
    for elem in elements:
        x_bin = round(elem["x0"] / 5) * 5  # 5pt bins
        x_positions[x_bin] += 1
    
    # Check if we have a multi-column layout
    left_region = sum(x_positions.get(x, 0) for x in range(0, int(page_width * 0.4), 5))
    middle_region = sum(x_positions.get(x, 0) for x in range(int(page_width * 0.4), int(page_width * 0.6), 5))
    right_region = sum(x_positions.get(x, 0) for x in range(int(page_width * 0.6), int(page_width), 5))
    
    total_elements = left_region + middle_region + right_region
    if total_elements == 0:
        return page_width * 0.6, False  # Default, no multi-column
    
    left_ratio = left_region / total_elements
    middle_ratio = middle_region / total_elements
    right_ratio = right_region / total_elements
    
    # Generate density profile for visualization
    x_vals = sorted(x_positions.keys())
    density = [x_positions[x] for x in x_vals]
    
    # Check for significant gaps in x-distribution
    is_multi_column = (left_ratio > 0.25 and right_ratio > 0.25 and middle_ratio < 0.15)
    
    # If we have too few points, fall back to simple approach
    if len(x_vals) < 10:
        if is_multi_column:
            return page_width * 0.5, True
        else:
            return page_width * 0.6, False
    
    # Use peak detection to find columns
    try:
        peaks, _ = find_peaks(density, height=max(density)*0.3, distance=page_width*0.1/5)
        peak_positions = [x_vals[p] for p in peaks]
        
        if len(peak_positions) <= 1:
            # Only one column detected
            return page_width * 0.6, False
        
        # Find valleys between peaks to identify column dividers
        valleys = []
        for i in range(len(peaks)-1):
            start_idx = peaks[i]
            end_idx = peaks[i+1]
            if start_idx >= len(density) or end_idx >= len(density):
                continue
                
            between_segment = density[start_idx:end_idx+1]
            if between_segment:
                min_val = min(between_segment)
                if min_val < max(density) * 0.3:  # Significant valley
                    min_idx = between_segment.index(min_val) + start_idx
                    if min_idx < len(x_vals):
                        valley_position = x_vals[min_idx]
                        valleys.append(valley_position)
        
        # Use the most significant valley as column divider
        if valleys and is_multi_column:
            # Find the valley closest to the middle
            middle_x = page_width / 2
            divider = min(valleys, key=lambda v: abs(v - middle_x))
            return divider, True
    except Exception as e:
        logger.error(f"Error in peak detection: {str(e)}")
        # Fall back to simple method
    
    # Default approach if peak detection fails
    if is_multi_column:
        return page_width * 0.5, True
    else:
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

def extract_sections(headings, elements, column_divider, is_multi_column):
    """
    Extract sections from the PDF, handling multi-column or single-column layout.
    """
    sections = {}
    
    if not headings:
        return sections
    
    # Group elements by page for efficiency
    elements_by_page = defaultdict(list)
    for elem in elements:
        elements_by_page[elem["page"]].append(elem)
    
    if not is_multi_column:
        # Single-column layout processing
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
                    
                    section_elements.append(elem)
            
            # Only add section if it has content
            if section_elements:
                section_key = f"{heading['type']}"
                sections[section_key] = {
                    "heading": heading,
                    "elements": section_elements,
                    "column": "single"
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
                
                # Only add section if it has content
                if section_elements:
                    section_key = f"{heading['type']}_{column}"
                    sections[section_key] = {
                        "heading": heading,
                        "elements": section_elements,
                        "column": column
                    }
    
    return sections

def consolidate_sections(sections):
    """Merge and deduplicate similar sections."""
    if not sections:
        return {}
        
    # Group sections by type
    grouped_sections = defaultdict(list)
    for section_key, section_data in sections.items():
        section_type = section_data["heading"]["type"]
        grouped_sections[section_type].append((section_key, section_data))
    
    # Consolidate each group
    consolidated = {}
    for section_type, section_group in grouped_sections.items():
        if len(section_group) == 1:
            # Only one section of this type, no need to consolidate
            section_key, section_data = section_group[0]
            consolidated[section_key] = section_data
        else:
            # Multiple sections of same type - analyze further
            
            # Check if they're from different columns in multi-column layout
            columns = set(data["column"] for _, data in section_group)
            
            if len(columns) > 1 and "left" in columns and "right" in columns:
                # Sections in different columns - keep separate but standardize naming
                for i, (key, data) in enumerate(section_group):
                    column = data["column"]
                    new_key = f"{section_type}_{column}"
                    consolidated[new_key] = data
            else:
                # Merge sections of same type in same column
                # Sort by confidence score
                sorted_sections = sorted(section_group, 
                                       key=lambda x: x[1]["heading"]["confidence"], 
                                       reverse=True)
                
                # Use the highest confidence section as base
                best_key, best_data = sorted_sections[0]
                best_elements = list(best_data["elements"])  # Create a copy
                
                # Add elements from other sections
                for _, other_data in sorted_sections[1:]:
                    best_elements.extend(other_data["elements"])
                
                # Sort all elements by page and position
                best_elements.sort(key=lambda e: (e["page"], e["y0"], e["x0"]))
                
                # Update the section with merged elements
                best_data["elements"] = best_elements
                consolidated[best_key] = best_data
    
    return consolidated

def assign_orphaned_elements(sections, elements):
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
                # Same page - calculate Euclidean distance
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

def fix_common_section_errors(sections):
    """Apply heuristics to fix common section detection errors."""
    if not sections:
        return {}
        
    # Check for sections with few elements that might be misidentified
    for section_key, section_data in list(sections.items()):
        # If a section has very few elements, check content
        if len(section_data["elements"]) < 3:
            # Get text from this small section
            text = " ".join(e["text"] for e in section_data["elements"]).lower()
            
            # Check if content suggests a different section type
            current_type = section_data["heading"]["type"]
            potential_type = None
            
            # Example heuristics
            if "skills" in text and "programming" in text and current_type != "SKILLS":
                potential_type = "SKILLS"
            elif any(word in text for word in ["university", "college", "degree", "gpa"]) and current_type != "EDUCATION":
                potential_type = "EDUCATION"
            elif any(word in text for word in ["developed", "created", "built", "implemented"]) and current_type != "PROJECTS":
                potential_type = "PROJECTS"
            
            # Update section type if warranted
            if potential_type:
                logger.info(f"Changing section type from {current_type} to {potential_type} based on content")
                new_key = section_key.replace(current_type, potential_type)
                section_data["heading"]["type"] = potential_type
                sections[new_key] = section_data
                del sections[section_key]
    
    return sections

def establish_section_hierarchy(sections):
    """Determine section hierarchy to handle main sections vs subsections."""
    if not sections:
        return {}
        
    # Analyze heading styling to detect hierarchy
    heading_font_sizes = []
    for section_data in sections.values():
        # Check if font_size exists in the heading dictionary, otherwise use a default value
        if "font_size" in section_data["heading"]:
            heading_font_sizes.append(section_data["heading"]["font_size"])
        else:
            # If font_size isn't available, skip this heading for hierarchy calculation
            logger.warning(f"Missing font_size in heading: {section_data['heading']['text']}")
    
    # If we have multiple distinct heading sizes and enough samples, we might have subsections
    if len(heading_font_sizes) > 1:
        mean_size = sum(heading_font_sizes) / len(heading_font_sizes)
        std_dev = (sum((x - mean_size) ** 2 for x in heading_font_sizes) / len(heading_font_sizes)) ** 0.5
        
        # If there's significant variation in heading sizes
        if std_dev > 1.0:
            # Calculate size threshold for main sections vs subsections
            main_section_threshold = mean_size + 0.5 * std_dev
            
            # Mark each section as main or sub based on heading size
            for section_key, section_data in sections.items():
                if "font_size" in section_data["heading"] and section_data["heading"]["font_size"] >= main_section_threshold:
                    section_data["is_main_section"] = True
                else:
                    section_data["is_main_section"] = False
    
    return sections

def extract_section_text(sections):
    """Convert the section elements into readable text with improved formatting."""
    section_texts = {}
    
    for section_key, section_data in sections.items():
        elements = section_data["elements"]
        section_type = section_data["heading"]["type"]
        column = section_data["column"]
        
        # Skip empty sections
        if not elements:
            continue
        
        # Sort elements by page, then by y position, then by x position
        sorted_elements = sorted(elements, key=lambda e: (e["page"], e["y0"], e["x0"]))
        
        # Group elements by line using a more sophisticated approach
        lines = defaultdict(list)
        line_heights = []  # Store line heights to calculate average
        current_page = None
        line_count = 0
        
        for elem in sorted_elements:
            # Reset line tracking when page changes
            if elem["page"] != current_page:
                current_page = elem["page"]
                line_count += 100  # Ensure lines on different pages have different IDs
            
            # Calculate element height
            elem_height = elem["y1"] - elem["y0"]
            if elem_height > 0:
                line_heights.append(elem_height)
            
            # If this is the first element being processed
            if not lines:
                line_key = (elem["page"], line_count)
                lines[line_key].append(elem)
                continue
            
            # Get the most recent line key and its elements
            recent_line_keys = sorted([k for k in lines.keys() if k[0] == elem["page"]], 
                                     key=lambda k: k[1], 
                                     reverse=True)
            
            if not recent_line_keys:
                # No lines on this page yet
                line_key = (elem["page"], line_count)
                lines[line_key].append(elem)
                continue
                
            recent_line_key = recent_line_keys[0]
            recent_line_elements = lines[recent_line_key]
            
            # Calculate average line height so far
            avg_line_height = sum(line_heights) / len(line_heights) if line_heights else 12
            
            # Check if this element belongs to the recent line
            last_elem = recent_line_elements[-1]
            vertical_distance = elem["y0"] - last_elem["y0"]
            
            if abs(vertical_distance) < avg_line_height * 0.5:
                # Same line - elements may be horizontally offset for formatting
                lines[recent_line_key].append(elem)
            else:
                # New line
                line_count += 1
                line_key = (elem["page"], line_count)
                lines[line_key].append(elem)
        
        # Build the text line by line
        text_lines = []
        for line_key in sorted(lines.keys()):
            line_elements = sorted(lines[line_key], key=lambda e: e["x0"])
            
            # Format the line text
            line_text = ""
            for i, elem in enumerate(line_elements):
                if i > 0:
                    # Check spacing between elements
                    prev_elem = line_elements[i-1]
                    gap = elem["x0"] - prev_elem["x1"]
                    
                    # Add appropriate spacing based on gap
                    if gap > 10:
                        line_text += " "  # Standard space for larger gaps
                    elif gap > 2:
                        line_text += " "  # Still need space for smaller gaps
                    # For very small or negative gaps, no additional space
                
                line_text += elem["text"]
            
            text_lines.append(line_text)
        
        # Format text based on section type
        formatted_text = format_section_text(section_type, text_lines)
        
        # Use a simple section key for better readability
        simple_key = section_type
        
        if simple_key in section_texts:
            # If we already have this section type, append new content
            section_texts[simple_key] += "\n\n--- " + column.upper() + " COLUMN ---\n" + formatted_text
        else:
            section_texts[simple_key] = "--- " + column.upper() + " COLUMN ---\n" + formatted_text
    
    return section_texts

def format_section_text(section_type, text_lines):
    """Format text based on section type for better readability."""
    if not text_lines:
        return ""
    
    if section_type in ["EDUCATION", "EXPERIENCE"]:
        # Try to identify date patterns and organizations for better formatting
        formatted_lines = []
        current_entry = []
        
        for line in text_lines:
            # Check if this line looks like the start of a new entry
            # (contains a year or organization name followed by date)
            if re.search(r'(20\d\d|19\d\d|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|present)', line.lower()):
                # Save the previous entry if it exists
                if current_entry:
                    formatted_lines.append("\n".join(current_entry))
                    formatted_lines.append("")  # Empty line between entries
                
                # Start a new entry
                current_entry = [line]
            else:
                # Continue the current entry
                current_entry.append(line)
        
        # Add the last entry
        if current_entry:
            formatted_lines.append("\n".join(current_entry))
        
        return "\n".join(formatted_lines)
    
    elif section_type == "SKILLS":
        # Try to detect if skills are presented as a list or in paragraphs
        if any("," in line for line in text_lines):
            # Likely a comma-separated list
            skills_text = "\n".join(text_lines)
            # Split by commas, clean up, and rejoin with proper formatting
            skills = [s.strip() for s in re.split(r',|\n', skills_text) if s.strip()]
            return "• " + "\n• ".join(skills)
        else:
            # Try to detect bullet points or similar list markers
            bullet_pattern = re.compile(r'^[\s•\-\*]+')
            has_bullets = any(bullet_pattern.match(line.strip()) for line in text_lines)
            
            if has_bullets:
                # Already has bullet formatting
                return "\n".join(text_lines)
            else:
                # Add bullets if lines seem like separate skills
                if all(len(line.strip()) < 50 for line in text_lines):
                    return "• " + "\n• ".join(line.strip() for line in text_lines)
                else:
                    # Longer lines - keep as paragraphs
                    return "\n".join(text_lines)
    
    elif section_type in ["PROJECTS", "ACHIEVEMENTS"]:
        # Check for project names and descriptions
        formatted_lines = []
        current_project = []
        
        for line in text_lines:
            # Check if this line looks like a project title
            # (short line, possibly with date or technology mention)
            if (len(line.strip()) < 60 and 
                ("|" in line or ":" in line or 
                 re.search(r'(20\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', line.lower()))):
                # Save the previous project if it exists
                if current_project:
                    formatted_lines.append("\n".join(current_project))
                    formatted_lines.append("")  # Empty line between projects
                
                # Start a new project
                current_project = [line]
            else:
                # Continue the current project
                current_project.append(line)
        
        # Add the last project
        if current_project:
            formatted_lines.append("\n".join(current_project))
        
        return "\n".join(formatted_lines)
    
    else:
        # Default formatting for other section types
        return "\n".join(text_lines)

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
        
        # Draw heading labels
        for heading in [h for h in headings if h["page"] == page_num]:
            heading_type = heading["type"]
            color = colors.get(heading_type, (0.3, 0.3, 0.3))
            
            # Draw a label above the heading
            label_point = fitz.Point(heading["x0"], heading["y0"] - 5)
            label_text = f"{heading_type} (Confidence: {heading['confidence']:.1f})"
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

def post_process_sections(sections, elements):
    """Apply post-processing to improve section identification."""
    # Step 1: Consolidate similar sections
    sections = consolidate_sections(sections)
    
    # Step 2: Fill in gaps (elements not assigned to any section)
    sections = assign_orphaned_elements(sections, elements)
    
    # Step 3: Fix common section detection errors
    sections = fix_common_section_errors(sections)
    
    # Step 4: Improve section hierarchy
    sections = establish_section_hierarchy(sections)
    
    return sections

def parse_resume(pdf_path):
    """Parse a resume PDF file, extract sections from multi-column layout, and visualize them."""
    logger.info(f"Processing resume: {pdf_path}")
    
    try:
        # Extract text elements with style information
        elements = extract_text_with_style(pdf_path)
        logger.info(f"Extracted {len(elements)} text elements")
        
        if not elements:
            logger.error(f"No text elements found in {pdf_path}")
            return None, {}, False  # Return is_multi_column as False when no elements found
        
        # Analyze font metrics
        font_metrics = analyze_font_metrics(elements)
        logger.info(f"Font metrics analysis:")
        logger.info(f"  - Median size: {font_metrics['median_size']:.2f}")
        logger.info(f"  - Heading size threshold: {font_metrics['heading_size_threshold']:.2f}")
        
        # Determine if the document has a multi-column layout
        column_divider, is_multi_column = analyze_layout(elements, pdf_path)
        logger.info(f"Layout analysis:")
        logger.info(f"  - Multi-column: {is_multi_column}")
        logger.info(f"  - Column divider at x-coordinate: {column_divider:.1f}")
        
        # Identify section headings, taking into account the column structure
        headings = identify_section_headings(elements, column_divider, font_metrics, is_multi_column)
        logger.info(f"Identified {len(headings)} section headings:")
        
        # Count headings in each column
        if is_multi_column:
            left_headings = [h for h in headings if h["column"] == "left"]
            right_headings = [h for h in headings if h["column"] == "right"]
            logger.info(f"  - Left column: {len(left_headings)} headings")
            logger.info(f"  - Right column: {len(right_headings)} headings")
        
        for heading in headings:
            logger.info(f"  - {heading['text']} ({heading['type']}, {heading['column']} column, confidence: {heading['confidence']:.1f})")
        
        # Extract sections, handling the multi-column layout
        sections = extract_sections(headings, elements, column_divider, is_multi_column)
        logger.info(f"Extracted {len(sections)} sections")
        
        # Apply post-processing to improve results
        sections = post_process_sections(sections, elements)
        logger.info(f"After post-processing: {len(sections)} sections")
        
        # Extract text from each section
        section_texts = extract_section_text(sections)
        
        # Visualize the sections and column structure
        out_path = visualize_sections(pdf_path, sections, headings, column_divider, is_multi_column)
        
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
        
        return out_path, section_texts, is_multi_column  # Return is_multi_column
        
    except Exception as e:
        logger.error(f"Error parsing resume {pdf_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}, False  # Return is_multi_column as False on error

def process_resume_folder(folder_path, output_folder=None):
    """
    Process all PDF files in a folder to identify and extract resume sections.
    
    Args:
        folder_path (str): Path to the folder containing resume PDFs
        output_folder (str, optional): Path to save output files. If None, uses folder_path + "_output"
    
    Returns:
        list: List of processed file paths and their extracted sections
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Error: {folder_path} is not a valid directory")
        return []
    
    if output_folder is None:
        output_folder = folder_path + "_output"
    
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
        try:
            # Now capturing is_multi_column from parse_resume
            out_path, sections, is_multi_column = parse_resume(pdf_path)
            elapsed_time = time.time() - start_time
            
            # Save each section to a separate text file
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Create a subfolder for this resume
            resume_folder = os.path.join(output_folder, base_name)
            os.makedirs(resume_folder, exist_ok=True)
            
            # Save each section
            for section_type, text in sections.items():
                section_file = os.path.join(resume_folder, f"{section_type}.txt")
                with open(section_file, 'w', encoding='utf-8') as f:
                    f.write(text)
            
            # Save full resume content as JSON
            json_file = os.path.join(resume_folder, f"{base_name}_sections.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                # Create a serializable version of sections
                serializable_sections = {}
                for section_type, text in sections.items():
                    serializable_sections[section_type] = text
                json.dump(serializable_sections, f, ensure_ascii=False, indent=2)
            
            # Save summary info - now correctly using is_multi_column
            summary_file = os.path.join(resume_folder, "summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Resume: {pdf_path}\n")
                f.write(f"Processing time: {elapsed_time:.2f} seconds\n")
                f.write(f"Layout: {'Multi-column' if is_multi_column else 'Single-column'}\n")
                f.write(f"Sections found: {', '.join(sections.keys())}\n")
            
            # Copy the highlighted PDF if it was created
            if out_path:
                import shutil
                highlighted_pdf = os.path.join(resume_folder, f"{base_name}_highlighted.pdf")
                shutil.copy2(out_path, highlighted_pdf)
            
            results.append({
                'pdf_path': pdf_path,
                'out_path': out_path,
                'sections': list(sections.keys()),
                'processing_time': elapsed_time,
                'is_multi_column': is_multi_column
            })
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
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
                f.write(f"  Layout: {'Multi-column' if r.get('is_multi_column', False) else 'Single-column'}\n")
                f.write(f"  Sections: {', '.join(r['sections'])}\n")
                f.write(f"  Processing time: {r['processing_time']:.2f} seconds\n\n")
    
    logger.info(f"\nProcessing complete. Results saved to {output_folder}")
    logger.info(f"Summary file created at {summary_path}")
    
    return results


def analyze_resume_sections(folder_path):
    """
    Analyze a folder of processed resumes to identify common sections and patterns.
    
    Args:
        folder_path (str): Path to the folder containing processed resume data
        
    Returns:
        dict: Analysis results
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Error: {folder_path} is not a valid directory")
        return {}
    
    # Get all subdirectories (each should be a processed resume)
    resume_dirs = [d for d in os.listdir(folder_path) 
                   if os.path.isdir(os.path.join(folder_path, d)) and d != "__pycache__"]
    
    if not resume_dirs:
        logger.error(f"No processed resume directories found in {folder_path}")
        return {}
    
    logger.info(f"Analyzing {len(resume_dirs)} processed resumes...")
    
    # Initialize counters
    section_counts = defaultdict(int)
    section_lengths = defaultdict(list)
    section_content_samples = defaultdict(list)
    
    # Process each resume directory
    for resume_dir in resume_dirs:
        dir_path = os.path.join(folder_path, resume_dir)
        
        # Get all text files (sections)
        section_files = [f for f in os.listdir(dir_path) 
                        if f.endswith('.txt') and f != "summary.txt"]
        
        # Count each section type
        for section_file in section_files:
            section_type = os.path.splitext(section_file)[0]
            section_counts[section_type] += 1
            
            # Calculate section length (character count)
            file_path = os.path.join(dir_path, section_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                section_lengths[section_type].append(len(content))
                
                # Store a sample of the content (first 100 chars)
                if len(section_content_samples[section_type]) < 3:  # Store up to 3 samples per section
                    section_content_samples[section_type].append(content[:100] + "...")
    
    # Calculate statistics
    results = {
        "total_resumes": len(resume_dirs),
        "section_frequency": {k: v/len(resume_dirs) for k, v in section_counts.items()},
        "section_counts": dict(section_counts),
        "average_section_length": {k: sum(v)/len(v) if v else 0 for k, v in section_lengths.items()},
        "section_samples": section_content_samples
    }
    
    # Print analysis results
    logger.info("\nResume Section Analysis:")
    logger.info("=======================")
    logger.info(f"Total resumes analyzed: {results['total_resumes']}")
    logger.info("\nSection frequency (% of resumes containing section):")
    for section, freq in sorted(results["section_frequency"].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {section}: {freq*100:.1f}%")
    
    logger.info("\nAverage section length (characters):")
    for section, length in sorted(results["average_section_length"].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {section}: {length:.0f} chars")
    
    # Save analysis to file
    analysis_path = os.path.join(folder_path, "section_analysis.json")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        # Create serializable version
        serializable_results = {
            "total_resumes": results["total_resumes"],
            "section_frequency": results["section_frequency"],
            "section_counts": results["section_counts"],
            "average_section_length": results["average_section_length"],
            "section_samples": {k: v[:1] for k, v in section_content_samples.items()}  # Just one sample per section
        }
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Analysis saved to {analysis_path}")
    
    return results

def generate_resume_report(pdf_path, sections, out_path):
    """Generate a comprehensive report of the resume parsing results."""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    report_path = os.path.join(os.path.dirname(out_path), f"{base_name}_report.html")
    
    # Create HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resume Analysis: {base_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .section h2 {{ margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
            pre {{ background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .stats {{ display: flex; flex-wrap: wrap; }}
            .stat-box {{ flex: 1; min-width: 200px; margin: 10px; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
            .visual {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Resume Analysis Report</h1>
            <div class="meta">
                <p><strong>File:</strong> {pdf_path}</p>
                <p><strong>Analysis Date:</strong> {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}</p>
            </div>
            
            <div class="section">
                <h2>Overview</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>Total Sections</h3>
                        <p>{len(sections)}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Layout</h3>
                        <p>{"Multi-column" if any("_left" in k or "_right" in k for k in sections) else "Single-column"}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Sections Detected</h2>
                <ul>
    """
    
    # Add each section to the report
    for section_type, text in sections.items():
        # Strip the column suffix for display
        display_name = section_type.split('_')[0] if '_' in section_type else section_type
        column = "Left Column" if "_left" in section_type else "Right Column" if "_right" in section_type else "Single Column"
        
        html += f"""
                    <li><strong>{display_name}</strong> ({column})</li>
        """
    
    html += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Section Content</h2>
    """
    
    # Add content of each section
    for section_type, text in sections.items():
        # Strip the column suffix for display
        display_name = section_type.split('_')[0] if '_' in section_type else section_type
        column = "Left Column" if "_left" in section_type else "Right Column" if "_right" in section_type else "Single Column"
        
        html += f"""
                <div class="section">
                    <h3>{display_name} ({column})</h3>
                    <pre>{text}</pre>
                </div>
        """
    
    html += """
            </div>
            
            <div class="section">
                <h2>Visual Analysis</h2>
                <p>For visual analysis of the resume structure, please refer to the highlighted PDF.</p>
                <p><a href="{0}_highlighted.pdf" target="_blank">View Highlighted PDF</a></p>
            </div>
        </div>
    </body>
    </html>
    """.format(base_name)
    
    # Write the report to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Report generated: {report_path}")
    
    return report_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if os.path.isdir(input_path):
            # Process all PDFs in the folder
            output_folder = sys.argv[2] if len(sys.argv) > 2 else None
            results = process_resume_folder(input_path, output_folder)
            
            # Analyze the results if we have an output folder
            if output_folder:
                analysis = analyze_resume_sections(output_folder)
                logger.info("Resume analysis complete.")
        else:
            # Process a single PDF
            out_path, sections = parse_resume(input_path)
            if out_path:
                # Generate report
                report_path = generate_resume_report(input_path, sections, out_path)
                logger.info(f"Processing complete. Report saved to: {report_path}")
    else:
        print("Usage:")
        print("  For a single resume: python resume_parser.py path/to/resume.pdf")
        print("  For a folder of resumes: python resume_parser.py path/to/resume/folder [output_folder]")