import pdfplumber
import fitz  # PyMuPDF
import os
import re
from collections import defaultdict
import string
from difflib import SequenceMatcher
import sys
import time

def normalize(text):
    """Normalize text by removing whitespace, punctuation and converting to lowercase."""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace and convert to lowercase
    return re.sub(r'\s+', '', text).lower()

def calculate_similarity(text1, text2):
    """Calculate text similarity using SequenceMatcher."""
    normalized_text1 = normalize(text1)
    normalized_text2 = normalize(text2)
    # Return similarity ratio between 0 and 1
    return SequenceMatcher(None, normalized_text1, normalized_text2).ratio()

class ResumeSection:
    """Class to represent a section in a resume with hierarchical structure."""
    def __init__(self, heading, section_type, level=0, parent=None):
        self.heading = heading  # Original heading text
        self.section_type = section_type  # Standardized section type
        self.level = level  # 0 for main section, 1+ for subsections
        self.parent = parent  # Parent section (None for main sections)
        self.children = []  # Child sections (subsections)
        self.elements = []  # Content elements belonging to this section
        self.coordinates = {}  # Position information (x0, y0, x1, y1, page)
        self.column = None  # Left or right column
        self.confidence = 0.0  # Confidence score for section detection
    
    def add_child(self, child_section):
        """Add a child (subsection) to this section."""
        self.children.append(child_section)
        child_section.parent = self
    
    def add_element(self, element):
        """Add a content element to this section."""
        self.elements.append(element)
    
    def set_coordinates(self, coords):
        """Set the position coordinates for this section."""
        self.coordinates = coords
    
    def set_column(self, column):
        """Set the column (left/right) for this section."""
        self.column = column
    
    def set_confidence(self, confidence):
        """Set the confidence score for this section detection."""
        self.confidence = confidence
    
    def get_content_text(self):
        """Get the text content of this section, excluding subsections."""
        # Sort elements by page number and y-position
        sorted_elements = sorted(self.elements, key=lambda e: (e.get("page", 0), e.get("y0", 0)))
        return "\n".join([elem.get("text", "") for elem in sorted_elements])
    
    def __str__(self):
        """String representation of the section."""
        indent = "  " * self.level
        return f"{indent}[{self.section_type}] {self.heading} ({self.confidence:.2f})"

class ResumeParser:
    """Enhanced resume parser with improved section detection."""
    
    # Mapping of common section names to standardized types
    SECTION_TYPE_MAPPING = {
        # Education-related sections
        "education  ": "EDUCATION",
        "academic  ": "EDUCATION",
        "qualification  ": "EDUCATION",
        "degree  ": "EDUCATION",
        "educational  ": "EDUCATION",
        
        # Experience-related sections
        "experience  ": "EXPERIENCE",
        "workexperience  ": "EXPERIENCE",
        "employment  ": "EXPERIENCE",
        "work  ": "EXPERIENCE",
        "career  ": "EXPERIENCE",
        "professional  ": "EXPERIENCE",
        "internship  ": "EXPERIENCE",
        "intern  ": "EXPERIENCE",
        
        # Skills-related sections
        "skills  ": "SKILLS",
        "technicalskills  ": "SKILLS",
        "softskill  ": "SKILLS",
        "competencies  ": "SKILLS",
        "expertise  ": "SKILLS",
        "technologies  ": "SKILLS",
        "tools  " : "SKILLS",
        "programming  ": "SKILLS",
        "language  ": "SKILLS",
        "languages  ": "SKILLS",
        
        # Projects-related sections
        "projects  ": "PROJECTS",
        "portfolio  ": "PROJECTS",
        "academic projects  ": "PROJECTS",
        "personal projects  ": "PROJECTS",
        
        # Contact-related sections
        "contact  ": "CONTACT",
        "contactme  ": "CONTACT",
        "contactinformation  ": "CONTACT",
        
        # Summary/Profile-related sections
        "summary  ": "PROFILE",
        "profile  ": "PROFILE",
        "objective  ": "PROFILE",
        "about  ": "PROFILE",
        "aboutme  ": "PROFILE",
        "bio  ": "PROFILE",
        "career objective  ": "PROFILE",
        "professional summary  ": "PROFILE",
        "careerobjective  ": "PROFILE",
        
        # Certifications-related sections
        "certifications  ": "CERTIFICATIONS",
        "certification  ": "CERTIFICATIONS",
        "credentials  ": "CERTIFICATIONS",
        "license  ": "CERTIFICATIONS",
        "licenses  ": "CERTIFICATIONS",
        
        # Awards-related sections
        "awards  ": "AWARDS",
        "honors  ": "AWARDS",
        "achievements  ": "AWARDS",
        "recognition  ": "AWARDS",
        "accomplishments  ": "AWARDS",
        "achievement  ": "AWARDS",
        
        # Languages-related sections
        "languages  ": "LANGUAGES",
        "languageskills  ": "LANGUAGES",
        "languageproficiency  ": "LANGUAGES",
        "spokenlanguages  ": "LANGUAGES",
        "languagesknown  ": "LANGUAGES",
        
        # Extracurricular sections
        "extracurricular  ": "EXTRACURRICULAR",
        "activities  ": "EXTRACURRICULAR",
        "volunteer  ": "EXTRACURRICULAR",
        "volunteering  ": "EXTRACURRICULAR",
        "leadership  ": "EXTRACURRICULAR",
        "extracirculuar  ": "EXTRACURRICULAR",
        
        # References sections
        "references  ": "REFERENCES",
        "reference  ": "REFERENCES",
        "recommendation  ": "REFERENCES",
        "recommendations  ": "REFERENCES",
        
        # Publications sections
        "publications  ": "PUBLICATIONS",
        "papers  ": "PUBLICATIONS",
        "research  ": "PUBLICATIONS",
        
        # Declaration sections
        "declaration  ": "DECLARATION",
        "decleration  ": "DECLARATION",
    }
    
    # Contextual validation patterns for sections
    SECTION_VALIDATION_PATTERNS = {
        "EDUCATION": [
            r"\b(?:20\d{2}|19\d{2})\b",  # Years (2000-2023 or 1900-1999)
            r"\b(?:university|college|school|institute|academy)\b",  # Educational institutions
            r"\b(?:bachelor|master|phd|degree|diploma|certification|gpa|cgpa)\b",  # Educational terms
            r"\b(?:B\.Tech|M\.Tech|B\.E|M\.E|B\.Sc|M\.Sc|MBA|BBA)\b"  # Common degree abbreviations
        ],
        "EXPERIENCE": [
            r"\b(?:20\d{2}|19\d{2})\b",  # Years
            r"\b(?:present|current|now)\b",  # Current work indicators
            r"\b(?:month|year|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",  # Time indicators
            r"\b(?:company|organization|firm|corporation|inc|ltd)\b",  # Company indicators
            r"\b(?:job|position|role|title|work|responsibility|team)\b"  # Job indicators
        ],
        "SKILLS": [
            r"\b(?:proficient|familiar|experienced|working knowledge|expertise|advanced|intermediate|beginner)\b",  # Proficiency levels
            r"\b(?:programming|language|technology|tool|framework|platform|software|hardware)\b",  # Skill categories
            r"\b(?:python|java|javascript|c\+\+|html|css|sql|react|node|php|tensor|linux|windows|office)\b"  # Common tech skills
        ],
        "PROJECTS": [
            r"\b(?:project|developed|implemented|created|built|designed)\b",  # Project activities
            r"\b(?:team|collaborated|solo|individual|group)\b",  # Project roles
            r"\b(?:github|link|url|source|code|repository)\b",  # Project links
            r"\b(?:application|website|system|platform|solution|tool)\b"  # Project types
        ],
        "PROFILE": [
            r"\b(?:seeking|looking|aim|goal|passionate|skilled|experienced|dedicated|motivated)\b",  # Personal statements
            r"\b(?:professional|career|develop|growth|opportunity|challenge)\b",  # Career focus
            r"\b(?:team player|detail-oriented|problem solver|self-motivated|fast learner)\b"  # Personal qualities
        ],
        "CERTIFICATIONS": [
            r"\b(?:certified|certificate|credential|qualified|authorized|accredited)\b",  # Certification terms
            r"\b(?:issued|awarded|received|earned|completed|achieved)\b",  # Acquisition terms
            r"\b(?:course|training|program|workshop)\b"  # Educational programs
        ],
        "LANGUAGES": [
            r"\b(?:fluent|native|proficient|intermediate|beginner|basic|advanced)\b",  # Language proficiency
            r"\b(?:speaking|reading|writing|understanding|verbal|written)\b",  # Language skills
            r"\b(?:english|spanish|french|german|mandarin|hindi|arabic|russian)\b"  # Common languages
        ]
    }
    
    def __init__(self):
        """Initialize the parser."""
        self.section_name_matcher = self._build_section_matcher()
    
    def _build_section_matcher(self):
        """Build a function to match section names to standardized types."""
        # Create a reverse mapping from normalized section names to standard types
        section_matcher = {}
        for name, section_type in self.SECTION_TYPE_MAPPING.items():
            normalized = normalize(name)
            if normalized:  # Skip empty strings
                section_matcher[normalized] = section_type
        
        def match_section_name(text):
            """
            Match a section name to a standardized type.
            Returns (matched_type, confidence_score) tuple.
            """
            if not text:
                return None, 0.0
            
            normalized = normalize(text)
            
            # Direct match
            if normalized in section_matcher:
                return section_matcher[normalized], 1.0
            
            # Fuzzy matching - check if section text contains any known section name
            best_match = None
            best_score = 0.0
            
            for known_name, section_type in self.SECTION_TYPE_MAPPING.items():
                # Calculate similarity score
                similarity = calculate_similarity(text, known_name)
                
                # Check if text contains the section name
                contains_score = 0.8 if normalized.find(normalize(known_name)) >= 0 else 0.0
                
                # Use the maximum of similarity and contains scores
                score = max(similarity, contains_score)
                
                if score > best_score:
                    best_score = score
                    best_match = section_type
            
            # Only return matches above a certain threshold
            if best_score >= 0.6:
                return best_match, best_score
            
            return None, 0.0
        
        return match_section_name
    
    def validate_section_content(self, section_type, text):
        """
        Validate if the content matches expected patterns for the section type.
        Returns a confidence score between 0 and 1.
        """
        if not section_type or section_type not in self.SECTION_VALIDATION_PATTERNS:
            return 0.5  # Default middle confidence for unknown section types
        
        # Get the validation patterns for this section type
        patterns = self.SECTION_VALIDATION_PATTERNS.get(section_type, [])
        if not patterns:
            return 0.5
        
        # Count how many patterns match the text
        match_count = 0
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                match_count += 1
        
        # Calculate confidence based on pattern matches
        confidence = min(1.0, match_count / (len(patterns) * 0.7))  # Scale to 1.0 max
        return confidence
    
    def extract_text_with_style(self, pdf_path):
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
                                
                                # Enhanced style detection
                                is_bold = ("bold" in font_name.lower() or 
                                         "heavy" in font_name.lower() or
                                         span.get("flags", 0) & 2 != 0)  # Check bold flag
                                
                                is_italic = ("italic" in font_name.lower() or 
                                           "oblique" in font_name.lower() or
                                           span.get("flags", 0) & 1 != 0)  # Check italic flag
                                
                                is_capital = text.isupper() and len(text) > 2  # Check if text is all uppercase
                                
                                # Check if text is spaced out (like "E X P E R I E N C E")
                                is_spaced = " " in text and all(len(part) == 1 for part in text.split())
                                
                                # Length filters to avoid decorative elements and icons
                                is_too_short = len(text.strip()) < 3
                                
                                # Detect if this might be an icon or decorative element
                                is_icon = (is_too_short and 
                                        not text.isalnum() and 
                                        len(text) == 1)
                                
                                # Check if this might be a section heading based on styling and content
                                section_type, match_confidence = self.section_name_matcher(text)
                                
                                # Calculate a heading likelihood score
                                heading_score = 0.0
                                
                                # Style-based factors
                                if is_bold:
                                    heading_score += 0.3
                                if is_capital:
                                    heading_score += 0.3
                                if is_spaced:
                                    heading_score += 0.2
                                if font_size > 11:
                                    heading_score += 0.2
                                
                                # Content-based factors
                                if section_type:
                                    heading_score += 0.4 * match_confidence
                                
                                # Negative factors
                                if is_icon:
                                    heading_score -= 0.5
                                if is_too_short and not is_icon:
                                    heading_score -= 0.3
                                if len(text) > 30:  # Likely not a heading if too long
                                    heading_score -= 0.3
                                
                                # Normalize score to 0-1 range
                                heading_score = max(0.0, min(1.0, heading_score))
                                
                                # Determine if likely a heading based on score
                                is_likely_heading = heading_score > 0.5
                                
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
                                    "is_italic": is_italic,
                                    "is_capital": is_capital,
                                    "is_spaced": is_spaced,
                                    "is_icon": is_icon,
                                    "heading_score": heading_score,
                                    "is_likely_heading": is_likely_heading,
                                    "section_type": section_type,
                                    "match_confidence": match_confidence
                                })
        
        doc.close()
        return all_elements
    
    def analyze_layout(self, elements, pdf_path):
        """
        Analyze the resume layout to detect columns and structural features.
        Returns the x-coordinate that separates left and right columns.
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
        
        # Improved column detection
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
                # Determine if this is truly a multi-column layout
                left_count = sum(x_hist[b] for b in x_hist if abs(b - left_col_x) < page_width * 0.2)
                right_count = sum(x_hist[b] for b in x_hist if abs(b - right_col_x) < page_width * 0.2)
                
                # If both columns have significant content
                if left_count > len(elements) * 0.2 and right_count > len(elements) * 0.2:
                    # Return the midpoint between the columns as a divider
                    return (left_col_x + right_col_x) / 2
        
        # Default to 80% of page width if we can't clearly detect columns
        # This effectively treats it as a single-column layout
        return page_width * 0.8
    
    def identify_section_headings(self, elements, column_divider):
        """
        Identify section headings based on text styling and patterns.
        Takes into account the multi-column layout.
        """
        heading_candidates = []
        
        # Group elements by vertical position to detect headers that span multiple elements
        y_grouped_elements = defaultdict(list)
        for i, elem in enumerate(elements):
            # Create a key based on page and rounded y-position
            y_key = (elem["page"], round(elem["y0"] / 5) * 5)  # Group elements within 5 points
            y_grouped_elements[y_key].append(i)
        
        # Process each vertical group
        for y_key, elem_indices in y_grouped_elements.items():
            # Skip groups with too many elements (likely not headings)
            if len(elem_indices) > 5:
                continue
            
            # Check if this group contains a heading
            contains_heading = False
            heading_text_parts = []
            max_heading_score = 0.0
            section_type = None
            match_confidence = 0.0
            heading_x0 = float('inf')
            heading_y0 = float('inf')
            heading_x1 = 0
            heading_y1 = 0
            heading_page = 0
            
            for idx in elem_indices:
                elem = elements[idx]
                
                # Skip elements that are clearly not headings
                if elem["heading_score"] < 0.3:
                    continue
                
                # Update heading information
                contains_heading = True
                heading_text_parts.append(elem["text"])
                
                # Track the bounding box of the entire heading
                heading_x0 = min(heading_x0, elem["x0"])
                heading_y0 = min(heading_y0, elem["y0"])
                heading_x1 = max(heading_x1, elem["x1"])
                heading_y1 = max(heading_y1, elem["y1"])
                heading_page = elem["page"]
                
                # Track the maximum heading score in this group
                if elem["heading_score"] > max_heading_score:
                    max_heading_score = elem["heading_score"]
                    section_type = elem["section_type"]
                    match_confidence = elem["match_confidence"]
            
            if contains_heading:
                # Combine parts to form the complete heading text
                heading_text = " ".join(heading_text_parts)
                
                # If section type wasn't identified from individual parts,
                # try to identify from the complete heading text
                if not section_type:
                    section_type, match_confidence = self.section_name_matcher(heading_text)
                
                # Calculate final confidence score
                confidence = max_heading_score * 0.7 + match_confidence * 0.3
                
                # Determine which column this heading belongs to
                column = "left" if heading_x0 < column_divider else "right"
                
                heading_candidates.append({
                    "text": heading_text,
                    "page": heading_page,
                    "x0": heading_x0,
                    "y0": heading_y0,
                    "x1": heading_x1,
                    "y1": heading_y1,
                    "type": section_type,
                    "column": column,
                    "confidence": confidence,
                    "elem_indices": elem_indices
                })
        
        # Filter and post-process heading candidates
        headings = self.filter_heading_candidates(heading_candidates, elements)
        
        # Detect hierarchical structure (main sections vs subsections)
        headings = self.detect_heading_hierarchy(headings, elements)
        
        return headings
    
    def filter_heading_candidates(self, candidates, elements):
        """
        Filter and clean up heading candidates to remove duplicates and false positives.
        """
        # Sort candidates by confidence, page, and y-position
        sorted_candidates = sorted(candidates, 
                                  key=lambda h: (-h["confidence"], h["page"], h["y0"]))
        
        # Group candidates by section type to detect duplicates
        type_groups = defaultdict(list)
        for heading in sorted_candidates:
            if heading["type"]:
                type_groups[heading["type"]].append(heading)
        
        # Filter out low-confidence duplicates
        filtered_headings = []
        processed_indices = set()
        
        for heading in sorted_candidates:
            # Skip headings with very low confidence
            if heading["confidence"] < 0.4:
                continue
            
            # Skip if we've already processed the elements in this heading
            if any(idx in processed_indices for idx in heading["elem_indices"]):
                continue
            
            # Check proximity to other headings of the same type
            is_duplicate = False
            if heading["type"]:
                same_type_headings = [h for h in type_groups[heading["type"]] 
                                    if h != heading and h["page"] == heading["page"]]
                
                for other in same_type_headings:
                    # Check if they're very close to each other (likely duplicates)
                    y_distance = abs(heading["y0"] - other["y0"])
                    if y_distance < 50:  # Arbitrary threshold for closeness
                        # Keep the one with higher confidence
                        if other["confidence"] > heading["confidence"]:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered_headings.append(heading)
                processed_indices.update(heading["elem_indices"])
        
        return filtered_headings
    
    def detect_heading_hierarchy(self, headings, elements):
        """
        Detect hierarchical structure in headings (main sections vs subsections).
        Uses a simplified approach without DBSCAN clustering to avoid dependency issues.
        """
        if not headings:
            return []
        
        # Analyze font sizes to identify levels
        font_sizes = []
        for h in headings:
            max_font_size = max([elements[idx]["font_size"] for idx in h["elem_indices"]])
            font_sizes.append(max_font_size)
        
        # Simple approach: use statistics to determine levels
        if len(font_sizes) >= 2:
            mean_size = sum(font_sizes) / len(font_sizes)
            
            # Calculate standard deviation manually to avoid numpy dependency
            variance = sum((x - mean_size) ** 2 for x in font_sizes) / len(font_sizes)
            std_size = variance ** 0.5  # Square root of variance
            
            # Find potential heading levels using statistical thresholds
            for i, heading in enumerate(headings):
                max_font_size = max([elements[idx]["font_size"] for idx in heading["elem_indices"]])
                
                # Main sections have above-average font size or are very bold/capitalized
                if max_font_size > mean_size + 0.3 * std_size:
                    headings[i]["level"] = 0  # Main section
                else:
                    # Check other criteria like boldness and capitalization
                    is_very_bold = all(elements[idx]["is_bold"] for idx in heading["elem_indices"])
                    is_all_caps = all(elements[idx]["is_capital"] for idx in heading["elem_indices"])
                    
                    if is_very_bold and is_all_caps and max_font_size >= mean_size:
                        headings[i]["level"] = 0  # Main section based on styling
                    else:
                        headings[i]["level"] = 1  # Subsection
        else:
            # If we only have one or two headings, assume they're all main sections
            for i in range(len(headings)):
                headings[i]["level"] = 0
        
        return headings
    
    def extract_sections(self, pdf_path, headings, elements, column_divider):
        """
        Extract sections from the PDF, handling the multi-column layout
        and hierarchical structure.
        """
        if not headings:
            return {}
        
        # Create ResumeSection objects for each heading
        section_objects = {}
        for heading in headings:
            section = ResumeSection(
                heading=heading["text"],
                section_type=heading["type"] if heading["type"] else "UNKNOWN",
                level=heading.get("level", 0)
            )
            section.set_coordinates({
                "x0": heading["x0"],
                "y0": heading["y0"],
                "x1": heading["x1"],
                "y1": heading["y1"],
                "page": heading["page"]
            })
            section.set_column(heading["column"])
            section.set_confidence(heading["confidence"])
            
            section_id = f"{heading['type'] or 'UNKNOWN'}_{len(section_objects)}"
            section_objects[section_id] = section
        
        # Build section hierarchy
        section_hierarchy = self.build_section_hierarchy(section_objects)
        
        # Group elements by page
        elements_by_page = defaultdict(list)
        for i, elem in enumerate(elements):
            # Skip elements that are part of headings
            is_heading_element = False
            for heading in headings:
                if i in heading["elem_indices"]:
                    is_heading_element = True
                    break
            
            if not is_heading_element:
                elements_by_page[elem["page"]].append(elem)
        
        # Assign elements to sections
        self.assign_elements_to_sections(section_objects, elements_by_page, column_divider)
        
        # Validate section content
        for section_id, section in section_objects.items():
            content_text = section.get_content_text()
            content_validation_score = self.validate_section_content(section.section_type, content_text)
            
            # Update confidence based on content validation
            updated_confidence = section.confidence * 0.7 + content_validation_score * 0.3
            section.set_confidence(updated_confidence)
        
        return section_objects
    
    def build_section_hierarchy(self, section_objects):
        """Build the hierarchical structure of sections based on levels and positions."""
        # Sort sections by page, level, and y-position
        sorted_sections = sorted(
            section_objects.items(),
            key=lambda x: (
                x[1].coordinates["page"],
                x[1].level,
                x[1].coordinates["y0"]
            )
        )
        
        # Process sections by column
        left_sections = [(id, sec) for id, sec in sorted_sections if sec.column == "left"]
        right_sections = [(id, sec) for id, sec in sorted_sections if sec.column == "right"]
        
        # Process each column to build hierarchy
        root_sections = {}
        
        for column_sections in [left_sections, right_sections]:
            current_parent = None
            
            for section_id, section in column_sections:
                if section.level == 0:  # Main section
                    root_sections[section_id] = section
                    current_parent = section
                elif current_parent:  # Subsection
                    current_parent.add_child(section)
        
        return root_sections
    
    def assign_elements_to_sections(self, section_objects, elements_by_page, column_divider):
        """Assign content elements to their respective sections."""
        # Sort sections by page and y-position for processing order
        sorted_sections = sorted(
            section_objects.items(),
            key=lambda x: (
                x[1].coordinates["page"],
                x[1].coordinates["y0"]
            )
        )
        
        # Process each page separately
        for page_num, page_elements in elements_by_page.items():
            # Get sections on this page, separated by column
            left_sections = [(id, sec) for id, sec in sorted_sections 
                            if sec.coordinates["page"] == page_num and sec.column == "left"]
            right_sections = [(id, sec) for id, sec in sorted_sections 
                             if sec.coordinates["page"] == page_num and sec.column == "right"]
            
            # Process left column elements
            left_elements = [elem for elem in page_elements if elem["x0"] < column_divider]
            self._assign_column_elements(left_elements, left_sections)
            
            # Process right column elements
            right_elements = [elem for elem in page_elements if elem["x0"] >= column_divider]
            self._assign_column_elements(right_elements, right_sections)
    
    def _assign_column_elements(self, elements, sections):
        """Assign elements to sections within a column."""
        # Sort elements by y-position
        elements = sorted(elements, key=lambda e: e["y0"])
        
        # Sort sections by y-position
        sections = sorted(sections, key=lambda s: s[1].coordinates["y0"])
        
        if not sections:
            return
        
        # For each element, find the appropriate section
        for elem in elements:
            # Find the section this element belongs to
            assigned_section = None
            
            for i, (section_id, section) in enumerate(sections):
                # If element is below this section heading
                if elem["y0"] >= section.coordinates["y1"]:
                    # Check if there's a next section
                    if i < len(sections) - 1:
                        next_section = sections[i+1][1]
                        # If element is above the next section heading
                        if elem["y0"] < next_section.coordinates["y0"]:
                            assigned_section = section
                            break
                    else:
                        # This is the last section in the column
                        assigned_section = section
                        break
            
            # If no section was found, assign to the first section if element is above it
            if not assigned_section and sections and elem["y0"] < sections[0][1].coordinates["y0"]:
                assigned_section = sections[0][1]
            
            # Add element to the assigned section
            if assigned_section:
                assigned_section.add_element(elem)
    
    def extract_section_text(self, section_objects):
        """Convert the section objects into a structured text representation."""
        section_texts = {}
        
        # Process each section
        for section_id, section in section_objects.items():
            # Skip subsections (they'll be included with their parent sections)
            if section.parent:
                continue
            
            # Get the standardized section type
            section_type = section.section_type
            
            # Create section text
            text_lines = [f"--- {section.heading} ({section.column.upper()} COLUMN) ---"]
            
            # Add section content
            content_text = section.get_content_text()
            if content_text:
                text_lines.append(content_text)
            
            # Add subsection content
            for child in section.children:
                text_lines.append(f"\n--- Subsection: {child.heading} ---")
                child_content = child.get_content_text()
                if child_content:
                    text_lines.append(child_content)
            
            # Join all lines with newlines
            section_text = "\n".join(text_lines)
            
            # Use a standardized key for better consistency
            if section_type in section_texts:
                # If we already have this section type, append new content
                section_texts[section_type] += f"\n\n{section_text}"
            else:
                section_texts[section_type] = section_text
        
        return section_texts
    
    def visualize_sections(self, pdf_path, section_objects, column_divider):
        """Create a visualized PDF with highlighted sections, showing the column structure."""
        # Colors for different section types (RGB, values from 0-1)
        colors = {
            "EDUCATION": (0, 0, 1),      # Blue
            "EXPERIENCE": (0, 0.6, 0),   # Green
            "SKILLS": (1, 0.5, 0),       # Orange
            "PROJECTS": (0.5, 0, 0.5),   # Purple
            "CONTACT": (0, 0.6, 0.6),    # Teal
            "PROFILE": (0.6, 0, 0),      # Dark Red
            "CERTIFICATIONS": (0.7, 0.7, 0), # Olive
            "AWARDS": (0.8, 0.4, 0),     # Brown
            "ACHIEVEMENTS": (0, 0.4, 0.8), # Light Blue
            "LANGUAGES": (0.5, 0.5, 0),  # Olive Green
            "DECLARATION": (0.4, 0.4, 0.4), # Gray
            "EXTRACURRICULAR": (0.2, 0.6, 0.4), # Aquamarine
            "REFERENCES": (0.6, 0.3, 0),  # Brown
            "PUBLICATIONS": (0.1, 0.1, 0.8), # Navy
            "UNKNOWN": (0.3, 0.3, 0.3)    # Dark Gray
        }
        
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
            
            # Draw section headings and content
            for section_id, section in section_objects.items():
                if section.coordinates["page"] != page_num:
                    continue
                
                section_type = section.section_type
                color = colors.get(section_type, colors["UNKNOWN"])
                
                # Draw heading with confidence score
                label_text = f"{section.heading} ({section.confidence:.2f})"
                label_point = fitz.Point(section.coordinates["x0"], section.coordinates["y0"] - 5)
                
                # Adjust label color based on confidence
                label_color = color
                if section.confidence < 0.6:
                    # Blend with gray for low confidence
                    blend_factor = max(0, section.confidence) / 0.6
                    label_color = (
                        color[0] * blend_factor + 0.5 * (1 - blend_factor),
                        color[1] * blend_factor + 0.5 * (1 - blend_factor),
                        color[2] * blend_factor + 0.5 * (1 - blend_factor)
                    )
                
                new_page.insert_text(label_point, label_text, fontsize=8, color=label_color)
                
                # Draw a rectangle around the heading
                heading_rect = fitz.Rect(
                    section.coordinates["x0"] - 2, 
                    section.coordinates["y0"] - 2, 
                    section.coordinates["x1"] + 2, 
                    section.coordinates["y1"] + 2
                )
                new_page.draw_rect(heading_rect, color=color, width=1)
                
                # Draw section content area
                elements = section.elements
                if elements:
                    # Find the bounds of elements in this section on this page
                    x0 = min(e["x0"] for e in elements)
                    y0 = min(e["y0"] for e in elements)
                    x1 = max(e["x1"] for e in elements)
                    y1 = max(e["y1"] for e in elements)
                    
                    # Draw a rectangle around the section content
                    content_rect = fitz.Rect(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
                    
                    # Use different dash pattern for subsections
                    dash_pattern = "[2 2]" if section.level > 0 else None
                    
                    new_page.draw_rect(content_rect, color=color, width=1, dashes=dash_pattern)
                    
                    # Draw connecting line from heading to content
                    if y0 > section.coordinates["y1"]:
                        mid_x = (section.coordinates["x0"] + section.coordinates["x1"]) / 2
                        new_page.draw_line(
                            (mid_x, section.coordinates["y1"]),
                            (mid_x, y0 - 5),
                            color=color,
                            width=0.5,
                            dashes="[1 1]"
                        )
        
        # Save the output PDF
        out_path = os.path.splitext(pdf_path)[0] + "_enhanced_analysis.pdf"
        output.save(out_path)
        print(f"Enhanced annotated PDF saved to: {out_path}")
        
        # Clean up
        output.close()
        doc.close()
        
        return out_path
    
    def parse_resume(self, pdf_path):
        """Parse a resume PDF file with improved section detection."""
        print(f"Processing resume: {pdf_path}")
        
        # Extract text elements with style information
        elements = self.extract_text_with_style(pdf_path)
        print(f"Extracted {len(elements)} text elements")
        
        # Analyze the layout to detect columns
        column_divider = self.analyze_layout(elements, pdf_path)
        print(f"Detected column divider at x-coordinate: {column_divider}")
        
        # Identify section headings with improved detection
        headings = self.identify_section_headings(elements, column_divider)
        print(f"Identified {len(headings)} section headings:")
        
        # Count headings in each column and by level
        left_headings = [h for h in headings if h["column"] == "left"]
        right_headings = [h for h in headings if h["column"] == "right"]
        main_headings = [h for h in headings if h.get("level", 0) == 0]
        sub_headings = [h for h in headings if h.get("level", 0) > 0]
        
        print(f"  - Left column: {len(left_headings)} headings")
        print(f"  - Right column: {len(right_headings)} headings")
        print(f"  - Main sections: {len(main_headings)}")
        print(f"  - Subsections: {len(sub_headings)}")
        
        # Print detected headings with confidence
        for heading in sorted(headings, key=lambda h: (h["page"], h["y0"])):
            level_indicator = "  ↳" if heading.get("level", 0) > 0 else "  -"
            confidence_indicator = "✓" if heading["confidence"] > 0.7 else "?"
            print(f"{level_indicator} {heading['text']} ({heading['type'] or 'UNKNOWN'}, "
                 f"{heading['column']} column, confidence: {heading['confidence']:.2f} {confidence_indicator})")
        
        # Extract sections with hierarchical structure
        section_objects = self.extract_sections(pdf_path, headings, elements, column_divider)
        print(f"Extracted {len(section_objects)} sections")
        
        # Extract text from each section
        section_texts = self.extract_section_text(section_objects)
        
        # Visualize the sections with hierarchy
        out_path = self.visualize_sections(pdf_path, section_objects, column_divider)
        
        # Print the extracted text
        print("\nExtracted Section Content:")
        print("=========================")
        for section_type, text in section_texts.items():
            print(f"\n{section_type}:")
            print("-" * len(section_type))
            print(text[:500] + "..." if len(text) > 500 else text)
        
        return out_path, section_texts


def parse_resume(pdf_path):
    """Parse a resume PDF file with the enhanced parser."""
    parser = ResumeParser()
    return parser.parse_resume(pdf_path)


# # Example usage
# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1:
#         pdf_path = sys.argv[1]
#     else:
#         pdf_path = input("Enter the path to the resume PDF: ")
    
#     parse_resume(pdf_path)


def process_resume_folder(folder_path, output_folder=None):
    """
    Process all PDF files in a folder to identify and extract resume sections.
    
    Args:
        folder_path (str): Path to the folder containing resume PDFs
        output_folder (str, optional): Path to save output files. If None, uses folder_path
    
    Returns:
        list: List of processed file paths and their extracted sections
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return []
    
    if output_folder is None:
        output_folder = folder_path + "_output"
    else:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
    
    # Get all PDF files in the folder
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files in {folder_path}")
    # Process each PDF file
    results = []
    for i, pdf_path in enumerate(pdf_files):
        print(f"\nProcessing file {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
        
        start_time = time.time()
        out_path, sections = parse_resume(pdf_path)
        elapsed_time = time.time() - start_time
        
        if out_path:

            
            results.append({
                'pdf_path': pdf_path,
                'out_path': out_path,
                'sections': list(sections.keys()),
                'processing_time': elapsed_time
            })
    
    # Create overall summary
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if os.path.isdir(input_path):
            # Process all PDFs in the folder
            output_folder = sys.argv[2] if len(sys.argv) > 2 else None
            process_resume_folder(input_path, output_folder)
        else:
            # Process a single PDF
            parse_resume(input_path)
    else:
        print("Usage:")
        print("  For a single resume: python resume_parser.py path/to/resume.pdf")
        print("  For a folder of resumes: python resume_parser.py path/to/resume/folder [output_folder]")