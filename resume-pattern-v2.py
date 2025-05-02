import pdfplumber
import fitz  # PyMuPDF
import os
import re
from collections import defaultdict
import string

class ResumeSection:
    """Class to represent a section in a resume."""
    def __init__(self, heading, section_type, level=0):
        self.heading = heading  # Original heading text
        self.section_type = section_type  # Standardized section type
        self.level = level  # 0 for main section, 1+ for subsections
        self.elements = []  # Content elements belonging to this section
        self.coordinates = {}  # Position information (x0, y0, x1, y1, page)
        self.column = None  # Left or right column
    
    def add_element(self, element):
        """Add a content element to this section."""
        self.elements.append(element)
    
    def set_coordinates(self, coords):
        """Set the position coordinates for this section."""
        self.coordinates = coords
    
    def set_column(self, column):
        """Set the column (left/right) for this section."""
        self.column = column
    
    def get_content_text(self):
        """Get the text content of this section."""
        # Sort elements by page number and y-position
        sorted_elements = sorted(self.elements, key=lambda e: (e.get("page", 0), e.get("y0", 0)))
        return "\n".join([elem.get("text", "") for elem in sorted_elements])

class ResumeParser:
    """Resume parser that identifies and extracts sections."""
    
    # Standard section names
    STANDARD_SECTION_NAMES = [
        "EDUCATION", "ACADEMIC BACKGROUND", "QUALIFICATIONS", 
        "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT",
        "SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES", "KEY SKILLS",
        "PROJECTS", "PROJECT EXPERIENCE", "KEY PROJECTS", 
        "CERTIFICATIONS", "CERTIFICATES", "COURSES",
        "ACHIEVEMENTS", "AWARDS", "HONORS", "ACCOMPLISHMENTS",
        "PUBLICATIONS", "RESEARCH", "PAPERS",
        "VOLUNTEER", "VOLUNTEERING", "COMMUNITY SERVICE",
        "LANGUAGES", "LANGUAGE SKILLS", 
        "INTERESTS", "HOBBIES", "ACTIVITIES",
        "PROFILE", "SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "ABOUT",
        "REFERENCES", "CONTACT", "CONTACT INFORMATION",
        "LEADERSHIP", "LEADERSHIP EXPERIENCE", 
        "EXTRACURRICULAR", "EXTRACURRICULAR ACTIVITIES",
        "INTERNSHIPS", "INTERNSHIP EXPERIENCE", "TRAINING",
        "STRENGTHS", "PERSONAL STRENGTHS", "KEY STRENGTHS",
    ]
    
    # Non-section patterns
    NON_SECTION_PATTERNS = [
        r"^[A-Z][a-z]+\s+[A-Z][a-z]+$",  # Name pattern
        r"^(?:https?://|www\.)",  # URLs
        r"^[\w.]+@[\w.]+\.\w+$",  # Email
        r"^\+?[\d\s-]{10,}$",     # Phone number
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",  # Date pattern
    ]
    
    def __init__(self):
        """Initialize the parser."""
        self.avg_font_size = 0
    
    def is_section_heading(self, text):
        """Check if text matches a standard section name."""
        if not text or len(text.strip()) < 3:
            return False, None
            
        # Check if this is a non-section pattern
        for pattern in self.NON_SECTION_PATTERNS:
            if re.search(pattern, text.strip()):
                return False, None
                
        # Check if the text contains any standard section name
        for section_name in self.STANDARD_SECTION_NAMES:
            if section_name.lower() in text.lower():
                return True, section_name
                
        return False, None
    
    def extract_text_with_style(self, pdf_path):
        """Extract text with style information using PyMuPDF."""
        doc = fitz.open(pdf_path)
        all_elements = []
        all_font_sizes = []
        
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
                                all_font_sizes.append(font_size)
                                
                                # Detect text styling
                                is_bold = ("bold" in font_name.lower() or 
                                         "heavy" in font_name.lower() or
                                         span.get("flags", 0) & 2 != 0)
                                
                                is_capital = text.isupper() and len(text) > 2
                                
                                all_elements.append({
                                    "page": page_num,
                                    "text": text,
                                    "x0": span["bbox"][0],
                                    "y0": span["bbox"][1],
                                    "x1": span["bbox"][2],
                                    "y1": span["bbox"][3],
                                    "font_size": font_size,
                                    "is_bold": is_bold,
                                    "is_capital": is_capital,
                                })
        
        doc.close()
        
        # Calculate average font size
        if all_font_sizes:
            self.avg_font_size = sum(all_font_sizes) / len(all_font_sizes)
        
        return all_elements
    
    def detect_columns(self, elements, pdf_path):
        """Detect if the resume has a multi-column layout."""
        with pdfplumber.open(pdf_path) as pdf:
            page_width = pdf.pages[0].width
        
        # Get all x-coordinates
        x_coords = [elem["x0"] for elem in elements]
        
        # Create histogram of x-coordinates
        x_hist = defaultdict(int)
        for x in x_coords:
            bin_x = round(x / 10) * 10
            x_hist[bin_x] += 1
        
        # Sort bins by frequency
        sorted_bins = sorted(x_hist.items(), key=lambda x: x[1], reverse=True)
        
        # Check for multi-column layout
        if len(sorted_bins) >= 2:
            left_col_x = sorted_bins[0][0]
            
            # Find another common x-position significantly different from the first
            for bin_x, count in sorted_bins:
                if abs(bin_x - left_col_x) > page_width * 0.2:
                    # Return the midpoint as column divider
                    return (left_col_x + bin_x) / 2
        
        # Default to single column layout
        return page_width * 0.8
    
    def identify_section_headings(self, elements, column_divider):
        """Identify section headings in the resume."""
        heading_candidates = []
        
        # Group elements by vertical position
        y_grouped_elements = defaultdict(list)
        for i, elem in enumerate(elements):
            y_key = (elem["page"], round(elem["y0"] / 5) * 5)
            y_grouped_elements[y_key].append(i)
        
        # Process each group
        for y_key, elem_indices in y_grouped_elements.items():
            # Skip large groups (likely not headings)
            if len(elem_indices) > 5:
                continue
            
            # Analyze this group
            texts = []
            is_bold = True
            max_font_size = 0
            is_capital = True
            left_x = float('inf')
            top_y = float('inf')
            right_x = 0
            bottom_y = 0
            page = None
            
            for idx in elem_indices:
                elem = elements[idx]
                texts.append(elem["text"])
                
                # Check styling
                is_bold = is_bold and elem["is_bold"]
                max_font_size = max(max_font_size, elem["font_size"])
                is_capital = is_capital and elem["is_capital"]
                
                # Track position
                left_x = min(left_x, elem["x0"])
                top_y = min(top_y, elem["y0"])
                right_x = max(right_x, elem["x1"])
                bottom_y = max(bottom_y, elem["y1"])
                page = elem["page"]
            
            # Combine texts
            full_text = " ".join(texts)
            
            # Check if this looks like a section heading
            is_section, section_type = self.is_section_heading(full_text)
            
            # Calculate heading score
            is_heading = False
            
            # Standard section name is a strong indicator
            if is_section:
                is_heading = True
            
            # Styling indicators
            if not is_heading and is_bold and max_font_size > self.avg_font_size * 1.1:
                is_heading = True
                
            if not is_heading and is_capital and max_font_size > self.avg_font_size:
                is_heading = True
            
            # Only include high-confidence headings
            if is_heading:
                # Determine column
                column = "left" if left_x < column_divider else "right"
                
                heading_candidates.append({
                    "text": full_text,
                    "page": page,
                    "x0": left_x,
                    "y0": top_y,
                    "x1": right_x,
                    "y1": bottom_y,
                    "type": section_type,
                    "column": column,
                    "elem_indices": elem_indices,
                    "font_size": max_font_size,
                })
        
        # Remove duplicates and filter headings
        return self.filter_headings(heading_candidates)
    
    def filter_headings(self, candidates):
        """Filter heading candidates to remove duplicates and false positives."""
        # Sort by page and y-position
        sorted_candidates = sorted(candidates, key=lambda h: (h["page"], h["y0"]))
        
        filtered_headings = []
        processed_indices = set()
        
        for i, heading in enumerate(sorted_candidates):
            # Skip if already processed
            if any(idx in processed_indices for idx in heading["elem_indices"]):
                continue
            
            # Check for duplicates
            is_duplicate = False
            for j, existing in enumerate(filtered_headings):
                if (existing["type"] == heading["type"] and 
                    existing["type"] is not None and
                    existing["page"] == heading["page"] and
                    abs(existing["y0"] - heading["y0"]) < 50):
                    
                    # Keep the one with larger font size
                    if existing["font_size"] < heading["font_size"]:
                        filtered_headings[j] = heading
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_headings.append(heading)
                processed_indices.update(heading["elem_indices"])
        
        # Identify heading hierarchy
        for i, heading in enumerate(filtered_headings):
            # Main sections usually have larger font size
            if heading["font_size"] >= self.avg_font_size * 1.2:
                filtered_headings[i]["level"] = 0  # Main section
            else:
                filtered_headings[i]["level"] = 1  # Subsection
        
        return filtered_headings
    
    def extract_sections(self, headings, elements, column_divider):
        """Extract sections from the resume."""
        if not headings:
            return {}
        
        # Create section objects
        section_objects = {}
        for heading in headings:
            section_type = heading["type"] if heading["type"] else "OTHER"
            section = ResumeSection(
                heading=heading["text"],
                section_type=section_type,
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
            
            section_id = f"{section_type}_{len(section_objects)}"
            section_objects[section_id] = section
        
        # Group elements by page
        elements_by_page = defaultdict(list)
        for elem in elements:
            # Skip elements that are part of headings
            is_heading_element = False
            for heading in headings:
                if any(idx for idx in heading["elem_indices"] if elements[idx] is elem):
                    is_heading_element = True
                    break
            
            if not is_heading_element:
                elements_by_page[elem["page"]].append(elem)
        
        # Assign elements to sections
        sorted_sections = sorted(
            section_objects.items(),
            key=lambda x: (
                x[1].coordinates["page"],
                x[1].coordinates["y0"]
            )
        )
        
        # Process each page
        for page_num, page_elements in elements_by_page.items():
            # Get sections on this page by column
            left_sections = [(id, sec) for id, sec in sorted_sections 
                           if sec.coordinates["page"] == page_num and sec.column == "left"]
            right_sections = [(id, sec) for id, sec in sorted_sections 
                            if sec.coordinates["page"] == page_num and sec.column == "right"]
            
            # Process columns
            left_elements = [elem for elem in page_elements if elem["x0"] < column_divider]
            right_elements = [elem for elem in page_elements if elem["x0"] >= column_divider]
            
            self._assign_elements_to_sections(left_elements, left_sections)
            self._assign_elements_to_sections(right_elements, right_sections)
        
        return section_objects
    
    def _assign_elements_to_sections(self, elements, sections):
        """Assign elements to their respective sections."""
        if not sections:
            return
            
        # Sort by y-position
        elements = sorted(elements, key=lambda e: e["y0"])
        sections = sorted(sections, key=lambda s: s[1].coordinates["y0"])
        
        for elem in elements:
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
                        # This is the last section
                        assigned_section = section
                        break
            
            # Add element to section
            if assigned_section:
                assigned_section.add_element(elem)
    
    def extract_section_text(self, section_objects):
        """Extract text content from each section."""
        section_texts = {}
        
        for section_id, section in section_objects.items():
            section_type = section.section_type
            
            # Get content text
            content_text = section.get_content_text()
            
            # Use standardized section type as key
            if section_type in section_texts:
                section_texts[section_type] += f"\n\n{content_text}"
            else:
                section_texts[section_type] = content_text
        
        return section_texts
    
    def visualize_sections(self, pdf_path, section_objects, column_divider):
        """Create a visualized PDF with highlighted sections."""
        # Colors for different section types
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
            "OTHER": (0.3, 0.3, 0.3)     # Gray (default)
        }
        
        doc = fitz.open(pdf_path)
        output = fitz.open()
        
        # Create new PDF with highlighted sections
        for page_num in range(len(doc)):
            # Create new page
            orig_page = doc[page_num]
            new_page = output.new_page(width=orig_page.rect.width, height=orig_page.rect.height)
            new_page.show_pdf_page(orig_page.rect, doc, page_num)
            
            # Draw column divider
            new_page.draw_line(
                (column_divider, 0), 
                (column_divider, orig_page.rect.height),
                color=(0.5, 0.5, 0.5), 
                width=0.5, 
                dashes="[2 2]"
            )
            
            # Draw sections
            for section_id, section in section_objects.items():
                if section.coordinates["page"] != page_num:
                    continue
                
                # Get color for section type
                section_type = section.section_type
                color = colors.get(section_type, colors["OTHER"])
                
                # Draw heading
                label_text = section.heading
                label_point = fitz.Point(section.coordinates["x0"], section.coordinates["y0"] - 5)
                new_page.insert_text(label_point, label_text, fontsize=8, color=color)
                
                # Draw rectangle around heading
                heading_rect = fitz.Rect(
                    section.coordinates["x0"] - 2,
                    section.coordinates["y0"] - 2,
                    section.coordinates["x1"] + 2,
                    section.coordinates["y1"] + 2
                )
                new_page.draw_rect(heading_rect, color=color, width=1)
                
                # Draw section content
                elements = section.elements
                if elements:
                    # Find bounds of elements in this section
                    x0 = min(e["x0"] for e in elements)
                    y0 = min(e["y0"] for e in elements)
                    x1 = max(e["x1"] for e in elements)
                    y1 = max(e["y1"] for e in elements)
                    
                    # Draw rectangle around content
                    content_rect = fitz.Rect(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
                    new_page.draw_rect(content_rect, color=color, width=1)
                    
                    # Draw connecting line
                    if y0 > section.coordinates["y1"]:
                        mid_x = (section.coordinates["x0"] + section.coordinates["x1"]) / 2
                        new_page.draw_line(
                            (mid_x, section.coordinates["y1"]),
                            (mid_x, y0 - 5),
                            color=color,
                            width=0.5,
                            dashes="[1 1]"
                        )
        
        # Save output
        out_path = os.path.splitext(pdf_path)[0] + "_sections.pdf"
        output.save(out_path)
        
        # Clean up
        output.close()
        doc.close()
        
        return out_path
    
    def parse_resume(self, pdf_path):
        """Parse a resume PDF file."""
        print(f"Processing resume: {pdf_path}")
        
        # Extract text elements
        elements = self.extract_text_with_style(pdf_path)
        print(f"Extracted {len(elements)} text elements")
        
        # Detect columns
        column_divider = self.detect_columns(elements, pdf_path)
        
        # Identify section headings
        headings = self.identify_section_headings(elements, column_divider)
        print(f"Identified {len(headings)} section headings:")
        
        # Print detected section types
        section_types = {}
        for heading in headings:
            section_type = heading["type"] if heading["type"] else "OTHER"
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        for section_type, count in section_types.items():
            print(f"  - {section_type}: {count}")
        
        # Extract sections
        section_objects = self.extract_sections(headings, elements, column_divider)
        
        # Extract text from sections
        section_texts = self.extract_section_text(section_objects)
        
        # Visualize sections
        out_path = self.visualize_sections(pdf_path, section_objects, column_divider)
        
        return out_path, section_texts


def parse_resume(pdf_path):
    """Parse a resume PDF file."""
    parser = ResumeParser()
    return parser.parse_resume(pdf_path)


def batch_process_resumes(folder_path, output_folder=None):
    """Process all PDF files in a folder."""
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return []
    
    # Create output folder
    if output_folder is None:
        output_folder = folder_path + "_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each file
    results = []
    for i, pdf_path in enumerate(pdf_files):
        print(f"\nProcessing file {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
        
        try:
            # Process resume
            parser = ResumeParser()
            output_path, sections = parser.parse_resume(pdf_path)
            
            # Save section text
            basename = os.path.splitext(os.path.basename(pdf_path))[0]
            text_path = os.path.join(output_folder, f"{basename}_sections.txt")
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"RESUME SECTIONS: {basename}\n")
                f.write("="*50 + "\n\n")
                
                for section_type, text in sections.items():
                    if section_type != "OTHER":
                        f.write(f"{section_type}:\n")
                        f.write("-" * len(section_type) + "\n")
                        f.write(text + "\n\n")
            
            # Track result
            results.append({
                'pdf_path': pdf_path,
                'output_pdf': output_path,
                'output_text': text_path,
                'sections': list(s for s in sections.keys() if s != "OTHER"),
                'success': True
            })
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            results.append({
                'pdf_path': pdf_path,
                'error': str(e),
                'success': False
            })
    
    # Print summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\nSuccessfully processed {successful} out of {len(pdf_files)} files")
    print(f"Output files saved to: {output_folder}")
    
    return results


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            # Process folder
            output_folder = sys.argv[2] if len(sys.argv) > 2 else None
            batch_process_resumes(path, output_folder)
        else:
            # Process single file
            parse_resume(path)
    else:
        print("Resume Parser - Section Extractor")
        print("Usage:")
        print("  For a single resume: python resume_parser.py path/to/resume.pdf")
        print("  For a folder of resumes: python resume_parser.py path/to/folder [output_folder]")