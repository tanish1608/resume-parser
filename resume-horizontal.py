import pdfplumber
import fitz  # PyMuPDF
import os
import re
from collections import defaultdict

def normalize(text):
    """Normalize text by removing whitespace and converting to lowercase."""
    return re.sub(r'\s+', '', text).lower()

def is_heading_text(text):
    """Check if the text is a common resume section heading."""
    heading_keywords = [
        "education", "skills", "projects", "certifications", "contact",
        "languages", "objective", "experience", "workexperience", 
        "accomplishments", "awards", "careergoals", "careerobjective",
        "declaration", "decleration", "profile", "summary", "achievements"
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
        "EDUCATION": ["education"],
        "EXPERIENCE": ["experience", "workexperience", "career"],
        "SKILLS": ["skills", "technicalskills", "softskill"],
        "PROJECTS": ["projects", "portfolio"],
        "CONTACT": ["contact"],
        "SUMMARY": ["summary", "profile", "objective", "careerobjective"],
        "CERTIFICATIONS": ["certifications", "certification", "credentials"],
        "AWARDS": ["awards"],
        "ACHIEVEMENTS": ["achievements", "accomplishments"],
        "LANGUAGES": ["languages"],
        "DECLARATION": ["declaration", "decleration"]
    }
    
    for section_name, keywords in section_types.items():
        if any(kw in clean for kw in keywords):
            return section_name
    
    return None

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
                            
                            # Detect if this might be a heading based on styling
                            is_likely_heading = (
                                (is_bold and is_heading_text(text)) or 
                                (is_capital and is_heading_text(text)) or
                                (font_size > 11 and is_heading_text(text)) or
                                (is_spaced and len(text) > 5)  # Spaced out text like "E D U C A T I O N"
                            )
                            
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
                                "is_likely_heading": is_likely_heading
                            })
    
    doc.close()
    return all_elements

def analyze_layout(elements, pdf_path):
    """
    Analyze the resume layout to detect columns.
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
        if len(text) < 3:
            continue
        
        # Different ways to identify headings
        is_heading = False
        heading_type = None
        
        # Check for spaced headings like "E D U C A T I O N"
        if elem["is_spaced"] and len(text) > 5:
            # Convert "E D U C A T I O N" to "EDUCATION" for checking
            condensed = text.replace(" ", "")
            heading_type = get_heading_type(condensed)
            if heading_type:
                is_heading = True
        
        # Check for all caps headings like "EDUCATION"
        elif elem["is_capital"] and is_heading_text(text):
            heading_type = get_heading_type(text)
            if heading_type:
                is_heading = True
        
        # Check for bold headings
        elif elem["is_bold"] and is_heading_text(text):
            heading_type = get_heading_type(text)
            if heading_type:
                is_heading = True
        
        # Check for large font headings
        elif elem["font_size"] > 11 and is_heading_text(text):
            heading_type = get_heading_type(text)
            if heading_type:
                is_heading = True
        
        if is_heading and heading_type:
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
        return sections
    
    # Separate headings by column
    left_headings = [h for h in headings if h["column"] == "left"]
    right_headings = [h for h in headings if h["column"] == "right"]
    
    # Sort headings in each column by page and vertical position
    left_headings = sorted(left_headings, key=lambda h: (h["page"], h["y0"]))
    right_headings = sorted(right_headings, key=lambda h: (h["page"], h["y0"]))
    
    # Group elements by page
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

def visualize_sections(pdf_path, sections, headings, column_divider):
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
    
    # Save the output PDF
    out_path = os.path.splitext(pdf_path)[0] + "_columns_highlighted.pdf"
    output.save(out_path)
    print(f"Annotated PDF saved to: {out_path}")
    
    # Clean up
    output.close()
    doc.close()
    
    return out_path

def parse_resume(pdf_path):
    """Parse a resume PDF file, extract sections from multi-column layout, and visualize them."""
    print(f"Processing resume: {pdf_path}")
    
    # Extract text elements with style information
    elements = extract_text_with_style(pdf_path)
    print(f"Extracted {len(elements)} text elements")
    
    # Analyze the layout to detect columns
    column_divider = analyze_layout(elements, pdf_path)
    print(f"Detected column divider at x-coordinate: {column_divider}")
    
    # Identify section headings, taking into account the column structure
    headings = identify_section_headings(elements, column_divider)
    print(f"Identified {len(headings)} section headings:")
    
    # Count headings in each column
    left_headings = [h for h in headings if h["column"] == "left"]
    right_headings = [h for h in headings if h["column"] == "right"]
    print(f"  - Left column: {len(left_headings)} headings")
    print(f"  - Right column: {len(right_headings)} headings")
    
    for heading in headings:
        print(f"  - {heading['text']} ({heading['type']}, {heading['column']} column)")
    
    # Extract sections, handling the multi-column layout
    sections = extract_sections(pdf_path, headings, elements, column_divider)
    print(f"Extracted {len(sections)} sections")
    
    # Extract text from each section
    section_texts = extract_section_text(sections)
    
    # Visualize the sections and column structure
    out_path = visualize_sections(pdf_path, sections, headings, column_divider)
    
    # Print the extracted text
    print("\nExtracted Section Content:")
    print("=========================")
    for section_type, text in section_texts.items():
        print(f"\n{section_type}:")
        print("-" * len(section_type))
        print(text[:500] + "..." if len(text) > 500 else text)
    
    return out_path, section_texts

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter the path to the resume PDF: ")
    
    parse_resume(pdf_path)