import pdfplumber
import fitz  # PyMuPDF
import os
import re
from collections import defaultdict
import sys
import time
import logging
import numpy as np
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants for section detection
MIN_HEADING_LENGTH = 2  # Minimum length for heading text
COLUMN_DETECTION_THRESHOLD = 0.2  # Percentage of page width to determine column separation
FONT_SIZE_THRESHOLD_FACTOR = 1.15  # Multiplier for average font size to identify headings
LINE_SPACING_TOLERANCE = 2  # Tolerance for grouping elements into lines (pixels)
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score to consider a detected heading valid

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
        "education", "academic", "academics", "qualification", "qualifications", "educational", "edu", 
        "schooling", "school", "studies", "study", "learning", "degree", "degrees", "university", 
        "universities", "college", "colleges", "educational background", "academic background", "educational qualifications", "academic record", "academic credentials",
        "academic history", "scholastic record", "academic profile", "university education",
        "college education", "school education", "tertiary education", "higher education",
        # International variations
        "bildung", "ausbildung", "educación", "educacion", "formation", "formazione", "opleiding",
        "utbildning", "educação", "uddannelse", "koulutus", "vzdělání", "教育", "教育背景", "学歴",
        "शिक्षा", "التعليم", "giáo dục", "pendidikan", "การศึกษา", "edukacja", "εκπαίδευση"
    ],
    
    "EXPERIENCE": [
        # English variations
        "experience", "experiences", "work", "workexperience", "work-experience", "work/experience", 
        "employment", "career", "job", "jobs", "workhistory", "work history", "professional", "prof", 
        "professionalexperience", "professional background", "employment history", "internship", 
        "internships", "exp", "wrk", "work exp", "prof exp", "occupation", "occupations", "positions",
        # International variations
        "berufserfahrung", "erfahrung", "experiencia", "expérience", "esperienza", "werkervaring",
        "arbeidservaring", "experiência", "erhvervserfaring", "työkokemus", "pracovní zkušenosti",
        "工作经验", "経験", "अनुभव", "خبرة", "kinh nghiệm", "pengalaman", "ประสบการณ์", "doświadczenie",
        "εμπειρία"
    ],
    
    "SKILLS": [
        # English variations
        "skills", "skill", "abilities", "ability", "expertise", "competencies", "competences", 
        "capabilities", "capability", "proficiencies", "proficiency", "talents", "talent", "aptitudes", 
        "technicalskills", "technical skills", "softskill", "soft skills", "hardskills", "hard skills", 
        "core competencies", "skls", "proficiencies", "competency", "tech skills",
        # International variations
        "fähigkeiten", "kenntnisse", "habilidades", "compétences", "competenze", "vaardigheden",
        "ferdigheter", "competências", "færdigheder", "taidot", "dovednosti", "技能", "スキル",
        "कौशल", "مهارات", "kỹ năng", "keterampilan", "ทักษะ", "umiejętności", "δεξιότητες"
    ],
    
    "PROJECTS": [
        # English variations
        "projects", "project", "works", "portfolio", "showcase", "achievements", "initiatives", 
        "implementations", "developments", "applications", "software projects", "development projects", 
        "proj", "key projects", "personal projects", "academic projects", "research projects",
        # International variations
        "projekte", "proyectos", "projets", "progetti", "projecten", "prosjekter", "projetos",
        "projekter", "projektit", "projekty", "项目", "プロジェクト", "परियोजनाएं", "المشاريع",
        "dự án", "proyek", "โครงการ", "projekty", "έργα"
    ],
    
    "CONTACT": [
        # English variations
        "contact", "contacts", "contact information", "contact details", "contact info", "personal",
        "personal information", "personal details", "personal info", "personal data", "personaldetails", 
        "personal-details", "info", "details", "get in touch", "reach me", "contact me", "connect",
        # International variations
        "kontakt", "contacto", "coordonnées", "contatto", "contactgegevens", "kontaktinformasjon",
        "contato", "kontaktoplysninger", "yhteystiedot", "kontakt", "联系方式", "連絡先", "संपर्क",
        "معلومات الاتصال", "thông tin liên hệ", "kontak", "ข้อมูลติดต่อ", "kontakt", "επικοινωνία"
    ],
    
    "SUMMARY": [
        # English variations
        "summary", "professional summary", "career summary", "profile", "professional profile", 
        "overview", "about", "about me", "introduction", "bio", "biography", "background", "brief",
        "executive summary", "career objective", "career goal", "careerobjective", "objective", 
        "objective", "summary of qualifications", "qualifications summary", "career profile", "summ",
        "curriculum", "vitae", "cv", "resume summary", "professional background", "candidate summary",
        # International variations
        "zusammenfassung", "profil", "resumen", "résumé", "sommario", "samenvatting", "sammendrag",
        "resumo", "resumé", "yhteenveto", "souhrn", "概要", "要約", "सारांश", "ملخص", "tóm tắt",
        "ringkasan", "สรุป", "podsumowanie", "περίληψη"
    ],
    
    "CERTIFICATIONS": [
        # English variations
        "certifications", "certification", "certificates", "certificate", "credentials", "licenses", 
        "license", "accreditation", "accreditations", "qualification", "qualifications", "certs", 
        "certified", "diplomas", "diploma", "professional certifications", "professional development",
        # International variations
        "zertifizierungen", "certificaciones", "certifications", "certificazioni", "certificeringen",
        "sertifiseringer", "certificações", "certificeringer", "sertifikaatit", "certifikace", "认证",
        "証明書", "प्रमाणन", "الشهادات", "chứng chỉ", "sertifikasi", "ใบรับรอง", "certyfikaty",
        "πιστοποιήσεις"
    ],
    
    "AWARDS": [
        # English variations
        "awards", "award", "honors", "honour", "achievements", "recognition", "accolades", "prizes", 
        "prize", "distinctions", "accomplishments", "scholarships", "fellowship", "grant", "grants", 
        "scholarship", "awrds", "achievements", "recognitions", "accolades", "honors and awards",
        # International variations
        "auszeichnungen", "premios", "prix", "premi", "prijzen", "utmerkelser", "prêmios", "priser",
        "palkinnot", "ocenění", "奖项", "賞", "पुरस्कार", "الجوائز", "giải thưởng", "penghargaan",
        "รางวัล", "nagrody", "βραβεία"
    ],
    
    "ACHIEVEMENTS": [
        # English variations
        "achievements", "achievement", "accomplishments", "accomplishment", "successes", "success", 
        "milestones", "highlights", "key achievements", "significant achievements", "major accomplishments",
        "achvmts", "success stories", "key accomplishments", "professional achievements", "notable achievements",
        # International variations
        "errungenschaften", "logros", "réalisations", "risultati", "prestaties", "prestasjoner",
        "realizações", "resultater", "saavutukset", "úspěchy", "成就", "実績", "उपलब्धियां",
        "الإنجازات", "thành tựu", "prestasi", "ความสำเร็จ", "osiągnięcia", "επιτεύγματα"
    ],
    
    "LANGUAGES": [
        # English variations
        "languages", "language", "language skills", "language proficiency", "language proficiencies", 
        "language abilities", "spoken languages", "linguistic skills", "idiomas", "foreign languages", "lang",
        # International variations
        "sprachen", "idiomas", "langues", "lingue", "talen", "språk", "línguas", "sprog", "kielet",
        "jazyky", "语言", "言語", "भाषाएँ", "اللغات", "ngôn ngữ", "bahasa", "ภาษา", "języki", "γλώσσες"
    ],
    
    "PUBLICATIONS": [
        # English variations
        "publications", "publication", "papers", "paper", "articles", "article", "journal articles", 
        "research papers", "published works", "books", "book", "conference proceedings", "conf. proc.", 
        "pubs", "published papers", "research publications", "academic publications", "scientific publications",
        # International variations
        "veröffentlichungen", "publicaciones", "publications", "pubblicazioni", "publicaties",
        "publikasjoner", "publicações", "publikationer", "julkaisut", "publikace", "出版物", "発表",
        "प्रकाशन", "المنشورات", "công bố", "publikasi", "สิ่งพิมพ์", "publikacje", "δημοσιεύσεις"
    ],
    
    "ACTIVITIES": [
        # English variations
        "activities", "activity", "extracurricular", "extra-curricular", "extracurricular activities", 
        "co-curricular", "cocurricular", "volunteer", "volunteering", "community service", "leadership",
        "involvement", "interests", "hobbies", "hobby", "activities & interests", "activities and interests",
        "extra", "volunteer work", "community involvement", "participation", "campus involvement",
        # International variations
        "aktivitäten", "actividades", "activités", "attività", "activiteiten", "aktiviteter",
        "atividades", "aktiviteter", "aktiviteetit", "aktivity", "活动", "アクティビティ", "गतिविधियां",
        "الأنشطة", "hoạt động", "kegiatan", "กิจกรรม", "działania", "δραστηριότητες"
    ],
    
    "REFERENCES": [
        # English variations
        "references", "reference", "referees", "referee", "recommendations", "recommendation", 
        "recommenders", "recommender", "professional references", "refs", "testimonials", "endorsements",
        # International variations
        "referenzen", "referencias", "références", "referenze", "referenties", "referanser",
        "referências", "referencer", "viitteet", "reference", "推荐人", "参考", "संदर्भ", "المراجع",
        "người giới thiệu", "referensi", "อ้างอิง", "referencje", "αναφορές"
    ],
    
    "STRENGTH": [
        # English variations
        "strength", "strengths", "key strengths", "personal strengths", "professional strengths", 
        "core strengths", "strength areas", "strong points", "strong suits", "forte", "strong areas",
        # International variations
        "stärken", "fortalezas", "forces", "punti di forza", "sterke punten", "styrker",
        "pontos fortes", "styrker", "vahvuudet", "silné stránky", "优势", "強み", "ताकत",
        "نقاط القوة", "điểm mạnh", "kekuatan", "จุดแข็ง", "mocne strony", "δυνατά σημεία"
    ],
    
    "INTERESTS": [
        # English variations
        "interests", "interest", "hobbies", "hobby", "passions", "passion", "personal interests", 
        "leisure activities", "recreational activities", "pastimes", "pastime", "interests & hobbies",
        "interests and hobbies", "intrsts", "personal activities", "recreational pursuits",
        # International variations
        "interessen", "hobbys", "intereses", "aficiones", "intérêts", "loisirs", "interessi",
        "hobby", "interesses", "hobbyer", "interesses", "hobbies", "mielenkiinnon kohteet",
        "harrastukset", "zájmy", "兴趣爱好", "趣味", "रुचियां", "الاهتمامات", "sở thích", "minat",
        "ความสนใจ", "zainteresowania", "ενδιαφέροντα"
    ],
    
    "DECLARATION": [
        # English variations
        "declaration", "decleration", "verification", "authentication", "confirmation", "affirmation", 
        "authorization", "declaration of authenticity", "decl", "statement", "personal declaration",
        # International variations
        "erklärung", "declaración", "déclaration", "dichiarazione", "verklaring", "erklæring",
        "declaração", "erklæring", "vakuutus", "prohlášení", "声明", "宣言", "घोषणा", "إعلان",
        "tuyên bố", "deklarasi", "คำประกาศ", "deklaracja", "δήλωση"
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
    Check if the text matches common resume section headings with improved matching.
    
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
    
    if not clean or len(clean) < MIN_HEADING_LENGTH:
        return False, 0.0, None
    
    # Direct match first (highest confidence)
    if clean in section_keywords:
        return True, 1.0, section_keywords[clean]
    
    # Check for partial matches
    matches = []
    
    # Check if any keyword is fully contained in the text
    for keyword, section_type in section_keywords.items():
        if keyword in clean:
            # Calculate confidence based on keyword length relative to text length
            confidence = len(keyword) / len(clean)
            matches.append((confidence, section_type))
    
    # Check if the text is fully contained in any keyword (abbreviated form)
    if len(clean) >= MIN_HEADING_LENGTH:
        for keyword, section_type in section_keywords.items():
            if clean in keyword and len(clean) >= len(keyword) * 0.5:  # At least half the keyword
                # More confidence if the text is a larger portion of the keyword
                confidence = len(clean) / len(keyword) * 0.9  # Slightly lower than direct match
                matches.append((confidence, section_type))
    
    # Special case for common headings - look for these specific patterns
    common_headings = {
        "education": "EDUCATION",
        "experience": "EXPERIENCE",
        "skills": "SKILLS",
        "projects": "PROJECTS",
        "objective": "OBJECTIVE",
        "summary": "SUMMARY",
        "contact": "CONTACT"
    }
    
    for keyword, section_type in common_headings.items():
        if keyword in clean or clean in keyword:
            matches.append((0.8, section_type))  # Higher confidence for these common headings
    
    # Find the best match
    if matches:
        best_match = max(matches, key=lambda x: x[0])
        if best_match[0] >= CONFIDENCE_THRESHOLD:
            return True, best_match[0], best_match[1]
    
    return False, 0.0, None

def extract_text_with_style(pdf_path):
    """Extract text with style information using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(pdf_path)
        all_elements = []
        
        for page_num, page in enumerate(doc):
            try:
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
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
                continue
        
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

# def analyze_layout(elements, pdf_path):
#     """
#     Analyze the resume layout to detect columns.
#     Uses improved histogram analysis with fallback mechanisms.
    
#     Args:
#         elements (list): List of text elements
#         pdf_path (str): Path to the PDF file
        
#     Returns:
#         float: X-coordinate that separates left and right columns
#     """
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             page_width = pdf.pages[0].width
#     except Exception as e:
#         logger.error(f"Failed to open PDF with pdfplumber: {str(e)}")
#         # Default to 612 (standard letter width in points)
#         page_width = 612
    
#     # Get all x-coordinates
#     x_coords = [elem["x0"] for elem in elements]
    
#     if len(x_coords) < 10:  # Not enough data points for analysis
#         return page_width * 0.6
    
#     # Create histogram to find columns
#     # Divide page width into bins (e.g., 20 bins)
#     num_bins = 20
#     bin_width = page_width / num_bins
    
#     # Create histogram
#     hist, bin_edges = np.histogram(x_coords, bins=num_bins)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     # Smooth the histogram to reduce noise
#     smoothed_hist = np.convolve(hist, [0.2, 0.6, 0.2], mode='same')
    
#     # Find peaks in the histogram (potential column starts)
#     peaks = []
#     for i in range(1, len(smoothed_hist)-1):
#         if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:
#             # Only consider significant peaks (more than 5% of elements)
#             if smoothed_hist[i] > len(elements) * 0.05:
#                 peaks.append((bin_centers[i], smoothed_hist[i]))
    
#     # Sort peaks by position (x-coordinate)
#     peaks.sort(key=lambda x: x[0])
    
#     # Check if we have multiple prominent peaks
#     if len(peaks) >= 2:
#         # Find valleys between peaks (potential column dividers)
#         valleys = []
#         for i in range(len(peaks) - 1):
#             left_peak = peaks[i]
#             right_peak = peaks[i + 1]
            
#             # Find the minimum height between these peaks
#             left_idx = int(left_peak[0] / bin_width)
#             right_idx = int(right_peak[0] / bin_width)
            
#             if left_idx < right_idx - 1:  # Ensure there's space between peaks
#                 valley_idx = left_idx + np.argmin(smoothed_hist[left_idx+1:right_idx]) + 1
#                 valley_height = smoothed_hist[valley_idx]
#                 valley_pos = bin_centers[valley_idx]
                
#                 # Only consider significant valleys (less than 50% of peak heights)
#                 if valley_height < 0.5 * min(left_peak[1], right_peak[1]):
#                     valleys.append((valley_pos, valley_height))
        
#         # If we found valleys, use the most significant one
#         if valleys:
#             # Sort by height (ascending) to find the deepest valley
#             valleys.sort(key=lambda x: x[1])
#             return valleys[0][0]
    
#     # Check for two-column layout using simple spatial analysis
#     # Find the median x-coordinate
#     median_x = np.median(x_coords)
    
#     # Check if there's a gap around the median (suggesting two columns)
#     left_of_median = [x for x in x_coords if x < median_x - page_width * 0.05]
#     right_of_median = [x for x in x_coords if x > median_x + page_width * 0.05]
#     middle_count = len(x_coords) - len(left_of_median) - len(right_of_median)
    
#     # If there are few elements in the middle, it's likely a two-column layout
#     if middle_count < len(x_coords) * 0.2 and left_of_median and right_of_median:
#         # Find the midpoint of the gap
#         return median_x
    
#     # Default: divide page at 60% width
#     return page_width * 6
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
    top_margin = page_height * 0.2
    bottom_margin = page_height * 0.8
    
    # Only include elements in the main content area (exclude header/footer)
    content_elements = [e for e in elements if top_margin <= e["y0"] <= bottom_margin]
    
    if len(content_elements) < 10:
        return page_width * 0.6  # Default if not enough elements
    
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
    return page_width * 0.6



def identify_potential_headings(elements, font_stats):
    """
    Identify potential headings with improved precision to avoid false positives.
    
    Args:
        elements (list): List of text elements
        font_stats (dict): Font statistics from analyze_font_statistics
        
    Returns:
        list: Updated elements with heading likelihood scores
    """
    if not elements or not font_stats:
        return elements
    
    heading_threshold = font_stats["heading_threshold"]
    max_font_size = font_stats["max"]
    most_common_size = font_stats["most_common"]
    
    # Pre-filter to identify obvious headings (EDUCATION, SKILLS, etc.)
    # These common section names get higher confidence automatically
    common_section_names = {
        "education", "experience", "skills", "projects", "certifications", 
        "languages", "summary", "profile", "objective", "contact", "awards",
        "achievements", "publications", "references"
    }
    
    # First pass: Mark obvious headings
    for elem in elements:
        normalized_text = normalize(preprocess_text(elem["text"])).lower()
        is_known_heading = False
        
        # Check if this is a common section name
        if normalized_text in common_section_names:
            elem["is_likely_heading"] = True
            elem["confidence"] = 0.95  # Very high confidence
            elem["section_type"] = normalized_text.upper()
            is_known_heading = True
        
        # Check for spaced out text like "S K I L L S"
        elif elem["is_spaced"] and len(elem["text"]) > 5:
            text_without_spaces = elem["text"].replace(" ", "").lower()
            if text_without_spaces in common_section_names:
                elem["is_likely_heading"] = True
                elem["confidence"] = 0.9
                elem["section_type"] = text_without_spaces.upper()
                is_known_heading = True
        
        # Only proceed with standard analysis if not already identified
        if not is_known_heading:
            # Reset values for non-obvious headings
            elem["is_likely_heading"] = False
            elem["confidence"] = 0.0
            elem["section_type"] = None
    
    # Second pass: Analyze styling and font sizes for non-obvious headings
    for elem in elements:
        # Skip elements already identified as headings
        if elem["is_likely_heading"]:
            continue
            
        # Font size analysis - must be significantly larger than common text
        font_size_confidence = 0.0
        if elem["font_size"] > heading_threshold:
            # Must be at least 20% larger than threshold to be considered
            font_size_confidence = 0.6 * min(1.0, (elem["font_size"] - heading_threshold) / (max_font_size - heading_threshold + 0.001))
        
        # Style analysis - must have strong styling cues
        style_confidence = 0.0
        if elem["is_bold"]:
            style_confidence += 0.3
        if elem["is_capital"] and len(elem["text"]) >= 3:  # Must be at least 3 chars
            style_confidence += 0.3
        
        # Keyword matching - must be a recognized section name
        is_heading, keyword_confidence, section_type = is_heading_text(elem["text"])
        
        # Length constraint - headings should be short
        length_factor = 1.0 if len(elem["text"]) < 30 else 0.5
        
        # Combined score with stricter threshold
        combined_confidence = (
            keyword_confidence * 0.5 +
            font_size_confidence * 0.3 +
            style_confidence * 0.2
        ) * length_factor
        
        # Higher threshold to avoid false positives
        if combined_confidence >= CONFIDENCE_THRESHOLD + 0.1:  # Add 0.1 for stricter filtering
            elem["is_likely_heading"] = True
            elem["confidence"] = combined_confidence
            elem["section_type"] = section_type
    
    # Final cleanup: Remove "OTHER" classifications with low confidence
    for elem in elements:
        if elem["is_likely_heading"] and elem["section_type"] == "OTHER":
            # Require very high confidence for "OTHER" sections
            if elem["confidence"] < 0.85:
                elem["is_likely_heading"] = False
                elem["section_type"] = None
                elem["confidence"] = 0.0
    
    return elements


      

def identify_section_headings(elements, column_divider):
    """
    Identify section headings from elements with improved confidence scoring.
    
    Args:
        elements (list): List of text elements with heading likelihood scores
        column_divider (float): X-coordinate dividing the columns
        
    Returns:
        list: List of identified section headings
    """
    headings = []
    
    # First sort elements by confidence score (descending)
    sorted_elements = sorted(elements, key=lambda e: e["confidence"], reverse=True)
    
    # Track which text positions we've already marked as headings
    # to avoid duplicate/overlapping headings
    heading_positions = set()
    
    # First pass - collect the most confident headings
    for elem in sorted_elements:
        # Skip if not likely a heading
        if not elem["is_likely_heading"]:
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
            "type": elem["section_type"] or "OTHER",  # Default to OTHER if type is None
            "column": column,
            "confidence": elem["confidence"]
        })
        
        # Mark this position as used
        heading_positions.add(position_key)
    
    # Second pass - try to identify missed headings by proximity
    # This helps find section headings that might not match keywords but
    # have similar formatting to other headings
    if headings:
        # Calculate average y-distance between headings on the same page and column
        y_distances = []
        for i in range(len(headings) - 1):
            for j in range(i + 1, len(headings)):
                if (headings[i]["page"] == headings[j]["page"] and 
                    headings[i]["column"] == headings[j]["column"]):
                    y_distances.append(abs(headings[i]["y0"] - headings[j]["y0"]))
        
        avg_y_distance = np.mean(y_distances) if y_distances else 100  # Default if no pairs
        
        # Find potential headings based on formatting similarity
        for elem in elements:
            # Skip elements already identified as headings
            position_key = (elem["page"], round(elem["y0"]), round(elem["x0"]))
            if position_key in heading_positions:
                continue
                
            # Skip very short text
            if len(elem["text"].strip()) < MIN_HEADING_LENGTH:
                continue
            
            # Find headings on the same page
            same_page_headings = [h for h in headings if h["page"] == elem["page"]]
            if not same_page_headings:
                continue
                
            # Check if this element has similar formatting to existing headings
            similar_format = False
            for heading in same_page_headings:
                # Find the corresponding element for the heading
                heading_elem = None
                for e in elements:
                    if (e["page"] == heading["page"] and 
                        abs(e["y0"] - heading["y0"]) < 2 and 
                        abs(e["x0"] - heading["x0"]) < 2):
                        heading_elem = e
                        break
                
                if not heading_elem:
                    continue
                
                # Compare font size (within 10%)
                font_size_match = abs(elem["font_size"] - heading_elem["font_size"]) / heading_elem["font_size"] < 0.1
                # Compare styling
                style_match = (elem["is_bold"] == heading_elem["is_bold"] and 
                               elem["is_capital"] == heading_elem["is_capital"])
                
                if font_size_match and style_match:
                    similar_format = True
                    break
            
            if similar_format:
                # Check if this element is positioned reasonably (not too close to other headings)
                too_close = False
                for heading in same_page_headings:
                    if heading["column"] == ("left" if elem["x0"] < column_divider else "right"):
                        if abs(elem["y0"] - heading["y0"]) < avg_y_distance * 0.5:
                            too_close = True
                            break
                
                if not too_close:
                    # Determine which column this heading belongs to
                    column = "left" if elem["x0"] < column_divider else "right"
                    
                    # Attempt to infer section type from text
                    _, _, section_type = is_heading_text(elem["text"])
                    
                    # Add to headings list with a lower confidence
                    headings.append({
                        "page": elem["page"],
                        "text": elem["text"],
                        "x0": elem["x0"],
                        "y0": elem["y0"],
                        "x1": elem["x1"],
                        "y1": elem["y1"],
                        "type": section_type or "OTHER",  # Default to OTHER if type is None
                        "column": column,
                        "confidence": 0.6  # Lower confidence for inferred headings
                    })
                    
                    # Mark this position as used
                    heading_positions.add(position_key)
    
    # Sort headings by page and y-coordinate
    return sorted(headings, key=lambda h: (h["page"], h["y0"]))

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
        
        # Count headings in each column
        left_headings = [h for h in headings if h["column"] == "left"]
        right_headings = [h for h in headings if h["column"] == "right"]
        logger.info(f"  - Left column: {len(left_headings)} headings")
        logger.info(f"  - Right column: {len(right_headings)} headings")
        
        # Log each heading with confidence
        for heading in headings:
            logger.info(f"  - {heading['text']} ({heading['type']}, {heading['column']} column, "
                       f"confidence: {heading.get('confidence', 0.0):.2f})")
        
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


