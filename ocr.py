import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF type detection
# ---------------------------------------------------------------------------

def detect_pdf_type(filepath: str) -> str:
    """Return 'text' if the PDF has embedded text, 'scan' otherwise."""
    import fitz  # pymupdf
    try:
        doc = fitz.open(filepath)
        if len(doc) == 0:
            return "scan"
        total_chars = sum(len(page.get_text()) for page in doc)
        avg_chars   = total_chars / len(doc)
        doc.close()
        return "text" if avg_chars > 100 else "scan"
    except Exception as e:
        logger.error("detect_pdf_type error for %s: %s", filepath, e)
        return "scan"


# ---------------------------------------------------------------------------
# Text PDF — pymupdf
# ---------------------------------------------------------------------------

def extract_text_pymupdf(filepath: str) -> str:
    """Extract embedded text from a text-based PDF."""
    import fitz
    try:
        doc   = fitz.open(filepath)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        logger.error("extract_text_pymupdf error for %s: %s", filepath, e)
        return ""


# ---------------------------------------------------------------------------
# Scan PDF — EasyOCR
# ---------------------------------------------------------------------------

def extract_text_easyocr(filepath: str) -> str:
    """OCR a scanned PDF using EasyOCR (ru + en)."""
    import fitz
    import easyocr
    import numpy as np

    try:
        reader = easyocr.Reader(['ru', 'en'], gpu=False)
        doc    = fitz.open(filepath)
        pages_text = []

        for page_num, page in enumerate(doc):
            # Render page to image (2x scale for better OCR quality)
            mat    = fitz.Matrix(2, 2)
            pix    = page.get_pixmap(matrix=mat)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            result = reader.readtext(img_np, detail=0, paragraph=True)
            pages_text.append("\n".join(result))
            logger.info("OCR page %d/%d done", page_num + 1, len(doc))

        doc.close()
        return "\n\n".join(pages_text)

    except Exception as e:
        logger.error("extract_text_easyocr error for %s: %s", filepath, e)
        return ""


# ---------------------------------------------------------------------------
# Structured PDF — marker-pdf
# ---------------------------------------------------------------------------

def extract_text_marker(filepath: str) -> str:
    """Convert a structured PDF to Markdown using marker-pdf."""
    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models

        models   = load_all_models()
        full_text, _, _ = convert_single_pdf(filepath, models)
        return full_text

    except Exception as e:
        logger.error("extract_text_marker error for %s: %s", filepath, e)
        return ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_pdf(filepath: str) -> str:
    """Detect PDF type and extract text using the best available method.

    Priority:
      1. marker-pdf  — for structured/multi-column layouts (returns Markdown)
      2. pymupdf     — for simple text PDFs
      3. easyocr     — for scanned PDFs
    Falls back down the chain if a method returns empty output.
    """
    try:
        pdf_type = detect_pdf_type(filepath)
        logger.info("PDF type detected: %s (%s)", pdf_type, Path(filepath).name)

        if pdf_type == "text":
            # Try marker first (richer Markdown output)
            text = extract_text_marker(filepath)
            if not text.strip():
                text = extract_text_pymupdf(filepath)
        else:
            # Scanned — go straight to EasyOCR
            text = extract_text_easyocr(filepath)

        if not text.strip():
            logger.warning("All extraction methods returned empty for %s", filepath)

        return text

    except Exception as e:
        logger.error("process_pdf fatal error for %s: %s", filepath, e)
        return ""
