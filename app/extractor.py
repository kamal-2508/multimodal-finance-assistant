# ─────────────────────────────────────────────────────────────────────────────
# app/extractor.py
#
# WHY THIS FILE EXISTS:
#   This is the first step in our pipeline. Before the AI can answer
#   questions about a financial document, it needs to READ and UNDERSTAND
#   the document structure — vendor names, amounts, dates, line items, etc.
#
#   We handle 3 input types:
#     1. PDF files  → parsed with Docling (understands tables + layout)
#     2. Images (PNG/JPG) → converted to base64 for vision LLM
#     3. JSON files → our generated sample data (already structured)
#
# DOCLING vs PyPDF2:
#   PyPDF2 just dumps raw text — it loses table structure.
#   Docling understands layout, so "Amount: $320.45" stays linked to
#   "Amazon Web Services EC2 Usage" instead of floating as separate text.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import base64
import logging
from pathlib import Path

from PIL import Image                          # For handling image files
import PyPDF2                                  # Fallback PDF text extractor

# Set up logging so we can see what the extractor is doing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── JSON extraction ──────────────────────────────────────────────────────────

def extract_from_json(filepath: str) -> dict:
    """
    Load a pre-structured JSON document (our generated samples).

    WHY: Our generated sample data is already in structured JSON format.
    In a real project, this function would handle cases where a user
    uploads a bank export file (many banks let you export CSV/JSON).
    """
    logger.info(f"Loading JSON document: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    return normalize_document(data)


# ─── PDF extraction ───────────────────────────────────────────────────────────

def extract_from_pdf(filepath: str) -> dict:
    """
    Extract text and fields from a PDF file.

    WHY DOCLING FIRST, PyPDF2 AS FALLBACK:
    Docling is better at preserving table structure (crucial for invoices),
    but it requires more memory. PyPDF2 is lightweight and works as backup.

    In production you'd always use Docling, but PyPDF2 ensures the app
    doesn't crash if Docling has installation issues in some environments.
    """
    logger.info(f"Extracting text from PDF: {filepath}")
    raw_text = ""

    # Try Docling first (better layout understanding)
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(filepath)
        # Markdown preserves table structure better than plain text
        raw_text = result.document.export_to_markdown()
        logger.info("✓ Docling extraction successful")

    except Exception as e:
        # Docling failed — fall back to PyPDF2 for basic text extraction
        logger.warning(f"Docling failed ({e}), falling back to PyPDF2")
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                raw_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            logger.info("✓ PyPDF2 fallback extraction successful")
        except Exception as e2:
            logger.error(f"Both PDF extractors failed: {e2}")
            raw_text = ""

    return parse_text_to_fields(raw_text, source_file=filepath)


# ─── Image extraction ─────────────────────────────────────────────────────────

def extract_from_image(filepath: str) -> dict:
    """
    Convert an image to base64 so a vision LLM can read it.

    WHY BASE64:
    Vision LLMs accept images as base64-encoded strings inside the API
    request. We don't OCR the image ourselves — the LLM handles rotated
    text, handwriting, and unusual layouts much better than Tesseract.
    """
    logger.info(f"Processing image: {filepath}")
    try:
        img = Image.open(filepath)
        width, height = img.size
        mode = img.mode

        # Convert RGBA to RGB (some PNGs have alpha channel)
        if mode == "RGBA":
            img = img.convert("RGB")

        with open(filepath, "rb") as f:
            image_bytes = f.read()
        b64_data = base64.b64encode(image_bytes).decode("utf-8")

        ext = Path(filepath).suffix.lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        logger.info(f"✓ Image encoded: {width}x{height}px, {len(b64_data)//1024}KB")

        return {
            "document_type": "image",
            "source_file": filepath,
            "image_base64": b64_data,
            "mime_type": mime_type,
            "dimensions": {"width": width, "height": height},
            "raw_text": None,
            "vendor_name": None,
            "total_amount": None,
            "invoice_date": None,
            "line_items": [],
            "extraction_method": "vision_llm",
        }

    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return {"document_type": "image", "error": str(e), "source_file": filepath}


# ─── Text parser (supports USD + INR formats) ─────────────────────────────────

def parse_text_to_fields(raw_text: str, source_file: str = "") -> dict:
    """
    Use LLM to extract fields from any invoice format.
    Works for Indian (INR/GST), US (USD), UK (VAT) — any format.
    """
    import os
    from groq import Groq

    fields = {
        "document_type": "invoice",
        "source_file": source_file,
        "raw_text": raw_text,
        "vendor_name": None,
        "total_amount": None,
        "invoice_date": None,
        "due_date": None,
        "invoice_number": None,
        "line_items": [],
        "tax_amount": None,
        "extraction_method": "llm",
    }

    if not raw_text:
        return fields

    try:
        # Call Groq LLM to extract fields from raw text
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""Extract these fields from the invoice text below.
Return ONLY a JSON object with exactly these keys:
- vendor_name (string)
- total_amount (number only, no currency symbol)
- tax_amount (number only, no currency symbol)
- invoice_number (string)
- invoice_date (string)
- document_type (invoice/receipt/statement)

Invoice text:
{raw_text[:2000]}

Return only valid JSON. No explanation. No markdown."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )

        # Parse LLM response as JSON
        import json
        result_text = response.choices[0].message.content.strip()

        # Clean up markdown if present
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(result_text)

        # Map extracted fields to our standard format
        fields["vendor_name"]    = extracted.get("vendor_name")
        fields["total_amount"]   = extracted.get("total_amount")
        fields["tax_amount"]     = extracted.get("tax_amount")
        fields["invoice_number"] = extracted.get("invoice_number")
        fields["invoice_date"]   = extracted.get("invoice_date")
        fields["document_type"]  = extracted.get("document_type", "invoice")
        fields["extraction_method"] = "llm"

        logger.info("✓ LLM extraction successful")

    except Exception as e:
        logger.warning(f"LLM extraction failed ({e}), fields will be N/A")

    return fields


# ─── Normalizer ───────────────────────────────────────────────────────────────

def normalize_document(data: dict) -> dict:
    """
    Normalize any document dict to a standard format.

    WHY: Our generated JSON samples have slightly different field names
    than what parse_text_to_fields() returns. This maps everything to
    a consistent schema that the rest of the pipeline expects.
    """
    return {
        "document_type":  data.get("document_type", "unknown"),
        "source_file":    data.get("source_file", ""),
        "raw_text":       json.dumps(data, indent=2),
        "vendor_name":    data.get("vendor_name"),
        "total_amount":   data.get("total_amount"),
        "invoice_date":   data.get("invoice_date") or data.get("statement_date"),
        "due_date":       data.get("due_date"),
        "invoice_number": data.get("invoice_number") or data.get("statement_id"),
        "line_items":     data.get("line_items") or data.get("transactions", []),
        "tax_amount":     data.get("tax_amount"),
        "extraction_method": "structured_json",
        "_original": data,
    }


# ─── Main router ──────────────────────────────────────────────────────────────

def extract_document(filepath: str) -> dict:
    """
    Main function: auto-detect file type and extract fields.

    This is called by main.py when a user uploads any file.
    Routes to the correct extractor based on file extension.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()

    if ext == ".json":
        return extract_from_json(filepath)
    elif ext == ".pdf":
        return extract_from_pdf(filepath)
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        return extract_from_image(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, PNG, JPG, JSON")


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_path = "./data/sample_docs/invoice_0001.json"
    if os.path.exists(sample_path):
        result = extract_document(sample_path)
        print("\n📄 Extracted fields:")
        for k, v in result.items():
            if k != "_original" and k != "raw_text":
                print(f"   {k}: {v}")
    else:
        print("⚠️  No sample files found. Run: python data/generate_samples.py first")