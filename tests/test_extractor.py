# ─────────────────────────────────────────────────────────────────────────────
# tests/test_extractor.py
#
# WHY TESTS MATTER:
#   Before deploying, we need to verify that our extractor correctly
#   parses financial documents. If extraction is broken, every AI answer
#   will be wrong — and the user won't know why.
#
# HOW TO RUN:
#   pytest tests/ -v
#   pytest tests/ -v --cov=app   (with coverage report)
#
# WHAT WE TEST:
#   1. JSON extraction (our generated samples)
#   2. Text field parsing (regex-based heuristics)
#   3. Image handling (base64 encoding)
#   4. Error handling (bad files, missing files)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path

# Add the app directory to Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from extractor import (
    extract_document,
    extract_from_json,
    parse_text_to_fields,
    normalize_document,
)


# ─── Fixtures (reusable test data) ────────────────────────────────────────────
# WHY FIXTURES: Instead of repeating setup code in every test,
# fixtures create shared test data once and inject it where needed.

@pytest.fixture
def sample_invoice_data():
    """A realistic invoice dict that matches our generated sample format."""
    return {
        "document_type": "invoice",
        "invoice_number": "INV-0042",
        "invoice_date": "2024-11-15",
        "due_date": "2024-12-15",
        "vendor_name": "Acme Cloud Services Ltd",
        "vendor_address": "123 Tech Street, San Francisco, CA 94105",
        "vendor_email": "billing@acme.com",
        "customer_name": "Jane Smith",
        "customer_address": "456 Main Ave, New York, NY 10001",
        "line_items": [
            {"description": "Cloud Storage (1TB)", "quantity": 1, "unit_price": 99.99, "amount": 99.99},
            {"description": "API Calls (1M requests)", "quantity": 1, "unit_price": 50.00, "amount": 50.00},
        ],
        "subtotal": 149.99,
        "tax_rate": 0.10,
        "tax_amount": 15.00,
        "total_amount": 164.99,
        "currency": "USD",
        "status": "Unpaid",
    }


@pytest.fixture
def sample_statement_data():
    """A realistic credit card statement dict."""
    return {
        "document_type": "credit_card_statement",
        "statement_id": "STMT-0007",
        "statement_date": "2024-11-01",
        "account_holder": "John Doe",
        "card_last_four": "4521",
        "previous_balance": 1500.00,
        "payments_received": 500.00,
        "new_charges": 320.45,
        "taxes_and_fees": 12.50,
        "total_due": 1332.95,
        "transactions": [
            {"date": "2024-10-28", "merchant": "Amazon Web Services", "description": "EC2 Usage", "amount": 320.45, "category": "Technology"},
        ],
        "currency": "USD",
    }


@pytest.fixture
def invoice_json_file(sample_invoice_data, tmp_path):
    """Write sample invoice data to a temp JSON file and return the path."""
    filepath = tmp_path / "test_invoice.json"
    filepath.write_text(json.dumps(sample_invoice_data, indent=2))
    return str(filepath)


# ─── Tests: JSON extraction ───────────────────────────────────────────────────

class TestJsonExtraction:
    """Tests for loading and normalizing JSON financial documents."""

    def test_extract_invoice_json(self, invoice_json_file):
        """Should successfully load and parse an invoice JSON file."""
        result = extract_from_json(invoice_json_file)

        # Check that key fields were extracted
        assert result["document_type"] == "invoice"
        assert result["vendor_name"] == "Acme Cloud Services Ltd"
        assert result["total_amount"] == 164.99
        assert result["invoice_date"] == "2024-11-15"

    def test_extract_statement_json(self, sample_statement_data, tmp_path):
        """Should correctly extract credit card statement fields."""
        filepath = tmp_path / "test_statement.json"
        filepath.write_text(json.dumps(sample_statement_data))

        result = extract_from_json(str(filepath))

        assert result["document_type"] == "credit_card_statement"
        assert result["total_amount"] == 1332.95
        # Statement uses statement_date, which should map to invoice_date
        assert result["invoice_date"] == "2024-11-01"

    def test_extract_preserves_line_items(self, invoice_json_file):
        """Line items should be preserved after extraction."""
        result = extract_from_json(invoice_json_file)

        assert len(result["line_items"]) == 2
        assert result["line_items"][0]["description"] == "Cloud Storage (1TB)"
        assert result["line_items"][0]["amount"] == 99.99

    def test_raw_text_is_generated(self, invoice_json_file):
        """raw_text should be populated so it can be embedded into ChromaDB."""
        result = extract_from_json(invoice_json_file)

        # raw_text must exist and be non-empty (it's what gets embedded)
        assert result["raw_text"] is not None
        assert len(result["raw_text"]) > 50
        # Should contain searchable content
        assert "Acme Cloud Services" in result["raw_text"]


# ─── Tests: Text parsing (heuristics) ────────────────────────────────────────

class TestTextParsing:
    """Tests for the regex-based field extraction from raw text."""

    def test_parse_total_amount(self):
        """Should extract total amount from invoice text."""
        text = "Invoice #1234\nSubtotal: $150.00\nTax: $15.00\nTotal: $165.00"
        result = parse_text_to_fields(text)
        assert result["total_amount"] == 165.00

    def test_parse_total_due_variant(self):
        """Should handle 'Total Due' as well as 'Total'."""
        text = "Statement Date: 2024-11-01\nNew Charges: $320.45\nTotal Due: $1,332.95"
        result = parse_text_to_fields(text)
        assert result["total_amount"] == 1332.95   # Commas removed

    def test_parse_invoice_number(self):
        """Should extract invoice number in various formats."""
        text = "Invoice #: INV-2024-0042\nDate: 2024-11-15"
        result = parse_text_to_fields(text)
        assert result["invoice_number"] == "INV-2024-0042"

    def test_parse_date(self):
        """Should extract invoice date."""
        text = "Invoice Date: 2024-11-15\nDue Date: 2024-12-15"
        result = parse_text_to_fields(text)
        assert result["invoice_date"] == "2024-11-15"
        assert result["due_date"] == "2024-12-15"

    def test_parse_tax(self):
        """Should extract tax amount."""
        text = "Subtotal: $150.00\nTax (10%): $15.00\nTotal: $165.00"
        result = parse_text_to_fields(text)
        assert result["tax_amount"] == 15.00

    def test_document_type_detection(self):
        """Should detect document type from keywords."""
        invoice_text = "INVOICE\nBilled to: Jane Smith"
        statement_text = "Credit Card Statement\nAccount: 4521"

        assert parse_text_to_fields(invoice_text)["document_type"] == "invoice"
        assert parse_text_to_fields(statement_text)["document_type"] == "credit_card_statement"

    def test_empty_text_returns_empty_fields(self):
        """Should handle empty text without crashing."""
        result = parse_text_to_fields("")
        assert result["total_amount"] is None
        assert result["vendor_name"] is None
        assert result["line_items"] == []


# ─── Tests: Document normalization ───────────────────────────────────────────

class TestNormalization:
    """Tests for normalize_document() which standardizes field names."""

    def test_normalize_maps_statement_date(self, sample_statement_data):
        """statement_date should be mapped to invoice_date field."""
        result = normalize_document(sample_statement_data)
        assert result["invoice_date"] == "2024-11-01"

    def test_normalize_maps_transactions(self, sample_statement_data):
        """transactions should be mapped to line_items field."""
        result = normalize_document(sample_statement_data)
        assert len(result["line_items"]) == 1
        assert result["line_items"][0]["merchant"] == "Amazon Web Services"

    def test_normalize_preserves_original(self, sample_invoice_data):
        """Original data should be preserved under _original key."""
        result = normalize_document(sample_invoice_data)
        assert "_original" in result
        assert result["_original"]["vendor_email"] == "billing@acme.com"

    def test_normalize_handles_missing_fields(self):
        """Should not crash on minimal data."""
        minimal = {"document_type": "unknown"}
        result = normalize_document(minimal)
        assert result["vendor_name"] is None
        assert result["total_amount"] is None


# ─── Tests: Main extract_document router ─────────────────────────────────────

class TestExtractDocumentRouter:
    """Tests for the main extract_document() function that routes by file type."""

    def test_routes_json_correctly(self, invoice_json_file):
        """Should route .json files to JSON extractor."""
        result = extract_document(invoice_json_file)
        assert result["document_type"] == "invoice"
        assert result["vendor_name"] == "Acme Cloud Services Ltd"

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            extract_document("/nonexistent/path/file.pdf")

    def test_raises_on_unsupported_type(self, tmp_path):
        """Should raise ValueError for unsupported file extensions."""
        bad_file = tmp_path / "document.xlsx"
        bad_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_document(str(bad_file))

    def test_handles_image_file(self, tmp_path):
        """Should handle image files by returning base64-encoded data."""
        # Create a minimal 1x1 PNG in memory
        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        img_path = tmp_path / "test_statement.png"
        img.save(str(img_path))

        result = extract_document(str(img_path))

        assert result["document_type"] == "image"
        assert result["image_base64"] is not None
        assert result["mime_type"] == "image/png"
        assert result["dimensions"]["width"] == 100
