# ─────────────────────────────────────────────────────────────────────────────
# generate_samples.py
#
# WHY THIS FILE EXISTS:
#   We need realistic financial documents to test our AI assistant.
#   Instead of using real private data, we generate fake-but-realistic
#   invoices and credit card statements using the Faker library.
#
# HOW TO RUN:
#   python data/generate_samples.py
#   → Creates 20 PDF invoices + 10 credit card statement text files
#     inside data/sample_docs/
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import random
from datetime import datetime, timedelta
from faker import Faker

# Faker gives us realistic names, addresses, company names etc.
fake = Faker()

# Where to save the generated files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def random_date(days_back=90):
    """Return a random date within the last `days_back` days."""
    start = datetime.now() - timedelta(days=days_back)
    random_days = random.randint(0, days_back)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


def generate_invoice(invoice_id: int) -> dict:
    """
    Generate one fake invoice as a Python dict.

    WHY A DICT?
    We store it as JSON so our extractor.py can later read it back
    and treat it like a real extracted document. In a real project,
    you'd have actual PDFs — here we simulate the extracted fields.
    """
    # Generate 2–5 random line items (products/services)
    num_items = random.randint(2, 5)
    line_items = []
    subtotal = 0.0

    for _ in range(num_items):
        qty = random.randint(1, 10)
        unit_price = round(random.uniform(10, 500), 2)
        amount = round(qty * unit_price, 2)
        subtotal += amount

        line_items.append({
            "description": fake.bs().title(),   # e.g. "Synergize Scalable Supply-Chains"
            "quantity": qty,
            "unit_price": unit_price,
            "amount": amount,
        })

    subtotal = round(subtotal, 2)
    tax_rate = random.choice([0.05, 0.10, 0.18])   # 5%, 10%, or 18% tax
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    return {
        # Document metadata
        "document_type": "invoice",
        "invoice_number": f"INV-{invoice_id:04d}",
        "invoice_date": random_date(),
        "due_date": random_date(30),

        # Vendor (who is billing)
        "vendor_name": fake.company(),
        "vendor_address": fake.address().replace("\n", ", "),
        "vendor_email": fake.company_email(),

        # Customer (who is being billed)
        "customer_name": fake.name(),
        "customer_address": fake.address().replace("\n", ", "),

        # Financial fields
        "line_items": line_items,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "tax_amount": tax,
        "total_amount": total,
        "currency": "USD",
        "payment_method": random.choice(["Credit Card", "Bank Transfer", "PayPal"]),
        "status": random.choice(["Paid", "Unpaid", "Overdue"]),
    }


def generate_credit_card_statement(statement_id: int) -> dict:
    """
    Generate one fake credit card statement.

    WHY CREDIT CARD STATEMENTS?
    The original project description specifically shows a credit card
    statement as the example. Users often see mystery charges and want
    the AI to explain them — that's our core use case.
    """
    # Generate 5–12 transactions
    num_transactions = random.randint(5, 12)
    transactions = []
    total_charges = 0.0

    # Realistic merchant categories users would actually question
    merchants = [
        ("Amazon Web Services", "Cloud computing usage charge"),
        ("Netflix", "Monthly streaming subscription"),
        ("Uber Eats", "Food delivery service"),
        ("Google Cloud", "Storage and compute services"),
        ("Spotify", "Music streaming subscription"),
        ("Adobe Creative Cloud", "Software subscription"),
        ("Whole Foods Market", "Grocery purchase"),
        ("Shell Gas Station", "Fuel purchase"),
        ("Delta Airlines", "Flight booking"),
        ("Airbnb", "Accommodation booking"),
        ("Zoom Video", "Video conferencing subscription"),
        ("GitHub", "Developer tools subscription"),
    ]

    for i in range(num_transactions):
        merchant, description = random.choice(merchants)
        amount = round(random.uniform(5, 500), 2)
        total_charges += amount
        transactions.append({
            "date": random_date(),
            "merchant": merchant,
            "description": description,
            "amount": amount,
            "category": random.choice(["Technology", "Food", "Travel", "Shopping", "Entertainment"]),
        })

    total_charges = round(total_charges, 2)
    previous_balance = round(random.uniform(1000, 5000), 2)
    payments = round(random.uniform(200, 1000), 2)

    return {
        "document_type": "credit_card_statement",
        "statement_id": f"STMT-{statement_id:04d}",
        "statement_date": random_date(30),
        "account_holder": fake.name(),
        "card_last_four": str(random.randint(1000, 9999)),
        "previous_balance": previous_balance,
        "payments_received": payments,
        "new_charges": total_charges,
        "taxes_and_fees": round(random.uniform(5, 25), 2),
        "total_due": round(previous_balance - payments + total_charges, 2),
        "transactions": transactions,
        "currency": "USD",
    }


def save_as_json(data: dict, filename: str):
    """Save a dict to a JSON file — this acts as our 'extracted document'."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Created: {filename}")


def main():
    print("🔧 Generating sample financial documents...")
    print(f"   Output folder: {OUTPUT_DIR}\n")

    # Generate 20 invoices
    print("📄 Generating invoices...")
    for i in range(1, 21):
        invoice = generate_invoice(i)
        save_as_json(invoice, f"invoice_{i:04d}.json")

    # Generate 10 credit card statements
    print("\n💳 Generating credit card statements...")
    for i in range(1, 11):
        statement = generate_credit_card_statement(i)
        save_as_json(statement, f"statement_{i:04d}.json")

    print(f"\n✅ Done! Generated 30 sample documents in '{OUTPUT_DIR}'")
    print("   Next step: run 'python app/main.py' to start the app")


if __name__ == "__main__":
    main()
