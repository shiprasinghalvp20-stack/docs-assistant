"""Generate samples/company_handbook.pdf - a small fictional handbook for demos and tests.

Each page holds different facts so we can later check that answers cite the right page.
Page 3 is intentionally blank to behave like a scanned page (no extractable text).

Run:  python scripts/make_sample_pdf.py
"""
from pathlib import Path

from fpdf import FPDF

OUTPUT = Path(__file__).resolve().parent.parent / "samples" / "company_handbook.pdf"

PAGES = [
    (
        "Example Corp Employee Handbook - Leave Policy",
        "Every full-time employee receives 24 days of paid annual leave per calendar year. "
        "Up to 5 unused days can be carried forward to the next year. "
        "Employees also receive 10 days of paid sick leave. A medical certificate is "
        "required for sick leave longer than 2 consecutive days.",
    ),
    (
        "Remote Work Policy",
        "Employees may work remotely up to 3 days per week with their manager's approval. "
        "All remote work must use the company VPN. Core collaboration hours are "
        "11:00 to 16:00 IST, during which employees must be reachable on chat.",
    ),
    None,  # blank page - simulates a scanned image with no text layer
    (
        "Expense Reimbursement",
        "Expense claims must be submitted within 30 days of the purchase date, with receipts. "
        "The daily meal allowance during business travel is Rs. 1,500. "
        "Claims above Rs. 25,000 require approval from the finance team.",
    ),
]


def build(output: Path = OUTPUT) -> Path:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page in PAGES:
        pdf.add_page()
        if page is None:
            continue
        title, body = page
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.multi_cell(0, 10, title)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 7, body)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    return output


if __name__ == "__main__":
    print(f"Wrote {build()}")
