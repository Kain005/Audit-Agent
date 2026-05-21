# quick_test.py — run this directly, not through Streamlit
from bank_layouts.hdfc_layout import HDFCLayoutParser

from ingestion.bank_layouts.hdfc_layout import HDFCLayoutParser
from ingestion.bank_layouts.sbi_layout import SBILayoutParser

# just test detect() works
pages = [{"page_number": 1, "raw_text": "HDFC BANK Statement", "confidence": 85.0, "low_confidence": False}]
print(HDFCLayoutParser().detect(pages))  # should print True