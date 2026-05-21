"""Windows-safe runtime paths for external OCR/PDF tools."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve repository root relative to this file.
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)

DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_POPPLER_PATH = r"C:\poppler\Library\bin"


def get_tesseract_path() -> Path:
    """Return the configured Tesseract executable path."""
    return Path(os.environ.get("TESSERACT_PATH", DEFAULT_TESSERACT_PATH))


def get_poppler_path() -> Path:
    """Return the configured Poppler bin directory path."""
    return Path(os.environ.get("POPPLER_PATH", DEFAULT_POPPLER_PATH))


def configure_pytesseract() -> Path:
    """Configure pytesseract command path from environment defaults."""
    import pytesseract

    tesseract_path = get_tesseract_path()
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
    return tesseract_path
