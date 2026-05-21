"""Shared custom exceptions and user-facing error messages."""

from __future__ import annotations


class AuditError(Exception):
    def __init__(
        self,
        message: str,
        error_code: str,
        file_name: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.file_name = file_name
        self.suggestion = suggestion
        super().__init__(message)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "error": self.error_code,
            "message": self.message,
            "file": self.file_name,
            "suggestion": self.suggestion,
        }


ERROR_MESSAGES: dict[str, dict[str, str]] = {
    "DATE_PARSE_FAILED": {
        "message": "Could not parse dates in this file",
        "suggestion": "Check date format is DD/MM/YYYY or DD-MM-YYYY",
    },
    "AMOUNT_PARSE_FAILED": {
        "message": "Could not read amount values",
        "suggestion": "Ensure amounts are numbers without special characters",
    },
    "COLUMN_NOT_FOUND": {
        "message": "Required columns not found in file",
        "suggestion": "Supported banks: HDFC, ICICI, SBI, PNB, Axis, Kotak, Yes Bank",
    },
    "EMPTY_FILE": {
        "message": "File has no data rows",
        "suggestion": "Check the file has at least 2 transactions",
    },
    "PDF_ENCRYPTED": {
        "message": "PDF is password protected",
        "suggestion": "Remove password protection before uploading",
    },
    "UNKNOWN_FORMAT": {
        "message": "File format not recognized",
        "suggestion": "Export CSV directly from your bank's website",
    },
    "OLLAMA_TIMEOUT": {
        "message": "Explanation generation timed out",
        "suggestion": "Click the finding again to retry",
    },
    # Backward-compatible aliases used by existing code/tests.
    "UNKNOWN_BANK_FORMAT": {
        "message": "Required columns not found in file",
        "suggestion": "Supported banks: HDFC, ICICI, SBI, PNB, Axis, Kotak, Yes Bank",
    },
    "OLLAMA_NOT_RUNNING": {
        "message": "Explanation generation timed out",
        "suggestion": "Click the finding again to retry",
    },
}

# Backward compatibility for existing imports.
ERROR_CODES = ERROR_MESSAGES


def get_error_code(code: str) -> dict[str, str]:
    if code not in ERROR_MESSAGES:
        raise KeyError(f"Unknown error code: {code}")
    return ERROR_MESSAGES[code]
