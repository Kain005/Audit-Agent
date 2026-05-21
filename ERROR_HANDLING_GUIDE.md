"""
Error Handling Guide for Finance Audit Agent

This document explains how the error handling system works throughout the project.
"""

# ============================================================================
# 1. ERROR CODES REFERENCE
# ============================================================================

ERROR_CODES = {
    # Parsing Errors
    "DATE_PARSE_FAILED": {
        "message": "Could not parse dates in this file",
        "suggestion": "Check date format is DD/MM/YYYY or DD-MM-YYYY"
    },
    "AMOUNT_PARSE_FAILED": {
        "message": "Could not read amount values",
        "suggestion": "Ensure amounts are numbers without special characters"
    },
    "COLUMN_NOT_FOUND": {
        "message": "Required columns not found",
        "suggestion": "Check file matches expected bank statement format"
    },
    "EMPTY_FILE": {
        "message": "File appears to be empty or has no data rows",
        "suggestion": "Check the file has at least 5 transactions"
    },
    
    # PDF Errors
    "PDF_SCANNED": {
        "message": "PDF appears to be scanned image",
        "suggestion": "OCR will be attempted but accuracy may be lower"
    },
    "PDF_ENCRYPTED": {
        "message": "PDF is password protected",
        "suggestion": "Remove password protection before uploading"
    },
    
    # Format/Detection Errors
    "UNKNOWN_BANK_FORMAT": {
        "message": "Bank format not recognized",
        "suggestion": "Supported banks: HDFC, ICICI, SBI, PNB, Axis, Kotak, Yes Bank"
    },
    "INVALID_FILE_FORMAT": {
        "message": "File format is not supported",
        "suggestion": "Supported formats: CSV, Excel (XLS/XLSX), PDF, TXT"
    },
    
    # Quality Errors
    "LOW_CONFIDENCE_EXTRACTION": {
        "message": "Document extracted with low confidence",
        "suggestion": "Results may be incomplete — manual review recommended"
    },
    
    # LLM Errors
    "OLLAMA_NOT_RUNNING": {
        "message": "Ollama is not running — explanations unavailable",
        "suggestion": "Run 'ollama serve' in a separate terminal"
    },
    
    # File Size Errors
    "FILE_TOO_LARGE": {
        "message": "File exceeds 10MB limit",
        "suggestion": "Split the file into smaller chunks"
    },
    
    # Generic Errors
    "EXTRACTION_FAILED": {
        "message": "Failed to extract data from file",
        "suggestion": "Check file is not corrupted and is readable"
    }
}


# ============================================================================
# 2. RAISING ERRORS IN BACKEND CODE
# ============================================================================

# BACKEND: How to raise structured errors

from utils.errors import AuditError, ERROR_CODES

# In ingestion/bank_statement_parser.py:
def parse_csv(self, file_path: str) -> pd.DataFrame:
    """Parse CSV and validate."""
    csv_path = Path(file_path)
    dataframe = pd.read_csv(csv_path)
    bank_name = self.detect_bank(dataframe)
    normalized = self._normalize_dataframe(dataframe, bank_name)
    
    # Validate and raise structured error if issues found
    self.validate_parsed_df(normalized, file_name=csv_path.name)
    
    return normalized


# Example error raising:
def validate_parsed_df(self, df, file_name=None):
    """Validate dataframe quality."""
    if len(df) < 2:
        raise AuditError(
            message=ERROR_CODES["EMPTY_FILE"]["message"],
            error_code="EMPTY_FILE",
            file_name=file_name,
            suggestion=ERROR_CODES["EMPTY_FILE"]["suggestion"]
        )
    
    # Check for parsing issues...
    nat_percentage = (df["date"].isna().sum() / len(df)) * 100
    if nat_percentage > 20:
        raise AuditError(
            message=ERROR_CODES["DATE_PARSE_FAILED"]["message"],
            error_code="DATE_PARSE_FAILED",
            file_name=file_name,
            suggestion=ERROR_CODES["DATE_PARSE_FAILED"]["suggestion"]
        )


# ============================================================================
# 3. API ERROR RESPONSES (HTTP)
# ============================================================================

# When AuditError is raised in backend, FastAPI exception handler
# automatically converts it to HTTP response:

# Request that causes DATE_PARSE_FAILED:
# POST /upload (file with >20% unparseable dates)

# Response (HTTP 422):
{
    "error": "DATE_PARSE_FAILED",
    "message": "Could not parse dates in this file",
    "file": "statement.csv",
    "suggestion": "Check date format is DD/MM/YYYY or DD-MM-YYYY"
}

# For unexpected backend errors (HTTP 500):
{
    "error": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "detail": "ValueError: unable to process...",
    "suggestion": "Try again or contact support"
}


# ============================================================================
# 4. FRONTEND ERROR DISPLAY
# ============================================================================

# FRONTEND: Error handling in Streamlit app

# The api_call() function handles HTTP errors:

def api_call(endpoint, method="GET", data=None, files=None):
    """Make API call with error handling."""
    try:
        # Make request...
        response.raise_for_status()
        return response.json()
    
    except HTTPError as exc:
        payload = exc.response.json()
        
        # Handle AuditError (422) - Structured error with suggestion
        if exc.response.status_code == 422 and "error" in payload:
            show_structured_error(
                error_code=payload.get("error"),
                message=payload.get("message"),
                suggestion=payload.get("suggestion"),
                file_name=payload.get("file")
            )
            return None
        
        # Handle server error (500)
        if exc.response.status_code == 500:
            show_structured_error(
                error_code=payload.get("error", "INTERNAL_ERROR"),
                message=payload.get("message"),
                suggestion=payload.get("suggestion")
            )
            # Show technical details in collapsible box
            if payload.get("detail"):
                with st.expander("🔧 Technical Details"):
                    st.code(payload.get("detail"))
            return None


# show_structured_error() renders the error to user:
def show_structured_error(error_code, message, suggestion=None, file_name=None):
    """Display error with suggestion in styled boxes."""
    # Red error box with main message
    st.markdown(f'<div style="...">⚠️ {message}</div>')
    
    # Muted file name if provided
    if file_name:
        st.markdown(f'<div>File: {file_name}</div>')
    
    # Green suggestion box
    if suggestion:
        st.markdown(f'<div style="...">💡 Suggestion: {suggestion}</div>')


# ============================================================================
# 5. USER EXPERIENCE FLOW
# ============================================================================

# USER UPLOADS FILE WITH BROKEN DATES:
# 1. User selects statement_bad_dates.csv and clicks Upload
# 2. Frontend POST /upload with file
# 3. Backend ingests CSV
# 4. bank_statement_parser.validate_parsed_df() detects >20% NaT
# 5. Raises AuditError("DATE_PARSE_FAILED", suggestion="Check date format...")
# 6. FastAPI @exception_handler(AuditError) catches it
# 7. Returns HTTP 422 with structured JSON
# 8. Frontend detects 422 status
# 9. Calls show_structured_error()
# 10. User sees:
#     ┌─────────────────────────────────┐
#     │ ⚠️ Could not parse dates...     │
#     │ File: statement_bad_dates.csv   │
#     └─────────────────────────────────┘
#     ┌─────────────────────────────────┐
#     │ 💡 Suggestion:                  │
#     │ Check date format is DD/MM/YYYY │
#     └─────────────────────────────────┘


# ============================================================================
# 6. ADDING NEW ERROR CODES
# ============================================================================

# To add a new error code:

# 1. Add to utils/errors.py ERROR_CODES dict:
ERROR_CODES = {
    # ... existing codes ...
    "MY_NEW_ERROR": {
        "message": "User-facing message (what went wrong)",
        "suggestion": "Actionable fix (how to resolve)"
    }
}

# 2. Use in backend code:
from utils.errors import AuditError, ERROR_CODES

raise AuditError(
    message=ERROR_CODES["MY_NEW_ERROR"]["message"],
    error_code="MY_NEW_ERROR",
    file_name="optional_file.csv",
    suggestion=ERROR_CODES["MY_NEW_ERROR"]["suggestion"]
)

# 3. Test in test_error_handling.py:
def test_my_new_error():
    # ... code that triggers error ...
    with pytest.raises(AuditError) as exc_info:
        # ... code ...
    assert exc_info.value.error_code == "MY_NEW_ERROR"


# ============================================================================
# 7. ERROR HANDLING CHECKLIST
# ============================================================================

# ✓ Parsers validate data and raise specific AuditError
# ✓ All AuditError includes error_code, message, and suggestion
# ✓ FastAPI exception handlers catch and convert to JSON
# ✓ Frontend displays error with suggestion (never raw JSON)
# ✓ User sees actionable guidance, not technical details
# ✓ Unexpected errors logged but don't expose internals
# ✓ Test coverage for error paths
# ✓ Consistent error styling across frontend

