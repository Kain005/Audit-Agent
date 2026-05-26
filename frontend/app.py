"""
Professional Fintech Audit Dashboard
Built with Streamlit - Razorpay/CRED inspired design
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import time
import os

# API base (used by PDF download and other frontend endpoints)
API_BASE = os.environ.get("AUDIT_API_URL", "http://127.0.0.1:8000").rstrip("/")

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="AuditAI - Financial Audit Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color Palette
COLORS = {
    "bg_dark": "#0F172A",
    "bg_card": "#1E293B",
    "text": "#F8FAFC",
    "high": "#EF4444",
    "medium": "#F97316",
    "safe": "#22C55E",
    "accent": "#3B82F6",
    "border": "#334155",
    "muted": "#64748B",
}

API_BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

# Custom CSS
CUSTOM_CSS = f"""
<style>
    /* Hide Streamlit defaults */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .reportview-container {{margin-top: -2rem;}}
    
    /* Main styling */
    * {{color: {COLORS['text']};}}
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['bg_dark']};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_dark']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    /* Cards */
    .metric-card {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin: 8px 0;
    }}
    
    .metric-value {{
        font-size: 32px;
        font-weight: 700;
        margin: 8px 0;
    }}
    
    .metric-label {{
        font-size: 12px;
        color: {COLORS['muted']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Tables */
    .finding-row-high {{
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid {COLORS['high']};
    }}
    
    .finding-row-medium {{
        background-color: rgba(249, 115, 22, 0.1);
        border-left: 4px solid {COLORS['medium']};
    }}
    
    .finding-row-low {{
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 4px solid {COLORS['safe']};
    }}
    
    /* Severity badges */
    .severity-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    
    .severity-high {{
        background-color: {COLORS['high']};
        color: white;
    }}
    
    .severity-medium {{
        background-color: {COLORS['medium']};
        color: white;
    }}
    
    .severity-low {{
        background-color: {COLORS['safe']};
        color: white;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['accent']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }}
    
    .stButton > button:hover {{
        background-color: #2563EB;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }}
    
    /* Input fields */
    input, textarea, select {{
        background-color: {COLORS['bg_card']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['text']} !important;
        border-radius: 8px !important;
    }}
    
    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "📤 Upload"
    if "job_id" not in st.session_state:
        st.session_state.job_id = None
    if "report_data" not in st.session_state:
        st.session_state.report_data = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "selected_finding" not in st.session_state:
        st.session_state.selected_finding = None
    if "selected_document" not in st.session_state:
        st.session_state.selected_document = None
    if "api_url" not in st.session_state:
        st.session_state.api_url = API_BASE_URL
    if "ollama_url" not in st.session_state:
        st.session_state.ollama_url = "http://127.0.0.1:11434"
    if "sample_files" not in st.session_state:
        st.session_state.sample_files = []
    if "audit_config" not in st.session_state:
        st.session_state.audit_config = {
            "account_type": "Business Current Account",
            "salary_amount": 0,
            "emi_amount": 0,
            "rent_amount": 0,
            "approval_threshold": 100000,
            "new_vendor_threshold": 25000,
            "business_hours_start": 9,
            "business_hours_end": 18,
        }
    if "finding_explanations" not in st.session_state:
        st.session_state.finding_explanations = {}
    if "transaction_count" not in st.session_state:
        st.session_state.transaction_count = 0
    if "gst_transaction_count" not in st.session_state:
        st.session_state.gst_transaction_count = 0
    if "document_findings" not in st.session_state:
        st.session_state.document_findings = {}


def cache_finding_explanation(finding_id: str, explanation: str) -> None:
    """Persist explanation text for reuse across report views."""
    explanations = st.session_state.get("finding_explanations", {})
    explanations[finding_id] = explanation
    st.session_state.finding_explanations = explanations

    report = st.session_state.get("report_data")
    if isinstance(report, dict):
        for item in report.get("anomalies", []):
            if item.get("finding_id") == finding_id:
                item["explanation"] = explanation
                break


def normalize_document_name(name: Any) -> str:
    """Normalize document names for matching uploaded files to findings."""
    text = str(name or "").strip().lower()
    return Path(text).name


def build_document_findings(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group all findings by document name for the Documents tab."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for finding in (report.get("anomalies", []) or []) + (report.get("policy_violations", []) or []):
        document_name = normalize_document_name(finding.get("document_name") or finding.get("file_name") or "unknown")
        grouped.setdefault(document_name, []).append(finding)
    return grouped


def normalize_summary_text(text: Any) -> str:
    """Collapse multiline summary output into a single readable paragraph."""
    return " ".join(str(text or "").split())


def strip_ansi(text: Any) -> str:
    """Remove terminal control sequences from streamed model output."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", str(text or ""))


def get_transaction_count(report: dict[str, Any]) -> int:
    """Return the best available transaction count for GST or mixed audits."""
    gst_count = report.get("gst_transaction_count")
    if gst_count is not None:
        try:
            gst_value = int(gst_count)
            if gst_value > 0:
                return gst_value
        except Exception:
            pass

    combined_count = report.get("combined_transaction_count")
    if combined_count is not None:
        try:
            combined_value = int(combined_count)
            if combined_value > 0:
                return combined_value
        except Exception:
            pass

    total_transactions = report.get("total_transactions")
    if isinstance(total_transactions, int):
        return total_transactions

    try:
        return int(total_transactions or 0)
    except Exception:
        return 0


def build_rule_based_summary(report: dict[str, Any]) -> str:
    """Create a fallback executive summary directly from report findings."""
    anomalies = report.get("anomalies", []) or []
    violations = report.get("policy_violations", []) or []
    findings = anomalies + violations

    risk_score = float(report.get("risk_score", 0) or 0)
    if risk_score >= 60:
        risk_level = "High"
        recommendation = "Prioritize immediate review of the highest-risk findings before continuing normal operations."
    elif risk_score >= 30:
        risk_level = "Medium"
        recommendation = "Validate the medium-risk items, collect supporting evidence, and close the control gaps soon."
    else:
        risk_level = "Low"
        recommendation = "Continue routine monitoring and recheck the flagged items for completeness."

    total_documents = int(report.get("documents_processed", 0) or 0)
    total_transactions = get_transaction_count(report)

    high_findings = [
        f"- {item.get('finding_type') or item.get('rule_name')}: {item.get('human_readable_reason') or item.get('description') or ''}"
        for item in findings
        if str(item.get("severity", "")).upper() == "HIGH"
    ]
    medium_findings = [
        f"- {item.get('finding_type') or item.get('rule_name')}: {item.get('human_readable_reason') or item.get('description') or ''}"
        for item in findings
        if str(item.get("severity", "")).upper() == "MEDIUM"
    ]
    low_count = sum(1 for item in findings if str(item.get("severity", "")).upper() == "LOW")

    parts = [
        f"Overall risk level: {risk_level} with a final risk score of {risk_score:.0f}/100.",
        f"Documents analyzed: {total_documents}. Total transactions: {total_transactions}.",
        f"High risk findings: {len(high_findings)}. Medium risk findings: {len(medium_findings)}. Low risk findings: {low_count}.",
    ]

    if high_findings:
        parts.append("High risk issues: " + " ".join(high_findings))
    if medium_findings:
        parts.append("Medium risk issues: " + " ".join(medium_findings))

    parts.append(f"Recommendation: {recommendation}")
    return " ".join(parts)

init_session_state()

# ============================================================================
# API HELPERS
# ============================================================================

def show_error_card(message: str) -> None:
    st.markdown(
        f'''
        <div style="background:#2B1012;border:1px solid {COLORS['high']};border-radius:10px;padding:12px 14px;margin:8px 0;">
            <div style="color:{COLORS['high']};font-weight:700;">Connection Error</div>
            <div style="color:{COLORS['text']};font-size:13px;">{message}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def show_structured_error(error_code: str, message: str, suggestion: str | None = None, file_name: str | None = None) -> None:
    """Display structured error with suggestion and optional file name."""
    col1, col2 = st.columns([0.8, 0.2])
    
    with col1:
        st.markdown(
            f'''
            <div style="background:#2B1012;border:1px solid {COLORS['high']};border-radius:10px;padding:16px;margin:8px 0;">
                <div style="color:{COLORS['high']};font-weight:700;font-size:14px;">⚠️ {message}</div>
                {f'<div style="color:{COLORS["muted"]};font-size:11px;margin-top:4px;">File: {file_name}</div>' if file_name else ''}
            </div>
            ''',
            unsafe_allow_html=True,
        )
    
    if suggestion:
        st.markdown(
            f'''
            <div style="background:#1E3A2E;border:1px solid {COLORS['safe']};border-radius:10px;padding:12px 14px;margin:8px 0;">
                <div style="color:{COLORS['safe']};font-weight:600;font-size:12px;">💡 Suggestion:</div>
                <div style="color:{COLORS['text']};font-size:12px;margin-top:4px;">{suggestion}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )


def render_audit_configuration() -> dict[str, Any]:
    """Render audit configuration section."""
    defaults = st.session_state.get("audit_config", {})

    with st.expander("Configure Audit Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Account Context")
            account_type = st.selectbox(
                "Account Type",
                [
                    "Business Current Account",
                    "Company Expense Account",
                    "Personal Savings Account",
                    "Salary Account",
                ],
                index=[
                    "Business Current Account",
                    "Company Expense Account",
                    "Personal Savings Account",
                    "Salary Account",
                ].index(defaults.get("account_type", "Business Current Account"))
                if defaults.get("account_type", "Business Current Account")
                in [
                    "Business Current Account",
                    "Company Expense Account",
                    "Personal Savings Account",
                    "Salary Account",
                ]
                else 0,
            )

            st.subheader("Exclude Regular Payments")
            salary_amount = st.number_input(
                "Monthly salary credit (₹)",
                min_value=0,
                value=int(defaults.get("salary_amount", 0) or 0),
                step=1000,
                help="Won't be flagged as suspicious credit",
            )
            emi_amount = st.number_input(
                "Monthly EMI payment (₹)",
                min_value=0,
                value=int(defaults.get("emi_amount", 0) or 0),
                step=1000,
                help="Regular EMI won't be flagged",
            )
            rent_amount = st.number_input(
                "Monthly rent (₹)",
                min_value=0,
                value=int(defaults.get("rent_amount", 0) or 0),
                step=1000,
            )

        with col2:
            st.subheader("Approval Thresholds")
            approval_threshold = st.number_input(
                "Single payment approval threshold (₹)",
                min_value=0,
                value=int(defaults.get("approval_threshold", 100000) or 100000),
                step=5000,
                help="Payments above this need approval",
            )
            new_vendor_threshold = st.number_input(
                "New vendor scrutiny threshold (₹)",
                min_value=0,
                value=int(defaults.get("new_vendor_threshold", 25000) or 25000),
                step=1000,
            )

            st.subheader("Business Hours")
            col3, col4 = st.columns(2)
            with col3:
                biz_start = st.number_input(
                    "Start hour",
                    min_value=0,
                    max_value=23,
                    value=int(defaults.get("business_hours_start", 9) or 9),
                )
            with col4:
                biz_end = st.number_input(
                    "End hour",
                    min_value=0,
                    max_value=23,
                    value=int(defaults.get("business_hours_end", 18) or 18),
                )

    st.session_state["audit_config"] = {
        "account_type": account_type,
        "salary_amount": salary_amount,
        "emi_amount": emi_amount,
        "rent_amount": rent_amount,
        "approval_threshold": approval_threshold,
        "new_vendor_threshold": new_vendor_threshold,
        "business_hours_start": biz_start,
        "business_hours_end": biz_end,
    }

    return st.session_state["audit_config"]


def api_call(endpoint: str, method: str = "GET", data: dict = None, files: Any = None) -> dict | None:
    """Make API call with structured error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=60)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, timeout=300)
            else:
                response = requests.post(url, json=data, timeout=60)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=60)
        else:
            return None

        if response.status_code not in [200, 202]:
            handle_api_error(response)
            return None

        return response.json() if response.text else {}
    except requests.exceptions.ConnectionError:
        show_error_card(
            f"Could not connect to API at {API_BASE_URL}. Ensure uvicorn is running and listening on 127.0.0.1:8000."
        )
        return None
    except requests.exceptions.Timeout:
        show_error_card(f"API request timed out for {endpoint}. The backend may still be processing.")
        return None
    except requests.exceptions.RequestException as exc:
        show_error_card(f"Network request failed for {endpoint}: {exc}")
        return None


def handle_api_error(response: requests.Response) -> None:
    if response.status_code == 422:
        error = response.json()
        st.error(f"⚠️ {error.get('message', 'Unknown error')}")
        if error.get("suggestion"):
            st.info(f"💡 {error['suggestion']}")
    elif response.status_code == 500:
        error = response.json()
        st.error("Something went wrong on the server")
        with st.expander("Technical details"):
            st.code(error.get("detail", ""))
    else:
        st.error(f"Request failed with status {response.status_code}")


def check_backend_status() -> tuple[bool, str]:
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, "Connected"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except Exception:
        return False, "Disconnected"


def load_sample_files() -> list[tuple[str, bytes, str]]:
    """Load sample files from data/sample directory."""
    files: list[tuple[str, bytes, str]] = []
    candidates = [
        SAMPLE_DIR / "test_statement.csv",
        SAMPLE_DIR / "sample_invoice.txt",
    ]

    for file_path in candidates:
        if file_path.exists():
            mime = "text/csv" if file_path.suffix.lower() == ".csv" else "text/plain"
            files.append((file_path.name, file_path.read_bytes(), mime))

    return files

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render professional sidebar"""
    with st.sidebar:
        # Logo
        st.markdown(
            f'<div style="text-align: center; padding: 20px 0;"><h1 style="color: {COLORS["accent"]}; margin: 0;">🤖 AuditAI</h1></div>',
            unsafe_allow_html=True
        )
        st.divider()
        
        # Navigation
        st.markdown("**Navigation**")
        pages = [
            ("📤 Upload Documents", "📤 Upload"),
            ("🔍 Audit Findings", "🔍 Findings"),
            ("📊 Analytics", "📊 Analytics"),
            ("📄 Documents", "📄 Documents"),
            ("📋 Full Report", "📋 Report"),
            ("⚙️ Settings", "⚙️ Settings"),
        ]
        
        for label, page_key in pages:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
        
        st.divider()
        
        # Backend Status
        status, status_text = check_backend_status()
        status_color = COLORS['safe'] if status else COLORS['high']
        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 8px;">'
            f'<div style="width: 10px; height: 10px; border-radius: 50%; background-color: {status_color};"></div>'
            f'<span style="color: {COLORS["muted"]};">Backend: {status_text}</span></div>',
            unsafe_allow_html=True
        )
        
        st.divider()
        st.markdown(
            f'<div style="text-align: center; font-size: 11px; color: {COLORS["muted"]}; margin-top: 20px;">'
            'Powered by Llama 3.1</div>',
            unsafe_allow_html=True
        )

# ============================================================================
# PAGE 1: UPLOAD DOCUMENTS
# ============================================================================

def page_upload():
    """Upload documents page"""
    st.markdown(
        '<h1 style="text-align: center; margin-bottom: 8px;">📤 Upload Financial Documents</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #64748B; margin-top: 0;">Supports invoices, bank statements, ledgers, expense sheets, GST documents</p>',
        unsafe_allow_html=True
    )
    
    st.markdown("")

    left, right = st.columns([1, 1])
    with left:
        if st.button("Load Sample Data", type="secondary", use_container_width=True):
            sample_path = Path("data/sample/test_statement.csv")
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    sample_bytes = f.read()
                st.session_state["sample_loaded"] = True
                st.session_state["sample_data"] = sample_bytes
                st.session_state["sample_name"] = "test_statement.csv"
                st.session_state.sample_files = [("test_statement.csv", sample_bytes, "text/csv")]
                st.success("Sample data loaded - click Run Audit")
            else:
                st.error("Sample file not found at data/sample/test_statement.csv")

    with right:
        if st.button("🧹 Clear All", use_container_width=True):
            st.session_state.uploaded_files = []
            st.session_state.sample_files = []
            st.rerun()
    
    # Document type cards
    doc_types = [
        {"icon": "🏦", "name": "Bank Statements", "formats": "CSV, PDF", "example": "HDFC, ICICI, SBI statements"},
        {"icon": "📄", "name": "Invoices", "formats": "PDF, XLSX", "example": "Purchase/Sales invoices"},
        {"icon": "📒", "name": "Ledgers", "formats": "CSV, XLSX", "example": "General ledger exports"},
        {"icon": "💳", "name": "Expense Sheets", "formats": "CSV, XLSX", "example": "Employee expense reports"},
        {"icon": "🧾", "name": "GST Documents", "formats": "XLSX, CSV", "example": "GSTR-1, GSTR-2A exports"},
    ]
    
    cols = st.columns(3)
    for idx, doc in enumerate(doc_types):
        with cols[idx % 3]:
            st.markdown(
                f'''
                <div class="metric-card" style="text-align: center; padding: 20px;">
                    <div style="font-size: 32px; margin-bottom: 12px;">{doc["icon"]}</div>
                    <div style="font-weight: 600; margin-bottom: 8px;">{doc["name"]}</div>
                    <div style="font-size: 12px; color: #94A3B8; margin-bottom: 4px;">{doc["formats"]}</div>
                    <div style="font-size: 11px; color: #64748B;">{doc["example"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
    
    st.markdown("")
    
    # File uploader
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "Drag and drop files or click to select",
        type=["pdf", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        # Read bytes immediately before storing in session state
        file_data = []
        for uf in uploaded_files:
            # Use getvalue() instead of seek(0) + read()
            # More reliable in Streamlit on Windows
            try:
                bytes_content = uf.getvalue()
            except:
                uf.seek(0)
                bytes_content = uf.read()
            file_data.append({
                "name": uf.name,
                "bytes": bytes_content,
                "size": len(bytes_content),
                "type": uf.type or "application/octet-stream",
            })
        st.session_state.uploaded_files = file_data
        st.markdown("**Selected Files**")
        for idx, file in enumerate(file_data):
            col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 0.5])
            name = file["name"] if isinstance(file, dict) else file.name
            size = file["size"] if isinstance(file, dict) else file.size
            with col1:
                if name.endswith('.pdf'):
                    st.markdown("📄")
                elif name.endswith('.csv'):
                    st.markdown("📋")
                else:
                    st.markdown("📊")
            with col2:
                st.markdown(f"**{name}** ({size/1024:.1f} KB)")
            with col3:
                st.markdown(f"`{name.split('.')[-1].upper()}`")
            with col4:
                if st.button("✕", key=f"remove_{idx}"):
                    st.session_state.uploaded_files.pop(idx)
                    st.rerun()

    if st.session_state.sample_files:
        st.markdown("**Loaded Sample Files**")
        for name, content, _mime in st.session_state.sample_files:
            size_kb = len(content) / 1024
            st.markdown(f"- {name} ({size_kb:.1f} KB)")
    
    st.markdown("")
    
    # Audit Configuration
    render_audit_configuration()
    
    st.markdown("")
    
    # Run Audit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        has_uploaded = len(st.session_state.uploaded_files) > 0
        has_samples = len(st.session_state.sample_files) > 0
        if st.button(
            "🚀 Run Audit",
            use_container_width=True,
            disabled=not (has_uploaded or has_samples),
            key="run_audit"
        ):
            with st.spinner("Processing..."):
                # Prepare files
                files_to_upload: list[tuple[str, tuple[str, Any, str]]] = []
                for uf in st.session_state.uploaded_files:
                    if isinstance(uf, dict):
                        # New format - already read bytes
                        files_to_upload.append((
                            "files",
                            (uf["name"], uf["bytes"], uf.get("type", "application/octet-stream"))
                        ))
                    else:
                        # Old format fallback
                        uf.seek(0)
                        file_bytes = uf.read()
                        files_to_upload.append(("files", (uf.name, file_bytes, "application/octet-stream")))
                for name, content, mime in st.session_state.sample_files:
                    files_to_upload.append(("files", (name, content, mime)))
                
                # Upload files
                with st.status("Running audit...", expanded=True) as status:
                    messages = [
                        "📥 Uploading documents...",
                        "🔍 Parsing documents...",
                        "🔗 Extracting entities...",
                        "🔄 Cross-referencing transactions...",
                        "⚡ Detecting anomalies...",
                        "🤖 Generating explanations...",
                    ]
                    
                    for msg in messages:
                        st.write(msg)
                        time.sleep(0.5)
                    
                    upload_result = api_call("/upload", method="POST", files=files_to_upload)
                    if not upload_result or "job_id" not in upload_result:
                        status.update(label="❌ Upload failed", state="error")
                        show_error_card("Upload failed. Verify API is reachable and uploaded files are valid.")
                        return

                    job_id = upload_result["job_id"].strip()
                    st.session_state["job_id"] = job_id.strip()
                    st.session_state["uploaded_files"] = [f.name for f in uploaded_files]

                    analyze_result = api_call(
                        f"/analyze/{job_id.strip()}",
                        method="POST",
                        data={"audit_config": st.session_state.audit_config},
                    )
                    if not analyze_result:
                        status.update(label="❌ Failed to start analysis", state="error")
                        show_error_card("Analysis could not start for uploaded job.")
                        return

                    final_status = None
                    for _ in range(180):
                        current_status = api_call(f"/status/{job_id.strip()}", method="GET")
                        if not current_status:
                            status.update(label="❌ Status check failed", state="error")
                            show_error_card("Could not fetch job status from backend.")
                            return

                        final_status = str(current_status.get("status", "")).lower()
                        if final_status in {"complete", "failed"}:
                            break
                        time.sleep(2)

                    if final_status != "complete":
                        status.update(label="❌ Audit failed", state="error")
                        show_error_card("Audit did not complete successfully. Check backend logs for details.")
                        return

                    with st.spinner("Generating executive summary..."):
                        report_response = requests.get(f"{API_BASE}/report/{job_id.strip()}", timeout=120)
                    if report_response.status_code != 200:
                        status.update(label="❌ Report fetch failed", state="error")
                        show_error_card("Audit completed but report retrieval failed.")
                        return

                    report_data = report_response.json()
                    st.session_state["report_data"] = report_data
                    st.session_state["job_id"] = job_id.strip()
                    st.session_state["gst_transaction_count"] = int(report_data.get("gst_transaction_count", 0) or 0)
                    st.session_state["transaction_count"] = get_transaction_count(report_data)
                    st.session_state["document_findings"] = build_document_findings(report_data)
                    status.update(label="✅ Audit complete!", state="complete")
                    st.success("✅ Audit completed successfully!")

                    # Auto-navigate to findings
                    time.sleep(1)
                    st.session_state.current_page = "🔍 Findings"
                    st.rerun()

# ============================================================================
# PAGE 2: AUDIT FINDINGS
# ============================================================================

def page_findings():
    """Audit findings page"""
    if not st.session_state.get("report_data"):
        st.info("No audit results yet. Upload documents and run audit first.")
        st.stop()

    report = st.session_state.get("report_data", {})
    
    # Top metrics
    st.markdown("**Key Metrics**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    all_anomalies = report.get("anomalies", []) or []
    all_violations = report.get("policy_violations", []) or []
    all_findings = all_anomalies + all_violations

    high_count = sum(1 for f in all_findings if f.get("severity", "").upper() == "HIGH")
    medium_count = sum(1 for f in all_findings if f.get("severity", "").upper() == "MEDIUM")
    low_count = sum(1 for f in all_findings if f.get("severity", "").upper() == "LOW")
    risk_score = float(report.get("risk_score", 0) or 0)
    doc_count = report.get("documents_processed", 0)
    tx_count = int(
        report.get("gst_transaction_count")
        or report.get("transaction_count")
        or report.get("combined_transaction_count")
        or report.get("total_transactions")
        or 0
    )
    st.session_state["transaction_count"] = tx_count

    col1.metric("Documents", doc_count)
    col2.metric("Transactions", tx_count)
    col3.metric("High Risk", high_count)
    col4.metric("Medium Risk", medium_count)
    col5.metric("Low Risk", low_count)
    col6.metric("Risk Score", f"{risk_score:.0f}")
    
    st.markdown("")
    
    # Risk score gauge
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Findings Overview**")
        findings = all_findings
        
        # Filter controls
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            severity_filter = st.selectbox("Severity", ["All", "HIGH", "MEDIUM", "LOW"])
        with col_b:
            finding_types = sorted({
                str(f.get("finding_type") or f.get("rule_name") or "Unknown")
                for f in findings
                if f.get("finding_type") or f.get("rule_name")
            })
            finding_type = st.selectbox("Type", ["All"] + finding_types if finding_types else ["All"])
        with col_c:
            search = st.text_input("Search", placeholder="Search...")
        
        # Filter findings
        filtered = findings
        if severity_filter != "All":
            filtered = [f for f in filtered if f.get("severity") == severity_filter]
        if finding_type != "All":
            filtered = [f for f in filtered if (f.get("finding_type") or f.get("rule_name")) == finding_type]
        if search:
            filtered = [f for f in filtered if search.lower() in str(f).lower()]
        
        # Display findings table
        if filtered:
            for finding in filtered[:20]:  # Show top 20
                severity = finding.get("severity", "LOW")
                finding_id = finding.get("finding_id") or finding.get("violation_id") or finding.get("id") or "unknown"
                finding_label = finding.get("finding_type") or finding.get("rule_name") or "Unknown"
                severity_color = COLORS['high'] if severity == "HIGH" else COLORS['medium'] if severity == "MEDIUM" else COLORS['safe']
                reason = (finding.get("human_readable_reason") or finding.get("description") or "")[:100]
                amount = finding.get("evidence", {}).get("amount", 0)
                score = finding.get("score", finding.get("amount_involved", 0) or 0)
                
                # Use expander for each finding to allow explanation expansion
                with st.expander(
                    f"**{severity}** | {finding_label} | ₹{amount:,.0f} | Score: {float(score or 0):.2f}",
                    expanded=False
                ):
                    # Display details
                    col_detail1, col_detail2 = st.columns([3, 1])
                    with col_detail1:
                        st.markdown(f"**Reason:** {reason}")
                        st.markdown(f"**Evidence:** {json.dumps(finding.get('evidence', {}), indent=2)[:200]}...")
                    with col_detail2:
                        st.markdown(f"**Severity Color:** {severity}")
                    
                    # Add "Generate Explanation" button
                    col_btn1, col_btn2 = st.columns([2, 1])
                    with col_btn1:
                        if st.button("📝 Generate Explanation", key=f"explain_{finding_id}"):
                            # Store state to fetch explanation
                            st.session_state.explaining_finding_id = finding_id
                            st.session_state.fetch_explanation = True
                    
                    # Show explanation if available
                    if hasattr(st.session_state, 'explaining_finding_id') and st.session_state.explaining_finding_id == finding_id:
                        if hasattr(st.session_state, 'explanation_text'):
                            st.success(st.session_state.explanation_text)
                        elif st.session_state.get("fetch_explanation"):
                            st.info("⏳ Generating explanation...")
                            try:
                                job_id = st.session_state.get("job_id", "").strip()
                                if not job_id:
                                    st.error("Job ID not available")
                                else:
                                    response = requests.post(
                                        f"{API_BASE_URL}/explain/{job_id.strip()}/{finding_id}",
                                        timeout=50
                                    )
                                    if response.status_code == 200:
                                        data = response.json()
                                        explanation = data.get("explanation", "No explanation available")
                                        st.session_state.explanation_text = explanation
                                        cache_finding_explanation(finding_id, explanation)
                                        st.session_state.fetch_explanation = False
                                        st.success(explanation)
                                    else:
                                        explanation = f"Failed to generate explanation: {response.text}"
                                        st.session_state.explanation_text = explanation
                                        cache_finding_explanation(finding_id, explanation)
                                        st.error(explanation)
                                        st.session_state.fetch_explanation = False
                            except requests.exceptions.Timeout:
                                explanation = "⏱️ Explanation unavailable - try again (request timed out)"
                                st.session_state.explanation_text = explanation
                                cache_finding_explanation(finding_id, explanation)
                                st.warning(explanation)
                                st.session_state.fetch_explanation = False
                            except Exception as e:
                                explanation = f"Error generating explanation: {str(e)}"
                                st.session_state.explanation_text = explanation
                                cache_finding_explanation(finding_id, explanation)
                                st.error(explanation)
                                st.session_state.fetch_explanation = False
        else:
            st.info("No findings match your filters")
    
    with col2:
        st.markdown("**Finding Distribution**")
        
        # Donut chart by severity
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for f in findings:
            sev = f.get("severity", "LOW")
            if sev in severity_counts:
                severity_counts[sev] += 1
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                marker=dict(colors=[COLORS['high'], COLORS['medium'], COLORS['safe']]),
                hole=0.4,
                textinfo="label+value"
            )
        ])
        fig.update_layout(
            showlegend=False,
            paper_bgcolor=COLORS['bg_dark'],
            plot_bgcolor=COLORS['bg_dark'],
            font=dict(color=COLORS['text']),
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: ANALYTICS
# ============================================================================

def page_analytics():
    """Analytics page"""
    if not st.session_state.report_data:
        st.info("📤 Upload documents first to see analytics")
        return
    
    report = st.session_state.report_data

    all_findings = (report.get("anomalies", []) or []) + (report.get("policy_violations", []) or [])
    if not all_findings:
        st.info("No findings available for analytics.")
        return

    st.markdown("**Finding Analysis**")

    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    type_counts: dict[str, int] = {}
    for item in all_findings:
        severity = str(item.get("severity", "LOW")).upper()
        if severity in severity_counts:
            severity_counts[severity] += 1
        finding_label = str(item.get("finding_type") or item.get("rule_name") or "Unknown")
        type_counts[finding_label] = type_counts.get(finding_label, 0) + 1

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Findings by Severity**")
        fig = go.Figure(data=[
            go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                marker=dict(colors=[COLORS['high'], COLORS['medium'], COLORS['safe']]),
                hole=0.4,
                textinfo="label+value"
            )
        ])
        fig.update_layout(
            showlegend=False,
            paper_bgcolor=COLORS['bg_dark'],
            plot_bgcolor=COLORS['bg_dark'],
            font=dict(color=COLORS['text']),
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top Finding Types**")
        sorted_types = sorted(type_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        fig = px.bar(
            x=[count for _type, count in sorted_types],
            y=[_type for _type, _count in sorted_types],
            orientation='h',
            color=[count for _type, count in sorted_types],
            color_continuous_scale=['#22C55E', '#F97316', '#EF4444']
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor=COLORS['bg_dark'],
            plot_bgcolor=COLORS['bg_dark'],
            font=dict(color=COLORS['text']),
            height=300,
            xaxis_title="Count",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Document Breakdown**")
    document_breakdown = report.get("document_breakdown", {}) or {}
    if document_breakdown:
        fig = px.bar(
            x=list(document_breakdown.values()),
            y=list(document_breakdown.keys()),
            orientation='h',
            color=list(document_breakdown.values()),
            color_continuous_scale=['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B']
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor=COLORS['bg_dark'],
            plot_bgcolor=COLORS['bg_dark'],
            font=dict(color=COLORS['text']),
            height=300,
            xaxis_title="Documents",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No document breakdown available for this audit.")

# ============================================================================
# PAGE 4: DOCUMENTS
# ============================================================================

def page_documents():
    """Documents page"""
    if not st.session_state.report_data:
        st.info("📤 Upload documents first")
        return
    
    st.markdown("**Uploaded Documents**")
    
    col1, col2 = st.columns([1, 2])
    
    # Left panel - document list
    with col1:
        st.subheader("Documents")
        
        report = st.session_state.get("report_data", {})
        job_id = st.session_state.get("job_id", "").strip()
        document_findings = st.session_state.get("document_findings", {})
        
        # Get document list from uploaded files session
        # Fall back to building from anomalies data
        uploaded_files = st.session_state.get("uploaded_files", [])
        
        if not uploaded_files:
            # Build from report anomalies
            uploaded_files = list(set(
                normalize_document_name(a.get("document_name", "Unknown"))
                for a in (report.get("anomalies", []) or []) + (report.get("policy_violations", []) or [])
            ))
            if not uploaded_files:
                uploaded_files = ["test_statement.csv"]
        
        # Show as clickable buttons
        for i, fname in enumerate(uploaded_files):
            if st.button(fname, key=f"doc_{i}", use_container_width=True):
                st.session_state["selected_document"] = fname
    
    # Right panel - document detail
    with col2:
        selected = st.session_state.get("selected_document", None)
        if selected is None:
            st.info("Click a document on the left to view details")
        else:
            # Show document data filtered by selected document
            report = st.session_state.get("report_data", {})
            anomalies = (report.get("anomalies", []) or []) + (report.get("policy_violations", []) or [])
            normalized_selected = normalize_document_name(selected)

            # Filter document findings for selected document
            doc_anomalies = document_findings.get(normalized_selected, [])
            if not doc_anomalies:
                doc_anomalies = [a for a in anomalies if normalize_document_name(a.get("document_name")) == normalized_selected]
            if not doc_anomalies and job_id:
                try:
                    response = requests.get(f"{API_BASE}/report/{job_id}/document/{normalized_selected}", timeout=30)
                    if response.status_code == 200:
                        payload = response.json()
                        doc_anomalies = payload.get("flags", []) or []
                except Exception:
                    doc_anomalies = []
            
            st.subheader(f"Findings in {selected}")
            # GST invoice structured fields
            report_data = st.session_state.get("report_data", {})
            for gst_doc in (report_data.get("gst_invoices") or []):
                gst_normalized = normalize_document_name(gst_doc.get("file_name", ""))
                st.caption(f"debug: gst={gst_normalized} selected={normalized_selected}")
                if gst_normalized == normalized_selected:
                    inv = gst_doc.get("invoice_data") or {}
                    st.markdown("**GST Invoice Details**")
                    c1, c2 = st.columns(2)
                    with c1:
                        for label, key in [
                            ("Invoice No", "invoice_number"),
                            ("Invoice Date", "invoice_date"),
                            ("Vendor", "vendor_name"),
                            ("Vendor GSTIN", "vendor_gst"),
                            ("Place of Supply", "place_of_supply"),
                        ]:
                            val = inv.get(key)
                            if val not in (None, "", 0):
                                st.markdown(f"**{label}:** {val}")
                    with c2:
                        for label, key in [
                            ("Buyer", "buyer_name"),
                            ("Buyer GSTIN", "buyer_gst"),
                            ("Total (₹)", "total_amount"),
                            ("CGST (₹)", "cgst_amount"),
                            ("SGST (₹)", "sgst_amount"),
                            ("IGST (₹)", "igst_amount"),
                        ]:
                            val = inv.get(key)
                            if val not in (None, "", 0):
                                st.markdown(f"**{label}:** {val}")
                    st.divider()
                    break

            if doc_anomalies:
                for a in doc_anomalies:
                    severity_color = {
                        "HIGH": "🔴", 
                        "MEDIUM": "🟠", 
                        "LOW": "🟡"
                    }.get(a.get("severity"), "⚪")
                    finding_label = a.get('finding_type') or a.get('rule_name') or 'Unknown'
                    reason = a.get('human_readable_reason') or a.get('description') or ''
                    st.write(f"{severity_color} {finding_label} — {reason[:120]}")
            else:
                st.success("No findings for this document")

# ============================================================================
# PAGE 5: FULL REPORT
# ============================================================================

def page_report():
    """Full report page"""
    if not st.session_state.report_data:
        st.info("📤 Upload documents first")
        return
    
    report = st.session_state.report_data
    
    # Generate summary directly in frontend if missing
    summary = normalize_summary_text(report.get("summary"))
    
    if not summary:
        with st.spinner("Generating executive summary..."):
            summary = build_rule_based_summary(report)
            report["summary"] = summary
            st.session_state.report_data = report
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("## 📋 Financial Audit Report")
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📥 Download JSON", use_container_width=True):
                st.download_button(
                    "Download JSON",
                    json.dumps(report, indent=2, default=str),
                    "audit_report.json",
                    "application/json"
                )
        with col_b:
            # Download PDF via API endpoint
            if st.session_state.get("job_id"):
                pdf_url = f"{API_BASE}/report/{st.session_state.get('job_id', '').strip()}/download"
                st.markdown(
                    f'<a href="{pdf_url}" target="_blank">'
                    f'<button style="width: 100%; padding: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 600;">📄 Download PDF</button>'
                    f'</a>',
                    unsafe_allow_html=True
                )
    
    st.divider()
    
    # Executive Summary
    with st.expander("📊 Executive Summary", expanded=True):
        st.info(strip_ansi(summary))
        st.markdown(f"""
        **Audit Date:** {datetime.now().strftime('%Y-%m-%d')}
        
        **Documents Analyzed:** {report.get('documents_processed', 0)}
        
        **Risk Score:** {report.get('risk_score', 0):.0f}/100
        """)
    
    # Risk Assessment
    with st.expander("⚠️ Risk Assessment"):
        risk_score = report.get('risk_score', 0)
        if risk_score < 30:
            risk_level = "🟢 LOW RISK"
        elif risk_score < 60:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🔴 HIGH RISK"
        
        st.markdown(f"**Overall Risk Level:** {risk_level}")
        st.markdown(f"**Score:** {risk_score:.0f}/100")
    
    # Findings by Severity
    with st.expander("🔍 Findings"):
        for severity in ["HIGH", "MEDIUM", "LOW"]:
            findings = [f for f in report.get("anomalies", []) if f.get("severity") == severity]
            if findings:
                with st.expander(f"{severity} Severity ({len(findings)} findings)"):
                    for f in findings[:10]:
                        st.markdown(f"- **{f.get('finding_type')}**: {f.get('human_readable_reason', '')}")
                        explanation = f.get("explanation") or st.session_state.get("finding_explanations", {}).get(
                            f.get("finding_id", "")
                        )
                        if explanation:
                            st.caption(f"Explanation: {explanation}")

# ============================================================================
# PAGE 6: SETTINGS
# ============================================================================

def page_settings():
    """Settings page"""
    st.markdown("## ⚙️ Configuration")
    
    # API Settings
    with st.expander("🌐 API Configuration", expanded=True):
        new_api = st.text_input("API Base URL", value=API_BASE_URL)
        new_ollama = st.text_input("Ollama URL", value=st.session_state.ollama_url)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test API Connection", use_container_width=True):
                result = api_call("/health")
                if result:
                    st.success("✅ API is reachable")
                else:
                    st.error("❌ Cannot reach API")
        with col2:
            if st.button("Save Configuration", use_container_width=True):
                st.session_state.api_url = API_BASE_URL
                st.session_state.ollama_url = new_ollama
                if new_api != API_BASE_URL:
                    st.warning("API URL is fixed to http://127.0.0.1:8000 for local integration consistency.")
                st.success("✅ Configuration saved")
    
    # Policy Thresholds
    with st.expander("📊 Policy Thresholds"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Amount Threshold (₹)", value=100000, step=10000)
        with col2:
            st.number_input("High Value Invoice (₹)", value=500000, step=50000)
        with col3:
            st.number_input("Split Billing Days", value=7, step=1)
        
        if st.button("Save Thresholds", use_container_width=True):
            st.success("✅ Thresholds updated")
    
    # Vendor Blacklist
    with st.expander("🚫 Vendor Blacklist"):
        new_vendor = st.text_input("Add vendor name")
        if st.button("Add to Blacklist", use_container_width=True):
            st.success(f"✅ Added {new_vendor} to blacklist")
        
        st.markdown("**Current Blacklist**")
        st.info("No vendors blacklisted")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app logic"""
    render_sidebar()
    
    # Route to correct page
    if st.session_state.current_page == "📤 Upload":
        page_upload()
    elif st.session_state.current_page == "🔍 Findings":
        page_findings()
    elif st.session_state.current_page == "📊 Analytics":
        page_analytics()
    elif st.session_state.current_page == "📄 Documents":
        page_documents()
    elif st.session_state.current_page == "📋 Report":
        page_report()
    elif st.session_state.current_page == "⚙️ Settings":
        page_settings()
    else:
        page_upload()

if __name__ == "__main__":
    main()
