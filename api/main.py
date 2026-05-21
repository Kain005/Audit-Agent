"""FastAPI service for file upload, audit processing, and report retrieval."""

from __future__ import annotations

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from .database import AuditJob, SessionLocal, create_tables, get_db
from utils.errors import AuditError, ERROR_CODES

try:
    from ..agent.audit_agent import run_audit
    from ..agent.explainer import Explainer
    from .report_generator import ReportGenerator
    from ..extraction.models import AuditReport
except ImportError:  # pragma: no cover - fallback for script-style execution
    from agent.audit_agent import run_audit
    from agent.explainer import Explainer
    from api.report_generator import ReportGenerator
    from extraction.models import AuditReport

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Finance Audit Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers for structured error responses
@app.exception_handler(AuditError)
async def audit_error_handler(request: Request, exc: AuditError) -> JSONResponse:
    """Handle AuditError exceptions with structured response."""
    return JSONResponse(
        status_code=422,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "file": exc.file_name,
            "suggestion": exc.suggestion,
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with safe error details."""
    import traceback

    LOGGER.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "detail": str(exc),
            "suggestion": "Check that all files are valid and try again",
        },
    )


PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
CHROMA_ROOT = PROJECT_ROOT / "data" / "chroma_db"

ALLOWED_SUFFIXES = {".pdf", ".csv", ".xlsx", ".xls", ".txt"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434"
ALLOWED_FINDING_SEVERITIES = {"HIGH", "MEDIUM", "LOW"}

executor = ThreadPoolExecutor(max_workers=2)
job_progress: dict[str, dict[str, Any]] = {}


class OllamaTestRequest(BaseModel):
    url: str


class AuditConfig(BaseModel):
    """Configuration for audit run."""
    account_type: str = "Business Current Account"
    salary_amount: float = 0
    emi_amount: float = 0
    rent_amount: float = 0
    approval_threshold: float = 100000
    new_vendor_threshold: float = 25000
    business_hours_start: int = 9
    business_hours_end: int = 18


class AnalyzeRequest(BaseModel):
    """Optional analyze request payload."""
    audit_config: dict[str, Any] | None = None


def _error_payload(error: str, detail: str, status: int) -> dict[str, Any]:
    return {"error": error, "detail": detail, "status": status}


def _status_payload(job: AuditJob) -> dict[str, Any]:
    status = job.status.lower()
    progress_map = {
        "pending": 0,
        "processing": 50,
        "complete": 100,
        "failed": 100,
    }
    step_map = {
        "pending": "queued",
        "processing": "analysis",
        "complete": "done",
        "failed": "failed",
    }
    return {
        "job_id": job.id,
        "status": job.status,
        "current_step": step_map.get(status, "unknown"),
        "progress_percent": progress_map.get(status, 0),
    }


def update_job_status(job_id: str, status: str, step: str, progress: int) -> None:
    """Update in-memory progress map and persist coarse status in DB."""
    progress_value = max(0, min(100, int(progress)))
    job_progress[job_id] = {
        "status": status,
        "step": step,
        "progress": progress_value,
        "updated_at": datetime.utcnow().isoformat(),
    }

    db = SessionLocal()
    try:
        job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
        if job is not None:
            job.status = status
            if status in {"complete", "failed"}:
                job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


def save_report(job_id: str, report: AuditReport) -> None:
    """Persist completed report for a job and mark complete."""
    db = SessionLocal()
    try:
        job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
        if job is None:
            raise RuntimeError("Job not found while saving report")

        report_dict = json.loads(report.model_dump_json())

        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(i) for i in obj]
            elif isinstance(obj, bool):
                return int(obj)
            elif isinstance(obj, float):
                import math
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
                return obj
            elif hasattr(obj, "__dict__"):
                return sanitize_for_json(obj.__dict__)
            return obj

        report_dict = sanitize_for_json(report_dict)
        job.report_json = json.dumps(report_dict)
        job.status = "complete"
        job.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


async def run_audit_background(job_id: str, file_paths: list[str], audit_config: dict[str, Any] | None = None) -> None:
    """Run audit in thread pool to avoid blocking request/background event loop."""
    try:
        update_job_status(job_id, "processing", "analysis", 50)

        loop = asyncio.get_event_loop()

        def _progress_callback(step: str, progress: int) -> None:
            update_job_status(job_id, "processing", step, progress)

        report = await loop.run_in_executor(
            executor,
            lambda: run_audit(file_paths, progress_callback=_progress_callback, audit_config=audit_config),
        )

        save_report(job_id, report)
        update_job_status(job_id, "complete", "done", 100)
    except Exception as exc:
        LOGGER.exception("Audit failed for job_id=%s: %s", job_id, exc)
        update_job_status(job_id, "failed", str(exc), 0)


def _require_job(db: Session, job_id: str) -> AuditJob:
    job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _require_report_json(job: AuditJob) -> str:
    if job.status != "complete" or not job.report_json:
        raise HTTPException(status_code=404, detail="Report not available")
    return job.report_json


@lru_cache(maxsize=128)
def _cached_report_payload(report_json: str) -> dict[str, Any]:
    return json.loads(report_json)


def _load_report_payload(report_json: str) -> dict[str, Any]:
    try:
        parsed = _cached_report_payload(report_json)
        return copy.deepcopy(parsed)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Stored report is invalid JSON: {exc}") from exc


@lru_cache(maxsize=128)
def _cached_findings(report_json: str) -> list[dict[str, Any]]:
    payload = _cached_report_payload(report_json)
    findings: list[dict[str, Any]] = []

    for anomaly in payload.get("anomalies", []):
        findings.append(
            {
                "id": anomaly.get("finding_id"),
                "source": "anomaly",
                "severity": str(anomaly.get("severity", "LOW")).upper(),
                "finding_type": anomaly.get("finding_type"),
                "document_name": anomaly.get("document_name"),
                "amount": (anomaly.get("evidence", {}) or {}).get("amount"),
                "description": anomaly.get("human_readable_reason"),
                "evidence": anomaly.get("evidence", {}),
                "transaction_ids": anomaly.get("transaction_ids", []),
                "raw": anomaly,
            }
        )

    for violation in payload.get("policy_violations", []):
        findings.append(
            {
                "id": violation.get("violation_id"),
                "source": "policy_violation",
                "severity": str(violation.get("severity", "LOW")).upper(),
                "finding_type": violation.get("rule_name"),
                "document_name": violation.get("document_name"),
                "amount": violation.get("amount_involved"),
                "description": violation.get("description"),
                "evidence": violation.get("evidence", {}),
                "transaction_ids": (violation.get("evidence", {}) or {}).get("transaction_references", []),
                "raw": violation,
            }
        )

    return findings


@lru_cache(maxsize=128)
def _cached_summary(report_json: str) -> dict[str, Any]:
    payload = _cached_report_payload(report_json)
    findings = _cached_findings(report_json)
    high_count = sum(1 for item in findings if item.get("severity") == "HIGH")
    medium_count = sum(1 for item in findings if item.get("severity") == "MEDIUM")
    low_count = sum(1 for item in findings if item.get("severity") == "LOW")

    concerns = payload.get("top_3_concerns")
    if not isinstance(concerns, list):
        severity_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        sorted_findings = sorted(
            findings,
            key=lambda item: (
                severity_rank.get(str(item.get("severity", "")).upper(), 0),
                float(item.get("amount") or 0),
            ),
            reverse=True,
        )
        concerns = [
            {
                "type": item.get("finding_type"),
                "severity": item.get("severity"),
                "description": item.get("description"),
            }
            for item in sorted_findings[:3]
        ]

    return {
        "risk_score": float(payload.get("risk_score", 0.0)),
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "doc_count": int(payload.get("documents_processed", 0)),
        "transaction_count": int(payload.get("total_transactions", 0)),
        "top_3_concerns": concerns,
    }


def _extract_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except Exception:
        pass

    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


@lru_cache(maxsize=128)
def _cached_analytics(report_json: str) -> dict[str, Any]:
    findings = _cached_findings(report_json)
    spending_by_category: dict[str, float] = {}
    vendor_totals: dict[str, dict[str, Any]] = {}
    timeline_map: dict[str, dict[str, Any]] = {}
    anomaly_scatter: list[dict[str, Any]] = []

    for item in findings:
        evidence = item.get("evidence", {}) or {}
        amount = _as_float(item.get("amount") or evidence.get("invoice_value") or evidence.get("tax_amount"))

        category = str(evidence.get("category") or "uncategorized")
        spending_by_category[category] = round(spending_by_category.get(category, 0.0) + amount, 2)

        vendor = str(evidence.get("vendor") or evidence.get("vendor_name") or item.get("document_name") or "unknown")
        vendor_entry = vendor_totals.setdefault(
            vendor,
            {"vendor": vendor, "total_amount": 0.0, "transaction_count": 0},
        )
        vendor_entry["total_amount"] = round(float(vendor_entry["total_amount"]) + amount, 2)
        vendor_entry["transaction_count"] = int(vendor_entry["transaction_count"]) + 1

        date_value = _extract_date(evidence.get("date") or evidence.get("invoice_date") or evidence.get("expense_date"))
        if date_value:
            day_entry = timeline_map.setdefault(
                date_value,
                {"date": date_value, "amount": 0.0, "transaction_count": 0},
            )
            day_entry["amount"] = round(float(day_entry["amount"]) + amount, 2)
            day_entry["transaction_count"] = int(day_entry["transaction_count"]) + 1

            if item.get("source") == "anomaly":
                anomaly_scatter.append(
                    {
                        "date": date_value,
                        "amount": amount,
                        "severity": item.get("severity"),
                        "description": item.get("description"),
                    }
                )

    vendor_concentration = sorted(
        vendor_totals.values(),
        key=lambda entry: float(entry.get("total_amount", 0.0)),
        reverse=True,
    )
    timeline = sorted(timeline_map.values(), key=lambda entry: str(entry.get("date", "")))

    return {
        "spending_by_category": spending_by_category,
        "vendor_concentration": vendor_concentration,
        "timeline": timeline,
        "anomaly_scatter": anomaly_scatter,
    }


def _ollama_status(url: str) -> tuple[bool, bool, int]:
    start = time.perf_counter()
    endpoint = f"{url.rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=5)
        elapsed = int((time.perf_counter() - start) * 1000)
        if response.status_code != 200:
            return (False, False, elapsed)

        payload = response.json() if response.content else {}
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return (True, bool(models), elapsed)
    except Exception:
        elapsed = int((time.perf_counter() - start) * 1000)
        return (False, False, elapsed)


@lru_cache(maxsize=32)
def _cached_pdf_bytes(report_json: str) -> bytes:
    """Cache rendered report PDF bytes for repeated download requests."""
    payload = _cached_report_payload(report_json)
    report = AuditReport.model_validate(payload)
    generator = ReportGenerator()
    return generator.generate_pdf(report)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code if response is not None else 500
        LOGGER.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(error="http_error", detail=detail, status=exc.status_code),
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(_: Request, exc: StarletteHTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(error="http_error", detail=detail, status=exc.status_code),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_error_payload(error="validation_error", detail=json.dumps(exc.errors()), status=422),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    LOGGER.exception("Unhandled API exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=_error_payload(error="internal_error", detail="Internal server error", status=500),
    )


@app.on_event("startup")
async def startup_event():
    from pathlib import Path

    dirs = [
        Path("data"),
        Path("data/uploads"),
        Path("data/sample"),
        Path("models"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    create_tables()

    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
    except:
        pass


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    if not files:
        raise HTTPException(status_code=422, detail="No files uploaded")

    upload_id = str(uuid.uuid4())
    upload_dir = UPLOAD_ROOT / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    try:
        for incoming in files:
            suffix = Path(incoming.filename or "").suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported file type for '{incoming.filename}'. Allowed: PDF, CSV, XLSX, XLS, TXT",
                )

            content = await incoming.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=422,
                    detail=f"File '{incoming.filename}' exceeds 10MB limit",
                )

            safe_name = Path(incoming.filename or "uploaded_file").name
            output_path = upload_dir / safe_name
            output_path.write_bytes(content)
            saved_paths.append(str(output_path))

        job = AuditJob(
            id=upload_id,
            status="pending",
            file_paths=json.dumps(saved_paths),
            created_at=datetime.utcnow(),
            completed_at=None,
            report_json=None,
        )
        db.add(job)
        db.commit()

        update_job_status(upload_id, "pending", "queued", 0)

        return {
            "job_id": upload_id,
            "file_count": len(saved_paths),
            "message": "Files uploaded and audit job created",
        }

    except Exception:
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        raise


@app.post("/analyze/{job_id}")
def analyze_job(
    job_id: str,
    request_body: AnalyzeRequest | None = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "processing":
        return {"job_id": job_id, "status": job.status, "message": "Job already processing"}

    if job.status == "complete":
        return {"job_id": job_id, "status": job.status, "message": "Job already complete"}

    job.status = "processing"
    db.commit()

    file_paths = job.get_file_paths()
    update_job_status(job_id, "processing", "analysis", 50)

    config_dict = request_body.audit_config if request_body and isinstance(request_body.audit_config, dict) else {}

    background_tasks.add_task(run_audit_background, job_id, file_paths, config_dict)
    return {"job_id": job_id, "status": "processing", "message": "Audit started in background"}


@app.get("/status/{job_id}", status_code=200)
def get_status(job_id: str, db: Session = Depends(get_db)):
    job = _require_job(db, job_id)
    progress = job_progress.get(job_id)
    if progress is None:
        status_payload = _status_payload(job)
        progress = {
            "status": status_payload["status"],
            "step": status_payload["current_step"],
            "progress": status_payload["progress_percent"],
        }
        job_progress[job_id] = progress

    return {
        "job_id": job_id,
        "status": progress.get("status", "unknown"),
        "current_step": progress.get("step", "unknown"),
        "progress_percent": int(progress.get("progress", 0) or 0),
    }


@app.get("/report/{job_id}", status_code=200)
def get_report(job_id: str, db: Session = Depends(get_db)):
    job = _require_job(db, job_id)
    report_json = _require_report_json(job)
    report_dict = _load_report_payload(report_json)
    
    # Generate summary if missing
    if not report_dict.get("summary"):
        try:
            report_obj = AuditReport(**report_dict)
            explainer = Explainer()
            summary = " ".join(str(explainer.generate_summary(report_obj)).split())
            
            # Update report with generated summary
            report_dict["summary"] = summary
            def sanitize_for_json(obj):
                if isinstance(obj, dict):
                    return {k: sanitize_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize_for_json(i) for i in obj]
                elif isinstance(obj, bool):
                    return int(obj)
                elif isinstance(obj, float):
                    import math
                    if math.isnan(obj) or math.isinf(obj):
                        return 0.0
                    return obj
                elif hasattr(obj, "__dict__"):
                    return sanitize_for_json(obj.__dict__)
                return obj

            job.report_json = json.dumps(sanitize_for_json(report_dict))
            db.commit()
        except Exception as e:
            LOGGER.warning(f"Failed to generate summary for job {job_id}: {e}")
            # Continue without summary on error
    
    return report_dict


@app.get("/report/{job_id}/summary", status_code=200)
def get_report_summary(job_id: str, db: Session = Depends(get_db)):
    job = _require_job(db, job_id)
    report_json = _require_report_json(job)
    return _cached_summary(report_json)


@app.get("/report/{job_id}/findings", status_code=200)
def get_report_findings(
    job_id: str,
    severity: str | None = Query(default=None),
    finding_type: str | None = Query(default=None),
    document_name: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    job = _require_job(db, job_id)
    report_json = _require_report_json(job)
    findings = copy.deepcopy(_cached_findings(report_json))

    if severity:
        severity_key = severity.strip().upper()
        if severity_key not in ALLOWED_FINDING_SEVERITIES:
            raise HTTPException(status_code=422, detail="Invalid severity. Use HIGH, MEDIUM, or LOW")
        findings = [item for item in findings if str(item.get("severity", "")).upper() == severity_key]

    if finding_type:
        ft = finding_type.strip().lower()
        findings = [item for item in findings if ft in str(item.get("finding_type", "")).lower()]

    if document_name:
        dn = document_name.strip().lower()
        findings = [item for item in findings if dn in str(item.get("document_name", "")).lower()]

    total = len(findings)
    start = (page - 1) * page_size
    end = start + page_size
    items = findings[start:end]

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items,
        "findings": items,
    }


@app.get("/report/{job_id}/analytics", status_code=200)
def get_report_analytics(job_id: str, db: Session = Depends(get_db)):
    job = _require_job(db, job_id)
    report_json = _require_report_json(job)
    return _cached_analytics(report_json)


@app.get("/report/{job_id}/document/{document_name}", status_code=200)
def get_report_document(job_id: str, document_name: str, db: Session = Depends(get_db)):
    job = _require_job(db, job_id)
    report_json = _require_report_json(job)
    findings = copy.deepcopy(_cached_findings(report_json))

    target = document_name.strip().lower()
    scoped = [item for item in findings if str(item.get("document_name", "")).strip().lower() == target]
    if not scoped:
        raise HTTPException(status_code=404, detail="Document not found in report findings")

    entities: list[dict[str, Any]] = []
    transaction_refs: list[str] = []
    for item in scoped:
        evidence = item.get("evidence", {}) or {}
        entity = {
            "vendor_name": evidence.get("vendor_name") or evidence.get("vendor"),
            "vendor_gst": evidence.get("vendor_gst") or evidence.get("gstin"),
            "category": evidence.get("category"),
            "invoice_number": evidence.get("invoice_number"),
            "invoice_date": evidence.get("invoice_date") or evidence.get("date"),
        }
        if any(value not in (None, "") for value in entity.values()):
            entities.append(entity)

        tx_ids = item.get("transaction_ids", [])
        if isinstance(tx_ids, list):
            for tx_id in tx_ids:
                if tx_id:
                    transaction_refs.append(str(tx_id))

    unique_transactions = sorted(set(transaction_refs))

    return {
        "document_name": document_name,
        "entities": entities,
        "transactions": [{"reference_id": ref} for ref in unique_transactions],
        "flags": scoped,
    }


@app.get("/report/{job_id}/download", status_code=200)
async def download_report(job_id: str):
    db = SessionLocal()
    job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
    db.close()

    if not job or job.status != "complete":
        raise HTTPException(status_code=404, detail="Report not found")

    # Parse report from JSON string
    import json

    try:
        report_dict = json.loads(job.report_json)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stored report JSON is invalid: {exc}") from exc

    # Generate PDF
    from api.report_generator import ReportGenerator

    try:
        generator = ReportGenerator()
        pdf_bytes = generator.generate_pdf(report_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=audit_report.pdf"
        },
    )


@app.get("/health", status_code=200)
def health_check():
    ollama_url = os.environ.get("OLLAMA_URL", OLLAMA_URL_DEFAULT)
    ollama_connected, _model_available, _elapsed = _ollama_status(ollama_url)
    return {
        "status": "online",
        "ollama": ollama_connected,
        "version": app.version,
    }


@app.post("/settings/test-ollama", status_code=200)
def test_ollama_connection(payload: OllamaTestRequest):
    if not payload.url or not payload.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="Invalid URL. Use http:// or https://")

    connected, model_available, elapsed = _ollama_status(payload.url)
    return {
        "connected": connected,
        "model_available": model_available,
        "response_time_ms": elapsed,
    }


@app.delete("/session/{job_id}", status_code=200)
def delete_session_files(job_id: str, db: Session = Depends(get_db)):
    job = _require_job(db, job_id)

    file_paths = job.get_file_paths()
    upload_dirs = {Path(path).parent for path in file_paths}

    for folder in upload_dirs:
        try:
            resolved_folder = folder.resolve()
            if UPLOAD_ROOT.resolve() in resolved_folder.parents or resolved_folder == UPLOAD_ROOT.resolve():
                shutil.rmtree(resolved_folder, ignore_errors=True)
        except Exception:
            continue

    job.file_paths = json.dumps([])
    db.commit()

    return {
        "job_id": job_id,
        "message": "Session files deleted; report retained in database",
    }


@app.post("/explain/{job_id}/{finding_id}", status_code=200)
def explain_finding(job_id: str, finding_id: str, db: Session = Depends(get_db)):
    """Generate explanation for a specific finding on-demand.
    
    Args:
        job_id: ID of the audit job
        finding_id: ID of the anomaly finding to explain
    
    Returns:
        {finding_id, explanation: str}
    """
    job = _require_job(db, job_id)
    
    if not job.report_json:
        raise HTTPException(status_code=400, detail="No report available for this job")
    
    try:
        report_data = json.loads(job.report_json)
        report = AuditReport(**report_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse report: {str(e)}")
    
    # Find the anomaly with matching finding_id
    finding = None
    for anomaly in report.anomalies:
        if anomaly.finding_id == finding_id:
            finding = anomaly
            break
    
    if not finding:
        raise HTTPException(status_code=404, detail=f"Finding {finding_id} not found in report")
    
    # Generate explanation in a thread with 30-second timeout
    def generate_explanation():
        try:
            explainer = Explainer()
            return explainer.explain_anomaly(finding)
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"
    
    try:
        # Run in thread pool with timeout
        future = executor.submit(generate_explanation)
        explanation = future.result(timeout=45)
    except TimeoutError:
        explanation = "Explanation timed out"
    except Exception as e:
        explanation = f"Explanation unavailable: {str(e)}"
    
    return {
        "finding_id": finding_id,
        "explanation": explanation,
    }
