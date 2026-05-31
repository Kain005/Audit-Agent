"""Core LangGraph workflow for finance audit processing."""

from __future__ import annotations

from functools import wraps
import logging
from pydoc import doc
import uuid
from datetime import datetime
from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

try:
    from ..analysis.anomaly_detector import AnomalyDetector
    from ..analysis.rules_engine import RulesEngine
    from ..config.loader import get_policy
    from ..extraction.cross_document_matcher import CrossDocumentMatcher
    from ..extraction.entity_extractor import EntityExtractor
    from ..extraction.models import (
        AnomalyFinding,
        AuditReport,
        InvoiceEntities,
        PolicyViolation,
        TransactionEntity,
    )
    from ..ingestion.document_router import DocumentRouter
    from ..ingestion.expense_parser import ExpenseParser
    from ..ingestion.gst_parser import GSTParser
    from .explainer import Explainer
except ImportError:  # pragma: no cover - fallback for script-style execution
    from analysis.anomaly_detector import AnomalyDetector
    from analysis.rules_engine import RulesEngine
    from config.loader import get_policy
    from extraction.cross_document_matcher import CrossDocumentMatcher
    from extraction.entity_extractor import EntityExtractor
    from extraction.models import (
        AnomalyFinding,
        AuditReport,
        InvoiceEntities,
        PolicyViolation,
        TransactionEntity,
    )
    from ingestion.document_router import DocumentRouter
    from ingestion.expense_parser import ExpenseParser
    from ingestion.gst_parser import GSTParser
    from agent.explainer import Explainer

LOGGER = logging.getLogger(__name__)


class AgentState(TypedDict):
    files: list[str]
    parsed_documents: list[dict[str, Any]]
    expense_sheets: list[pd.DataFrame]
    gst_documents: list[dict[str, Any]]
    invoices: list[InvoiceEntities]
    transactions: list[TransactionEntity]
    known_vendors: list[str]
    anomalies: list[AnomalyFinding]
    vendor_risk_findings: list[AnomalyFinding]
    gst_anomalies: list[AnomalyFinding]
    violations: list[PolicyViolation]
    expense_violations: list[PolicyViolation]
    document_summary: dict[str, int]
    reconciliation: dict[str, Any]
    combined_df: pd.DataFrame | None
    combined_transaction_count: int
    gst_transaction_count: int
    explained_findings: list[Any]
    report: AuditReport | None
    errors: list[str]
    current_step: str
    progress_callback: Callable[[str, int], None] | None
    audit_config: dict[str, Any]


STEP_PROGRESS_MAP = {
    "ingest": 20,
    "extract": 40,
    "analyze": 60,
    "explain": 80,
    "compile": 100,
}


def _emit_progress(state: AgentState, step_name: str) -> None:
    callback = state.get("progress_callback")
    if not callback:
        return

    progress_value = STEP_PROGRESS_MAP.get(step_name)
    if progress_value is None:
        return

    try:
        callback(step_name, progress_value)
    except Exception as exc:  # pragma: no cover - non-critical progress callback failures
        LOGGER.warning("Progress callback failed for step=%s: %s", step_name, exc)


def _fallback_report_from_state(state: AgentState, error_message: str | None = None) -> AuditReport:
    anomalies = state.get("anomalies", [])
    violations = state.get("violations", [])

    high_count = sum(1 for item in anomalies if item.severity == "HIGH") + sum(
        1 for item in violations if str(item.severity).upper() == "HIGH"
    )
    medium_count = sum(1 for item in anomalies if item.severity == "MEDIUM") + sum(
        1 for item in violations if str(item.severity).upper() == "MEDIUM"
    )
    low_count = sum(1 for item in anomalies if item.severity == "LOW") + sum(
        1 for item in violations if str(item.severity).upper() == "LOW"
    )

    base = (high_count * 18) + (medium_count * 6) + (low_count * 2)
    risk_score = min(100.0, round((base / 300) * 100, 1))

    summary = _extract_summary(state.get("explained_findings", []))
    if error_message:
        summary = f"{summary or ''}\nPipeline warning: {error_message}".strip()

    reconciliation = state.get("reconciliation", {})
    unmatched_invoices = [_invoice_identifier(inv) for inv in reconciliation.get("unmatched_invoices", [])]
    unmatched_payments = [_payment_identifier(txn) for txn in reconciliation.get("unmatched_payments", [])]

    return AuditReport(
        report_id=str(pd.Timestamp.utcnow().value),
        generated_at=pd.Timestamp.utcnow().isoformat(),
        documents_processed=len(state.get("files", [])),
        total_transactions=(
    sum(
        len(doc.get("data")) if isinstance(doc.get("data"), pd.DataFrame) else 0
        for doc in state.get("parsed_documents", [])
        if doc.get("document_type") in ("bank_statement", "expense_sheet")
    )
    or len(state.get("transactions", []))
),
        total_invoices=len(state.get("invoices", [])),
        risk_score=risk_score,
        anomalies=anomalies,
        policy_violations=violations,
        unmatched_invoices=unmatched_invoices,
        unmatched_payments=unmatched_payments,
        summary=summary,
        document_breakdown=dict(state.get("document_summary", {})),
        top_3_concerns=_top_concerns(anomalies, violations, limit=3),
    )


def safe_node(func):
    @wraps(func)
    def wrapper(state):
        try:
            return func(state)
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors = list(state.get("errors", []))
            errors.append(f"{func.__name__}: {str(e)}")
            # Return full state fields to prevent LangGraph from losing upstream data
            return {
                "errors": errors,
                "invoices": state.get("invoices", []),
                "transactions": state.get("transactions", []),
                "parsed_documents": state.get("parsed_documents", []),
                "expense_sheets": state.get("expense_sheets", []),
                "gst_documents": state.get("gst_documents", []),
                "known_vendors": state.get("known_vendors", []),
                "document_summary": state.get("document_summary", {}),
                "reconciliation": state.get("reconciliation", {}),
            }
    return wrapper


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        import math
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, float):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj


@safe_node
def ingest_node(state: AgentState) -> dict[str, Any]:
    """Ingest uploaded files through document router."""
    router = DocumentRouter()
    files = state.get("files", [])
    raw_documents = router.process_batch(files)
    with open("C:\\Financial Audit\\ingest_debug.txt", "w") as _f:
        for d in raw_documents:
            _f.write(f"type={d.get('document_type')} parser={d.get('parser_used')} invoice_data_keys={list((d.get('invoice_data') if isinstance(d.get('invoice_data'), dict) else (d.get('invoice_data').model_dump() if d.get('invoice_data') else {})).keys())}\n")
    parsed_documents: list[dict[str, Any]] = []
    errors = list(state.get("errors", []))

    document_summary: dict[str, int] = {
        "invoice": 0,
        "gst_invoice": 0,
        "bank_statement": 0,
        "ledger": 0,
        "expense_sheet": 0,
        "gst_document": 0,
    }

    for doc in raw_documents:
        normalized_doc = dict(doc)
        raw_type = str(normalized_doc.get("document_type", "unknown"))
        if raw_type == "gst":
            normalized_doc["document_type"] = "gst_document"

        normalized_type = str(normalized_doc.get("document_type", "unknown"))
        if normalized_type in document_summary:
            document_summary[normalized_type] += 1

        parsed_documents.append(normalized_doc)

    invoices_docs = [doc for doc in parsed_documents if doc.get("document_type") == "invoice"]
    gst_invoice_docs = [doc for doc in parsed_documents if doc.get("document_type") == "gst_invoice"]
    invoices: list[Any] = []
    for doc in invoices_docs:
        invoice_data = doc.get("invoice_data")
        if invoice_data is None:
            continue
        if isinstance(invoice_data, InvoiceEntities):
            invoices.append(invoice_data)
        elif isinstance(invoice_data, dict):
            try:
                invoices.append(InvoiceEntities(**invoice_data))
            except Exception as exc:
                errors.append(f"Invoice deserialization failed: {exc}")
    for doc in gst_invoice_docs:
        invoice_data = doc.get("invoice_data")
        if invoice_data is None:
            continue
        if isinstance(invoice_data, InvoiceEntities):
            invoices.append(invoice_data)
        elif isinstance(invoice_data, dict):
            try:
                invoices.append(InvoiceEntities(**invoice_data))
            except Exception as exc:
                errors.append(f"GSTInvoice deserialization failed: {exc}")
    bank_docs = [doc for doc in parsed_documents if doc.get("document_type") == "bank_statement"]
    ledger_docs = [doc for doc in parsed_documents if doc.get("document_type") == "ledger"]
    expense_docs = [doc for doc in parsed_documents if doc.get("document_type") == "expense_sheet"]
    gst_docs = [doc for doc in parsed_documents if doc.get("document_type") == "gst_document"]
    
    for doc in parsed_documents:
        parse_errors = doc.get("parse_errors", [])
        if isinstance(parse_errors, list):
            for err in parse_errors:
                if err:
                    errors.append(str(err))

    LOGGER.info(
        "Step ingest: processed=%s invoices=%s gst_invoices=%s bank_statements=%s ledgers=%s expense_sheets=%s gst_documents=%s",
        len(files),
        len(invoices_docs),
        len(gst_invoice_docs),
        len(bank_docs),
        len(ledger_docs),
        len(expense_docs),
        len(gst_docs),
    )

    return {
        "parsed_documents": parsed_documents,
        "document_summary": document_summary,
        "invoices": invoices,
        "expense_sheets": expense_docs,
        "errors": errors,
        "current_step": "ingest",
    }


@safe_node
def extract_node(state: AgentState) -> dict[str, Any]:
    """Extract invoice entities, transactions, and known vendors from parsed documents."""
    extractor = EntityExtractor()
    print(f"DEBUG parsed_documents count: {len(state.get('parsed_documents', []))}", flush=True)

    expense_parser = ExpenseParser()
    gst_parser = GSTParser()

    invoices: list[InvoiceEntities] = list(state.get("invoices", []))
    
    transactions: list[TransactionEntity] = []
    expense_sheets: list[pd.DataFrame] = []
    gst_documents: list[dict[str, Any]] = []
    known_vendors: set[str] = set()
    errors = list(state.get("errors", []))

    for parsed in state.get("parsed_documents", []):
        doc_type = str(parsed.get("document_type", "unknown"))
        print(f"DEBUG doc_type={doc_type} parser_used={parsed.get('parser_used')}", flush=True)
        parser_used = str(parsed.get("parser_used", ""))
        raw_text = parsed.get("raw_text")
        data = parsed.get("data")

        try:
            if doc_type == "invoice":
                invoice_text = str(raw_text or "")
                print(f"DEBUG invoice doc: raw_text_len={len(invoice_text)} data_type={type(data).__name__}", flush=True)


                if not invoice_text and isinstance(data, pd.DataFrame) and not data.empty:
                    invoice_text = data.to_string(index=False)

                if invoice_text.strip():
                    print(f"DEBUG invoice_text length: {len(invoice_text or '')}", flush=True)
                    invoices.append(extractor.extract_invoice_entities(invoice_text))
                else:
                    errors.append(f"No extractable invoice text found for parser={parser_used}")

            elif doc_type == "bank_statement":
                if isinstance(data, pd.DataFrame) and not data.empty:
                    transactions.extend(_transactions_from_dataframe(data))
                else:
                    errors.append(f"Bank statement has no rows for parser={parser_used}")

            elif doc_type == "ledger":
                if isinstance(data, pd.DataFrame) and not data.empty:
                    vendor_col = _resolve_vendor_column(data)
                    for value in data[vendor_col].dropna().astype(str).tolist():
                        cleaned = value.strip()
                        if cleaned:
                            known_vendors.add(cleaned)
                else:
                    errors.append(f"Ledger has no rows for parser={parser_used}")

            elif doc_type == "expense_sheet":
                if isinstance(data, pd.DataFrame) and not data.empty:
                    expense_df = data.copy()
                    if "description" in expense_df.columns:
                        if "category" not in expense_df.columns:
                            expense_df["category"] = ""
                        expense_df["category"] = expense_df.apply(
                            lambda row: str(row.get("category", "")).strip()
                            if str(row.get("category", "")).strip()
                            else expense_parser.categorize_expense(str(row.get("description", ""))),
                            axis=1,
                        )
                    expense_sheets.append(expense_df)
                else:
                    errors.append(f"Expense sheet has no rows for parser={parser_used}")

            elif doc_type == "gst_document":
                if isinstance(data, pd.DataFrame) and not data.empty:
                    gst_doc = dict(parsed)
                    gst_df = data.copy()
                    gst_df["document_name"] = str(parsed.get("file_name") or "gst_document")
                    if "transaction_id" not in gst_df.columns:
                        gst_df["transaction_id"] = gst_df.index.astype(str)
                    gst_doc["data"] = gst_df
                    gst_doc["document_name"] = str(parsed.get("file_name") or "gst_document")
                    validation_failures = gst_parser.validate_gst_entries(data)
                    gst_doc["validation_failures"] = validation_failures
                    gst_documents.append(gst_doc)
                    if validation_failures:
                        errors.append(
                            f"GST validation found {len(validation_failures)} issue(s) for parser={parser_used}"
                        )
                else:
                    errors.append(f"GST document has no rows for parser={parser_used}")

        except Exception as exc:
            errors.append(f"Extraction failed for document ({doc_type}): {exc}")

    LOGGER.info(
        "Step extract: invoices=%s, transactions=%s, expense_sheets=%s, gst_documents=%s, known_vendors=%s",
        len(invoices),
        len(transactions),
        len(expense_sheets),
        len(gst_documents),
        len(known_vendors),
    )

    
    return {
        "invoices": invoices,
        "transactions": transactions,
        "expense_sheets": expense_sheets,
        "gst_documents": gst_documents,
        "known_vendors": sorted(known_vendors),
        "errors": errors,
        "current_step": "extract",
    }


@safe_node
def cross_reference_node(state: AgentState) -> dict[str, Any]:
    """Cross-reference invoices and transactions for payment reconciliation."""
    matcher = CrossDocumentMatcher()
    errors = list(state.get("errors", []))

    try:
        reconciliation = matcher.reconcile(
            state.get("invoices", []),
            state.get("transactions", []),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        errors.append(f"Cross-reference failed: {exc}")
        reconciliation = {
            "matched_pairs": [],
            "unmatched_invoices": state.get("invoices", []),
            "unmatched_payments": state.get("transactions", []),
        }
    

    LOGGER.info(
        "Step cross_reference: matched_pairs=%s",
        len(reconciliation.get("matched_pairs", [])),
    )

    
    return {
        "reconciliation": reconciliation,
        "errors": errors,
        "current_step": "cross_reference",
    }


@safe_node
def analyze_node(state: AgentState) -> dict[str, Any]:
    """Run anomaly detection and policy rules evaluation."""
    errors = list(state.get("errors", []))
    detector = AnomalyDetector()
    rules_engine = RulesEngine()
    audit_config = state.get("audit_config", {})

    all_transactions: list[pd.DataFrame] = []
    for doc in state.get("parsed_documents", []):
        if doc.get("document_type") == "bank_statement":
            df = doc.get("data")
            if df is not None and not df.empty:
                df = df.copy()
                file_name = doc.get("file_name")
                bank_name = doc.get("bank_name")
                if isinstance(file_name, str) and file_name.strip() and file_name.strip().lower() != "unknown":
                    document_name = file_name.strip()
                elif isinstance(bank_name, str) and bank_name.strip():
                    document_name = bank_name.strip()
                else:
                    document_name = "unknown"
                df["document_name"] = document_name
                all_transactions.append(df)

    tx_df = pd.concat(all_transactions, ignore_index=True) if all_transactions else pd.DataFrame()

# ── TEMPORARY DEBUG ──────────────────────────────────────────────────────────
    if not tx_df.empty:
        from analysis.anomaly_detector import AnomalyDetector as _AD
        _d = _AD()
        _prepared = _d._prepare_dataframe(tx_df)
        _tagged = _d.tag_transaction_categories(_prepared)

        print(f"DEBUG category counts:\n{_tagged['transaction_category'].value_counts().to_dict()}", flush=True)

        f1 = _d.detect_amount_outliers(_tagged)
        print(f"DEBUG amount_outliers={len(f1)}", flush=True)

        f2 = _d.detect_timing_anomalies(_tagged)
        print(f"DEBUG timing_anomalies={len(f2)}", flush=True)

        f3 = _d.detect_frequency_anomalies(_tagged)
        print(f"DEBUG frequency_anomalies={len(f3)}", flush=True)

        f4 = _d.detect_split_billing(_tagged, {})
        print(f"DEBUG split_billing={len(f4)}", flush=True)

        f5 = _d.detect_multivariate_anomalies(_tagged)
        print(f"DEBUG multivariate={len(f5)}", flush=True)

        f6 = _d.detect_suspicious_payment_patterns(_tagged)
        print(f"DEBUG suspicious_patterns={len(f6)}", flush=True)

        print(f"DEBUG debit nonzero={(_tagged['debit']>0).sum()}, credit nonzero={(_tagged['credit']>0).sum()}", flush=True)
        print(f"DEBUG amount stats: mean={_tagged['amount'].mean():.0f} std={_tagged['amount'].std():.0f}", flush=True)
        print(f"DEBUG weekend rows: {(_tagged['parsed_date'].dt.weekday >= 5).sum()}", flush=True)
# ── END TEMPORARY DEBUG ──────────────────────────────────────────────────────
    expense_df = _concat_dataframes(state.get("expense_sheets", []))



    gst_df = _concat_dataframes([
        doc.get("data") for doc in state.get("gst_documents", []) if isinstance(doc.get("data"), pd.DataFrame)
    ])
    gst_transaction_count = len(gst_df)

    all_docs = state.get("parsed_documents", [])
    dfs_to_combine: list[pd.DataFrame] = []

    for doc in all_docs:
        data = doc.get("data")
        if data is not None and hasattr(data, "shape") and not data.empty:
            dfs_to_combine.append(data)
        
        # TO:
        elif doc.get("document_type") == "invoice": 

            invoice = doc.get("invoice_data")
            if invoice is not None:
                # invoice_data may be InvoiceEntities object or plain dict
                if hasattr(invoice, "line_items"):
                    line_items = invoice.line_items
                elif isinstance(invoice, dict):
                    line_items = invoice.get("line_items", [])
                else:
                    line_items = []
                rows = []
                for item in line_items:
                    if isinstance(item, dict):
                        rows.append({
                            "description": item.get("description"),
                            "amount": item.get("amount"),
                            "quantity": item.get("quantity"),
                            "unit_price": item.get("unit_price"),
                            "gst_rate": item.get("gst_rate"),
                            "document_name": doc.get("file_name", "unknown"),
                        })
                    else:
                        rows.append({
                            "description": item.description,
                            "amount": item.amount,
                            "quantity": item.quantity,
                            "unit_price": item.unit_price,
                            "gst_rate": item.gst_rate,
                            "document_name": doc.get("file_name", "unknown"),
                        })
                if rows:
                    dfs_to_combine.append(pd.DataFrame(rows))

    if dfs_to_combine:
        combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    exclude = []
    if audit_config:
        exclude = [
            audit_config.get("salary_amount", 0),
            audit_config.get("emi_amount", 0),
            audit_config.get("rent_amount", 0),
        ]
        exclude = [x for x in exclude if x > 0]

    # Extract approval_threshold from audit config
    approval_threshold = audit_config.get("approval_threshold") if audit_config else None

    anomalies: list[AnomalyFinding] = []
    vendor_risk_findings: list[AnomalyFinding] = []
    gst_anomalies: list[AnomalyFinding] = []
    violations: list[PolicyViolation] = []
    expense_violations: list[PolicyViolation] = []

    try:
        # Run detection when either bank transactions or GST data are present.
        # Prefer bank transactions as primary input, but allow GST-only audits.
        primary_df = tx_df if not tx_df.empty else (expense_df if not expense_df.empty else gst_df)
        if not primary_df.empty:
            all_findings = detector.detect_all(
                primary_df,
                rules_engine.policy,
                expense_df=expense_df,
                gst_df=gst_df,
                transactions_df=tx_df if not tx_df.empty else None,
                invoices=state.get("invoices", []),
                exclude_amounts=exclude or None,
            )
            anomalies = all_findings
        
        # Separate gst anomalies if any
        gst_anomalies = [f for f in anomalies if hasattr(f, 'anomaly_type') and f.anomaly_type == "gst"]
        vendor_risk_findings = [f for f in anomalies if hasattr(f, 'anomaly_type') and f.anomaly_type == "vendor_risk"]
        
    except Exception as exc:
        errors.append(f"Anomaly detection failed: {exc}")
        anomalies = []
        vendor_risk_findings = []
        gst_anomalies = []

    try:
        # Use evaluate_all with approval_threshold override
        violations = rules_engine.evaluate_all(
            state.get("invoices", []),
            tx_df.to_dict("records") if not tx_df.empty else [],
            state.get("known_vendors", []),
            approval_threshold=approval_threshold,
        )

        if not expense_df.empty:
            employee_monthly_spend = _employee_monthly_spend(expense_df)
            split_billed_vendors = {
                str(f.evidence.get("vendor", "")).lower()
                for f in anomalies
                if isinstance(f, AnomalyFinding) and f.finding_type == "split_billing"
            }
            for _, row in expense_df.iterrows():
                expense_payload = row.to_dict()

                submitted_by = str(expense_payload.get("submitted_by") or expense_payload.get("employee") or "").strip()
                expense_date = pd.to_datetime(expense_payload.get("date"), errors="coerce")
                if submitted_by and pd.notna(expense_date):
                    key = (submitted_by.lower(), expense_date.strftime("%Y-%m"))
                    expense_payload["employee_monthly_spend"] = employee_monthly_spend.get(
                        key,
                        _to_float(expense_payload.get("amount")),
                    )

                violation = rules_engine.check_expense_limits(expense_payload, rules_engine.policy)
                if violation:
                    # Suppress category_daily_limit if this row is already covered
                    # by a split_billing finding — avoids double-flagging same transactions
                    if violation.rule_name == "expense_category_daily_limit":
                        row_desc = str(expense_payload.get("description", "")).lower()
                        if any(sv and sv in row_desc for sv in split_billed_vendors):
                            continue
                    expense_violations.append(violation)

        all_vendor_candidates: set[tuple[str, str]] = set()
        for invoice in state.get("invoices", []):
            name = (invoice.vendor_name or "").strip()
            gstin = (invoice.vendor_gst or "").strip()
            if name or gstin:
                all_vendor_candidates.add((name, gstin))

        for vendor in state.get("known_vendors", []):
            name = str(vendor or "").strip()
            if name:
                all_vendor_candidates.add((name, ""))

        for txn in state.get("transactions", []):
            name = str(txn.party_name or "").strip()
            if name:
                all_vendor_candidates.add((name, ""))

        for vendor_name, vendor_gst in sorted(all_vendor_candidates):
            blacklisted = rules_engine.check_vendor_blacklist(vendor_name, vendor_gst)
            if blacklisted:
                violations.append(blacklisted)

        employee_names = _extract_employee_names_from_expense_sheets(state.get("expense_sheets", []))
        for vendor_name, _ in sorted(all_vendor_candidates):
            related_party = rules_engine.check_related_party(vendor_name, employee_names)
            if related_party:
                violations.append(related_party)

        violations.extend(expense_violations)
        violations = rules_engine._deduplicate_violations(violations)
    except Exception as exc:
        errors.append(f"Rules evaluation failed: {exc}")
        violations = []
        expense_violations = []

    LOGGER.info(
        "Step analyze: anomalies=%s gst_anomalies=%s vendor_risk_findings=%s violations=%s expense_violations=%s",
        len(anomalies),
        len(gst_anomalies),
        len(vendor_risk_findings),
        len(violations),
        len(expense_violations),
    )

    return {
        "anomalies": anomalies,
        "vendor_risk_findings": vendor_risk_findings,
        "gst_anomalies": gst_anomalies,
        "violations": violations,
        "expense_violations": expense_violations,
        "combined_df": combined_df,
        "combined_transaction_count": len(combined_df),
        "gst_transaction_count": gst_transaction_count,
        "errors": errors,
        "current_step": "analyze",
    }


@safe_node
def explain_node(state: AgentState) -> dict[str, Any]:
    """Generate explanations for HIGH-severity findings and create summary text."""
    explainer = Explainer()
    explained_findings: list[Any] = []
    errors = list(state.get("errors", []))

    anomalies = state.get("anomalies", [])
    violations = state.get("violations", [])

    for finding in anomalies:
        if finding.severity == "HIGH":
            try:
                explanation = explainer.explain_anomaly(finding)
                finding.explanation = explanation
                explained_findings.append(
                    {
                        "type": "anomaly",
                        "id": finding.finding_id,
                        "severity": finding.severity,
                        "explanation": explanation,
                    }
                )
            except Exception as exc:
                errors.append(f"Anomaly explanation failed ({finding.finding_id}): {exc}")

    for violation in violations:
        if str(violation.severity).upper() == "HIGH":
            try:
                explanation = explainer.explain_violation(violation)
                explained_findings.append(
                    {
                        "type": "violation",
                        "id": violation.violation_id,
                        "severity": violation.severity,
                        "explanation": explanation,
                    }
                )
            except Exception as exc:
                errors.append(f"Violation explanation failed ({violation.violation_id}): {exc}")

    try:
        temp_report = AuditReport(
            report_id="preview",
            generated_at="",
            documents_processed=len(state.get("files", [])),
            total_transactions=(
    sum(
        len(doc.get("data")) if isinstance(doc.get("data"), pd.DataFrame) else 0
        for doc in state.get("parsed_documents", [])
        if doc.get("document_type") in ("bank_statement", "expense_sheet")
    )
    or len(state.get("transactions", []))
),
            total_invoices=len(state.get("invoices", [])),
            risk_score=0.0,
            anomalies=anomalies,
            policy_violations=violations,
            unmatched_invoices=[],
            unmatched_payments=[],
            summary=None,
        )
        summary_text = explainer.generate_summary(temp_report)
        explained_findings.append({"type": "summary", "text": summary_text})
    except Exception as exc:
        errors.append(f"Summary generation failed: {exc}")

    LOGGER.info("Step explain: explained_entries=%s", len(explained_findings))

    return {
        "anomalies": anomalies,
        "explained_findings": explained_findings,
        "errors": errors,
        "current_step": "explain",
    }


@safe_node
def compile_report_node(state: AgentState) -> dict[str, Any]:
    """Compile final report from current state and reconciliation outputs."""
    anomalies = state.get("anomalies", [])
    violations = state.get("violations", [])

    all_findings = anomalies + violations

    # Separate anomalies from policy violations for weighted scoring
    anomaly_findings = [f for f in all_findings if isinstance(f, AnomalyFinding)]
    violation_findings = [f for f in all_findings if isinstance(f, PolicyViolation)]

    # Weight anomalies by their actual score, not just severity count
    HIGH_WEIGHT = 18
    MEDIUM_WEIGHT = 6
    LOW_WEIGHT = 2

    # Anomalies: weight by score magnitude to avoid noise inflating the total
    anomaly_score_sum = 0.0
    for f in anomaly_findings:
        if f.severity == "HIGH":
            anomaly_score_sum += HIGH_WEIGHT * float(f.score)
        elif f.severity == "MEDIUM":
            # Multivariate and IQR-only findings get half weight
            is_low_confidence = f.finding_type in ("multivariate_anomaly",) or (
                isinstance(f.evidence, dict) and not f.evidence.get("z_method_flag", True)
            )
            weight = MEDIUM_WEIGHT * 0.5 if is_low_confidence else MEDIUM_WEIGHT
            anomaly_score_sum += weight * float(f.score)
        else:
            anomaly_score_sum += LOW_WEIGHT * float(f.score)

    # Policy violations are more definitive — weight higher
    violation_score_sum = 0.0
    for f in violation_findings:
        sev = str(f.severity).upper()
        if sev == "HIGH":
            violation_score_sum += HIGH_WEIGHT * 1.2
        elif sev == "MEDIUM":
            violation_score_sum += MEDIUM_WEIGHT * 1.0
        else:
            violation_score_sum += LOW_WEIGHT

    raw_score = anomaly_score_sum + violation_score_sum

    # Normalize: 300 raw points = 100 risk score, with diminishing returns above 50
    normalized = (raw_score / 300) * 100
    if normalized > 50:
        normalized = 50 + (normalized - 50) * 0.5

    risk_score = min(100.0, round(normalized, 1))

    high_count = sum(1 for f in all_findings if str(f.severity).upper() == "HIGH")
    medium_count = sum(1 for f in all_findings if str(f.severity).upper() == "MEDIUM")
    low_count = sum(1 for f in all_findings if str(f.severity).upper() == "LOW")

    reconciliation = state.get("reconciliation", {})
    unmatched_invoices_raw = reconciliation.get("unmatched_invoices", [])
    unmatched_payments_raw = reconciliation.get("unmatched_payments", [])

    unmatched_invoices = [_invoice_identifier(inv) for inv in unmatched_invoices_raw]
    unmatched_payments = [_payment_identifier(txn) for txn in unmatched_payments_raw]

    summary = _extract_summary(state.get("explained_findings", []))

    top_3_concerns = _top_concerns(anomalies, violations, limit=3)
    document_breakdown = dict(state.get("document_summary", {}))
    gst_documents = state.get("gst_documents", [])
    gst_transaction_count = sum(
        len(doc["data"]) for doc in gst_documents
        if isinstance(doc.get("data"), pd.DataFrame)
    )

    gst_invoice_payloads = [
        {
            "file_name": doc.get("file_name", ""),
            "invoice_data": doc.get("invoice_data") or {},
        }
        for doc in state.get("parsed_documents", [])
        if doc.get("document_type") == "gst_invoice"
    ]

    report = AuditReport(
        report_id=str(uuid.uuid4()),
        generated_at=datetime.now().isoformat(),
        documents_processed=len(state.get("files", [])),
        total_transactions=(
    sum(
        len(doc.get("data")) if isinstance(doc.get("data"), pd.DataFrame) else 0
        for doc in state.get("parsed_documents", [])
        if doc.get("document_type") in ("bank_statement", "expense_sheet")
    )
    or len(state.get("transactions", []))
),
        gst_transaction_count=gst_transaction_count,
        combined_transaction_count=int(state.get("combined_transaction_count", 0) or 0),
        total_invoices=len(state.get("invoices", [])),
        risk_score=float(risk_score),
        anomalies=anomalies,
        policy_violations=violations,
        unmatched_invoices=unmatched_invoices,
        unmatched_payments=unmatched_payments,
        summary=summary,
        document_breakdown=document_breakdown,
        top_3_concerns=top_3_concerns,
        gst_invoices=gst_invoice_payloads if gst_invoice_payloads else None,
    )

    report_payload = sanitize_for_json(report.model_dump())
    report = AuditReport.model_validate(report_payload)

    LOGGER.info("Step compile: risk_score=%s", risk_score)

    return {
        "report": report,
        "current_step": "compile",
    }


def should_explain(state: AgentState) -> str:
    """Always skip explain node; explanations generated on-demand via API."""
    return "compile"


workflow = StateGraph(AgentState)
workflow.add_node("ingest", ingest_node)
workflow.add_node("extract", extract_node)
workflow.add_node("cross_reference", cross_reference_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("compile", compile_report_node)

workflow.add_edge(START, "ingest")
workflow.add_edge("ingest", "extract")
workflow.add_edge("extract", "cross_reference")
workflow.add_edge("cross_reference", "analyze")
workflow.add_edge("analyze", "compile")
workflow.add_edge("compile", END)

graph = workflow.compile()


def run_audit(
    file_paths: list[str],
    progress_callback: Callable[[str, int], None] | None = None,
    audit_config: dict[str, Any] | None = None,
) -> AuditReport:
    """Run the audit graph for provided files and return the compiled report."""
    initial_state: AgentState = {
        "files": file_paths,
        "parsed_documents": [],
        "expense_sheets": [],
        "gst_documents": [],
        "invoices": [],
        "transactions": [],
        "known_vendors": [],
        "anomalies": [],
        "vendor_risk_findings": [],
        "gst_anomalies": [],
        "violations": [],
        "expense_violations": [],
        "document_summary": {
            "invoice": 0,
            "bank_statement": 0,
            "ledger": 0,
            "expense_sheet": 0,
            "gst_document": 0,
        },
        "reconciliation": {},
        "explained_findings": [],
        "report": None,
        "errors": [],
        "current_step": "start",
        "progress_callback": progress_callback,
        "audit_config": audit_config or {},
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        LOGGER.exception("Audit graph execution failed: %s", exc)
        final_state = dict(initial_state)
        final_state["errors"] = list(initial_state.get("errors", [])) + [f"graph execution failed: {exc}"]
        final_state["report"] = _fallback_report_from_state(initial_state, error_message=str(exc))

    report = final_state.get("report")
    if report is None:
        return _fallback_report_from_state(final_state, error_message="report generation fallback")
    return report


def _resolve_vendor_column(df: pd.DataFrame) -> str:
    for column in ["party_name", "vendor_name", "name", "account", "particulars"]:
        if column in df.columns:
            return column
    return df.columns[0]


def _transactions_from_dataframe(df: pd.DataFrame) -> list[TransactionEntity]:
    transactions: list[TransactionEntity] = []

    for index, row in df.iterrows():
        date_value = row.get("date")
        date_text = str(date_value) if pd.notna(date_value) else ""

        debit = _to_float(row.get("debit"))
        credit = _to_float(row.get("credit"))

        if credit > 0:
            amount = credit
            tx_type = "credit"
        else:
            amount = debit
            tx_type = "debit"

        description = str(row.get("description", "") or "").strip()
        party_name = _extract_party_name(description)

        transactions.append(
            TransactionEntity(
                date=date_text,
                description=description,
                party_name=party_name,
                amount=float(amount),
                transaction_type=tx_type,
                reference_id=str(row.get("transaction_id", "") or "").strip() or None,
                upi_details=None,
                bank_name=str(row.get("bank_name", "") or "").strip() or None,
                category=str(row.get("category", "") or "").strip() or None,
            )
        )

    return transactions


def _transactions_to_dataframe(transactions: list[TransactionEntity]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for transaction in transactions:
        debit = float(transaction.amount) if transaction.transaction_type == "debit" else 0.0
        credit = float(transaction.amount) if transaction.transaction_type == "credit" else 0.0

        records.append(
            {
                "date": transaction.date,
                "description": transaction.description,
                "party_name": transaction.party_name,
                "amount": float(transaction.amount),
                "debit": debit,
                "credit": credit,
                "transaction_id": transaction.reference_id or "",
                "bank_name": transaction.bank_name,
                "category": transaction.category,
                "document_name": transaction.bank_name or "transaction",
            }
        )

    return pd.DataFrame(records)


def _to_float(value: Any) -> float:
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


def _extract_party_name(description: str) -> str | None:
    text = (description or "").strip()
    if not text:
        return None

    # Use first chunk as a loose vendor/party fallback for statement-derived transactions.
    for separator in ["/", "-", "|"]:
        if separator in text:
            candidate = text.split(separator)[0].strip()
            return candidate or text

    return text


def _invoice_identifier(invoice: Any) -> str:
    if isinstance(invoice, InvoiceEntities):
        return invoice.invoice_number or invoice.vendor_name or "invoice"
    if isinstance(invoice, dict):
        return str(invoice.get("invoice_number") or invoice.get("vendor_name") or "invoice")
    return str(invoice)


def _payment_identifier(payment: Any) -> str:
    if isinstance(payment, TransactionEntity):
        return payment.reference_id or payment.description or "payment"
    if isinstance(payment, dict):
        return str(payment.get("reference_id") or payment.get("description") or "payment")
    return str(payment)


def _extract_summary(explained_findings: list[Any]) -> str | None:
    for item in explained_findings:
        if isinstance(item, dict) and item.get("type") == "summary":
            text = str(item.get("text") or "").strip()
            if text:
                return text
    return None


def _concat_dataframes(frames: list[Any]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


def _employee_monthly_spend(expense_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    if expense_df.empty:
        return {}

    data = expense_df.copy()
    if "submitted_by" not in data.columns or "amount" not in data.columns or "date" not in data.columns:
        return {}

    data["submitted_by_key"] = data["submitted_by"].fillna("").astype(str).str.strip().str.lower()
    data["amount_value"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0.0)
    data["date_value"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date_value"])
    data["month_key"] = data["date_value"].dt.strftime("%Y-%m")

    grouped = data.groupby(["submitted_by_key", "month_key"], dropna=False)["amount_value"].sum()
    return {(key[0], key[1]): float(value) for key, value in grouped.to_dict().items() if key[0] and key[1]}


def _extract_employee_names_from_expense_sheets(expense_sheets: list[pd.DataFrame]) -> list[str]:
    names: set[str] = set()
    for frame in expense_sheets:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for col in ["submitted_by", "approved_by", "employee", "employee_name"]:
            if col in frame.columns:
                for value in frame[col].dropna().astype(str).tolist():
                    cleaned = value.strip()
                    if cleaned:
                        names.add(cleaned)
    return sorted(names)


def _top_concerns(
    anomalies: list[AnomalyFinding],
    violations: list[PolicyViolation],
    limit: int = 3,
) -> list[dict[str, Any]]:
    severity_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    rows: list[dict[str, Any]] = []

    for item in anomalies:
        rows.append(
            {
                "kind": "anomaly",
                "id": item.finding_id,
                "type": item.finding_type,
                "severity": item.severity,
                "score": float(item.score),
                "summary": item.human_readable_reason,
            }
        )

    for item in violations:
        rows.append(
            {
                "kind": "policy_violation",
                "id": item.violation_id,
                "type": item.rule_name,
                "severity": str(item.severity).upper(),
                "score": 0.85 if item.severity == "HIGH" else 0.65 if item.severity == "MEDIUM" else 0.45,
                "summary": item.description,
            }
        )

    rows.sort(
        key=lambda r: (severity_rank.get(str(r.get("severity", "")).upper(), 0), float(r.get("score", 0.0))),
        reverse=True,
    )
    return rows[: max(0, limit)]
