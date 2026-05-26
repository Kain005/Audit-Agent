"""Pydantic models for extraction, analysis, and reporting flows."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class LineItem(BaseModel):
    description: str
    quantity: float | None
    unit_price: float | None
    amount: float
    gst_rate: float | None


class InvoiceEntities(BaseModel):
    vendor_name: str | None
    vendor_gst: str | None
    vendor_address: str | None = None
    invoice_number: str | None
    invoice_date: str | None
    due_date: str | None = None
    line_items: list[LineItem]
    subtotal: float | None
    gst_amount: float | None
    total_amount: float | None
    payment_terms: str | None = None
    currency: str = "INR"
    buyer_name: str | None = None 
    buyer_gst: str | None = None
    place_of_supply: str | None = None
    cgst_amount: float | None = None
    sgst_amount: float | None = None
    igst_amount: float | None = None
    raw_text: str | None


class TransactionEntity(BaseModel):
    date: str
    description: str
    party_name: str | None
    amount: float
    transaction_type: Literal["debit", "credit"]
    reference_id: str | None
    upi_details: dict[str, Any] | None
    bank_name: str | None
    category: str | None


class AnomalyFinding(BaseModel):
    finding_id: str
    document_name: str
    finding_type: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    score: float
    evidence: dict[str, Any]
    human_readable_reason: str
    explanation: str | None
    transaction_ids: list[str]


class PolicyViolation(BaseModel):
    violation_id: str
    rule_name: str
    severity: str
    description: str
    evidence: dict[str, Any]
    recommendation: str
    document_name: str
    amount_involved: float | None


class AuditReport(BaseModel):
    report_id: str
    generated_at: str
    documents_processed: int
    total_transactions: int
    gst_transaction_count: int = 0
    combined_transaction_count: int = 0
    total_invoices: int
    risk_score: float
    anomalies: list[AnomalyFinding]
    policy_violations: list[PolicyViolation]
    unmatched_invoices: list[str]
    unmatched_payments: list[str]
    summary: str | None
    document_breakdown: dict[str, int] | None = None
    top_3_concerns: list[dict[str, Any]] | None = None
    gst_invoices: list[dict[str, Any]] | None = None

