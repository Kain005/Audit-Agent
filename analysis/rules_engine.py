"""Policy rules evaluation engine for invoices and transactions."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import dateparser
import yaml

try:
    from ..config.loader import get_policy
    from ..extraction.models import InvoiceEntities, PolicyViolation, TransactionEntity
except ImportError:  # pragma: no cover - fallback for script-style execution
    from config.loader import get_policy
    from extraction.models import InvoiceEntities, PolicyViolation, TransactionEntity


class RulesEngine:
    """Evaluate finance policies against extracted invoices and transactions."""

    def __init__(self) -> None:
        self.policy = get_policy()

    def check_amount_threshold(self, transaction: TransactionEntity, category: str) -> PolicyViolation | None:
        """Check category amount limit and approval threshold."""
        category_key = (category or "").strip().lower().replace(" ", "_")
        amount = float(transaction.amount)

        category_limit = getattr(self.policy.amount_thresholds, category_key, None)
        approval_limit = float(self.policy.amount_thresholds.vendor_payment_approval_above)

        if category_limit is None and amount <= approval_limit:
            return None

        category_exceeded = category_limit is not None and amount > float(category_limit)
        approval_required = amount > approval_limit

        if not category_exceeded and not approval_required:
            return None

        if category_exceeded:
            description = (
                f"Transaction amount {amount:.2f} exceeds category limit {float(category_limit):.2f} "
                f"for '{category_key}'."
            )
            severity = "HIGH"
            recommendation = "Block auto-processing and require finance manager approval."
        else:
            description = (
                f"Transaction amount {amount:.2f} exceeds approval threshold {approval_limit:.2f}."
            )
            severity = "MEDIUM"
            recommendation = "Route transaction for manual approval before payment release."

        return self._build_violation(
            rule_name="amount_threshold",
            severity=severity,
            description=description,
            evidence={
                "category": category_key,
                "amount": amount,
                "category_limit": float(category_limit) if category_limit is not None else None,
                "approval_threshold": approval_limit,
                "category_exceeded": category_exceeded,
                "approval_required": approval_required,
                "transaction_reference": transaction.reference_id,
            },
            recommendation=recommendation,
            document_name=transaction.bank_name or "transaction",
            amount_involved=amount,
        )

    def check_split_billing(self, vendor_transactions: list[TransactionEntity], approval_threshold_override: float | None = None) -> PolicyViolation | None:
        """Check split billing patterns against policy threshold and window rules.
        
        Args:
            vendor_transactions: List of transactions from the same vendor
            approval_threshold_override: Optional override for the approval threshold
        """
        if not vendor_transactions:
            return None

        min_invoices = int(self.policy.split_billing.min_invoices_to_flag)
        within_days = int(self.policy.split_billing.within_days)
        per_txn_limit = float(self.policy.split_billing.all_below_threshold)
        approval_threshold = approval_threshold_override or float(self.policy.amount_thresholds.vendor_payment_approval_above)

        sorted_txns = sorted(
            vendor_transactions,
            key=lambda txn: self._parse_date(txn.date) or datetime.min,
        )

        for start_index in range(len(sorted_txns)):
            start_txn = sorted_txns[start_index]
            start_date = self._parse_date(start_txn.date)
            if start_date is None:
                continue

            window_end = start_date + timedelta(days=within_days)
            window: list[TransactionEntity] = []

            for txn in sorted_txns[start_index:]:
                txn_date = self._parse_date(txn.date)
                if txn_date is None:
                    continue
                if txn_date <= window_end:
                    window.append(txn)

            if len(window) < min_invoices:
                continue

            amounts = [float(txn.amount) for txn in window]
            if not all(amount < per_txn_limit for amount in amounts):
                continue

            combined_amount = float(sum(amounts))
            if combined_amount <= approval_threshold:
                continue

            vendor_name = (window[0].party_name or "unknown").strip() or "unknown"
            references = [txn.reference_id for txn in window if txn.reference_id]

            return self._build_violation(
                rule_name="split_billing",
                severity="HIGH",
                description=(
                    f"{len(window)} transactions for vendor '{vendor_name}' appear split within {within_days} days; "
                    f"combined amount would be {combined_amount:.2f} which requires approval."
                ),
                evidence={
                    "vendor_name": vendor_name,
                    "window_days": within_days,
                    "transaction_count": len(window),
                    "per_transaction_limit": per_txn_limit,
                    "approval_threshold": approval_threshold,
                    "combined_amount": combined_amount,
                    "transaction_references": references,
                },
                recommendation="Aggregate related payments and enforce single approval workflow.",
                document_name=window[0].bank_name or "transaction",
                amount_involved=combined_amount,
            )

        return None

    def check_round_number(self, transaction: TransactionEntity) -> PolicyViolation | None:
        """Flag suspicious round-number transactions."""
        amount = float(transaction.amount)
        tolerance = float(self.policy.round_number_bias.tolerance)

        for suspicious in self.policy.round_number_bias.suspicious_round_amounts:
            if abs(amount - float(suspicious)) <= tolerance:
                return self._build_violation(
                    rule_name="round_number_bias",
                    severity="LOW",
                    description=f"Transaction amount {amount:.2f} matches suspicious round value {float(suspicious):.2f}.",
                    evidence={
                        "amount": amount,
                        "suspicious_value": float(suspicious),
                        "tolerance": tolerance,
                        "transaction_reference": transaction.reference_id,
                    },
                    recommendation="Review invoice/approval trail for potential amount engineering.",
                    document_name=transaction.bank_name or "transaction",
                    amount_involved=amount,
                )

        return None

    def check_new_vendor(
        self,
        vendor_name: str,
        transaction: TransactionEntity,
        known_vendors: list[str],
    ) -> PolicyViolation | None:
        """Flag high-value transactions to unseen vendors."""
        candidate = (vendor_name or "").strip().lower()
        if not candidate:
            return None

        known_set = {vendor.strip().lower() for vendor in known_vendors if isinstance(vendor, str) and vendor.strip()}
        amount = float(transaction.amount)
        scrutiny_threshold = float(self.policy.new_vendor.scrutiny_above_amount)

        if candidate in known_set or amount <= scrutiny_threshold:
            return None

        recommendation = "Perform vendor onboarding verification before payment."
        if self.policy.new_vendor.require_gst_verification:
            recommendation = "Perform vendor onboarding and GST verification before payment."

        return self._build_violation(
            rule_name="new_vendor_screening",
            severity="MEDIUM",
            description=(
                f"Vendor '{vendor_name}' is not in known vendor list and transaction amount {amount:.2f} "
                f"exceeds scrutiny threshold {scrutiny_threshold:.2f}."
            ),
            evidence={
                "vendor_name": vendor_name,
                "amount": amount,
                "scrutiny_threshold": scrutiny_threshold,
                "known_vendor": False,
                "transaction_reference": transaction.reference_id,
            },
            recommendation=recommendation,
            document_name=transaction.bank_name or "transaction",
            amount_involved=amount,
        )

    def check_duplicate_invoice(
        self,
        invoice: InvoiceEntities,
        history: list[InvoiceEntities],
    ) -> PolicyViolation | None:
        """Flag exact duplicate invoice numbers in historical records."""
        current_number = (invoice.invoice_number or "").strip()
        if not current_number:
            return None

        current_vendor = (invoice.vendor_name or "").strip().lower()

        for old in history:
            old_number = (old.invoice_number or "").strip()
            old_vendor = (old.vendor_name or "").strip().lower()
            if not old_number:
                continue

            if old_number == current_number and old_vendor == current_vendor:
                amount = invoice.total_amount if invoice.total_amount is not None else invoice.subtotal
                return self._build_violation(
                    rule_name="duplicate_invoice",
                    severity="HIGH",
                    description=(
                        f"Invoice number '{current_number}' already exists for vendor '{invoice.vendor_name or 'unknown'}'."
                    ),
                    evidence={
                        "invoice_number": current_number,
                        "vendor_name": invoice.vendor_name,
                        "historical_invoice_number": old.invoice_number,
                    },
                    recommendation="Block payment and verify whether this is a duplicate submission.",
                    document_name=current_number,
                    amount_involved=float(amount) if amount is not None else None,
                )

        return None

    def check_gst_consistency(self, invoice: InvoiceEntities) -> PolicyViolation | None:
        """Validate GST arithmetic consistency and GST format."""
        issues: list[str] = []
        evidence: dict[str, Any] = {
            "invoice_number": invoice.invoice_number,
            "vendor_name": invoice.vendor_name,
        }

        tolerance_pct = float(self.policy.gst_rules.amount_tolerance_percent)

        if invoice.total_amount is not None and invoice.subtotal is not None:
            expected_gst = float(invoice.total_amount) - float(invoice.subtotal)
            actual_gst = float(invoice.gst_amount) if invoice.gst_amount is not None else 0.0
            tolerance = abs(float(invoice.total_amount)) * (tolerance_pct / 100.0)
            diff = abs(actual_gst - expected_gst)

            evidence.update(
                {
                    "expected_gst": round(expected_gst, 2),
                    "actual_gst": round(actual_gst, 2),
                    "difference": round(diff, 2),
                    "tolerance": round(tolerance, 2),
                }
            )

            if diff > tolerance:
                issues.append("GST amount inconsistent with total - subtotal")

        if invoice.vendor_gst:
            if not re.fullmatch(self.policy.gst_rules.format_regex, invoice.vendor_gst.strip().upper()):
                issues.append("Vendor GST number format is invalid")
                evidence["vendor_gst"] = invoice.vendor_gst

        if not issues:
            return None

        amount = invoice.total_amount if invoice.total_amount is not None else invoice.subtotal
        return self._build_violation(
            rule_name="gst_consistency",
            severity="HIGH",
            description="; ".join(issues),
            evidence=evidence,
            recommendation="Recompute GST fields and validate GSTIN before accounting entry.",
            document_name=invoice.invoice_number or "invoice",
            amount_involved=float(amount) if amount is not None else None,
        )

    def check_expense_limits(self, expense: dict[str, Any], rules: Any) -> PolicyViolation | None:
        """Check expense category/day caps, employee monthly caps, and receipt threshold."""
        payload = expense or {}
        category = str(payload.get("category") or "uncategorized").strip().lower()
        employee = str(payload.get("submitted_by") or payload.get("employee") or "unknown").strip()
        amount = self._to_float(payload.get("amount"))
        amount = amount if amount is not None else 0.0

        expense_date = self._parse_date(str(payload.get("date") or ""))
        if expense_date is None:
            expense_date = datetime.utcnow()

        category_daily_limits = self._rule_map(
            rules,
            ["category_daily_limits", "expense_category_daily_limits", "per_category_daily_limits"],
            default={"travel": 15000, "food": 5000, "office": 10000, "software": 25000},
        )
        employee_monthly_limits = self._rule_map(
            rules,
            ["employee_monthly_limits", "per_employee_monthly_limits"],
            default={"default": 100000},
        )
        receipt_required_above = self._rule_number(
            rules,
            ["receipt_required_above", "receipt_mandatory_above", "receipt_threshold"],
            default=5000.0,
        )

        category_limit = self._to_float(category_daily_limits.get(category))
        if category_limit is None:
            category_limit = self._to_float(category_daily_limits.get("default"))

        if category_limit is not None and amount > category_limit:
            return self._build_violation(
                rule_name="expense_category_daily_limit",
                severity="HIGH",
                description=(
                    f"Expense amount {amount:.2f} exceeds daily limit {category_limit:.2f} for category '{category}'."
                ),
                evidence={
                    "category": category,
                    "submitted_by": employee,
                    "expense_date": expense_date.date().isoformat(),
                    "amount": amount,
                    "category_daily_limit": category_limit,
                },
                recommendation="Require manager override and supporting justification for category limit breach.",
                document_name=str(payload.get("expense_id") or payload.get("document_name") or "expense"),
                amount_involved=amount,
            )

        month_key = expense_date.strftime("%Y-%m")
        employee_monthly_spend = self._to_float(payload.get("employee_monthly_spend"))
        if employee_monthly_spend is None:
            employee_monthly_spend = amount

        employee_limit = self._to_float(employee_monthly_limits.get(employee))
        if employee_limit is None:
            employee_limit = self._to_float(employee_monthly_limits.get("default"))

        if employee_limit is not None and employee_monthly_spend > employee_limit:
            return self._build_violation(
                rule_name="expense_employee_monthly_limit",
                severity="HIGH",
                description=(
                    f"Employee '{employee}' monthly expense {employee_monthly_spend:.2f} exceeds limit {employee_limit:.2f} "
                    f"for {month_key}."
                ),
                evidence={
                    "employee": employee,
                    "month": month_key,
                    "employee_monthly_spend": employee_monthly_spend,
                    "employee_monthly_limit": employee_limit,
                    "expense_amount": amount,
                },
                recommendation="Suspend reimbursement until monthly cap approval is granted.",
                document_name=str(payload.get("expense_id") or payload.get("document_name") or "expense"),
                amount_involved=amount,
            )

        receipt_value = payload.get("receipt_attached")
        receipt_attached = self._to_bool(receipt_value)
        if amount > receipt_required_above and not receipt_attached:
            return self._build_violation(
                rule_name="expense_receipt_required",
                severity="MEDIUM",
                description=(
                    f"Expense amount {amount:.2f} exceeds receipt threshold {receipt_required_above:.2f} "
                    "but no receipt is attached."
                ),
                evidence={
                    "submitted_by": employee,
                    "category": category,
                    "amount": amount,
                    "receipt_attached": receipt_attached,
                    "receipt_threshold": receipt_required_above,
                },
                recommendation="Collect receipt before reimbursement approval.",
                document_name=str(payload.get("expense_id") or payload.get("document_name") or "expense"),
                amount_involved=amount,
            )

        return None

    def check_vendor_blacklist(self, vendor_name: str, vendor_gst: str) -> PolicyViolation | None:
        """Check vendor against blacklist by fuzzy name and exact GSTIN."""
        vendor_name_clean = (vendor_name or "").strip()
        vendor_gst_clean = (vendor_gst or "").strip().upper()
        blacklist = self._load_blacklisted_vendors()

        for entry in blacklist:
            blocked_name = str(entry.get("name") or "").strip()
            blocked_gstin = str(entry.get("gstin") or "").strip().upper()

            name_similarity = (
                SequenceMatcher(None, vendor_name_clean.lower(), blocked_name.lower()).ratio()
                if vendor_name_clean and blocked_name
                else 0.0
            )
            name_match = name_similarity > 0.9
            gst_match = bool(vendor_gst_clean and blocked_gstin and vendor_gst_clean == blocked_gstin)

            if not name_match and not gst_match:
                continue

            reason = str(entry.get("reason") or "Blacklisted vendor")
            return self._build_violation(
                rule_name="vendor_blacklist",
                severity="HIGH",
                description=(
                    f"Vendor '{vendor_name_clean or blocked_name}' matches blacklist "
                    f"({reason})."
                ),
                evidence={
                    "vendor_name": vendor_name_clean,
                    "vendor_gst": vendor_gst_clean,
                    "blacklist_name": blocked_name,
                    "blacklist_gst": blocked_gstin,
                    "blacklist_reason": reason,
                    "similarity": round(name_similarity, 4),
                    "name_match": name_match,
                    "gst_match": gst_match,
                },
                recommendation="Block payment and escalate to compliance team immediately.",
                document_name=vendor_name_clean or "vendor",
                amount_involved=None,
            )

        return None

    def check_payment_terms_violation(
        self,
        invoice: InvoiceEntities,
        payment_date: str,
    ) -> PolicyViolation | None:
        """Check payment timing against invoice terms and flag risky early/late settlement."""
        terms = (invoice.payment_terms or "").strip().lower()
        if not terms:
            return None

        match = re.search(r"net\s*(\d+)", terms, flags=re.IGNORECASE)
        if not match:
            return None

        net_days = int(match.group(1))
        invoice_dt = self._parse_date(invoice.invoice_date)
        paid_dt = self._parse_date(payment_date)
        if invoice_dt is None or paid_dt is None:
            return None

        due_dt = invoice_dt + timedelta(days=net_days)
        amount = float(invoice.total_amount) if invoice.total_amount is not None else float(invoice.subtotal or 0.0)

        if paid_dt < due_dt and amount > 50000:
            days_early = (due_dt - paid_dt).days
            return self._build_violation(
                rule_name="payment_terms_early_payment",
                severity="HIGH",
                description=(
                    f"Payment was made {days_early} day(s) before due date under Net {net_days} terms "
                    f"for amount {amount:.2f}."
                ),
                evidence={
                    "invoice_number": invoice.invoice_number,
                    "payment_terms": invoice.payment_terms,
                    "invoice_date": invoice_dt.date().isoformat(),
                    "due_date": due_dt.date().isoformat(),
                    "payment_date": paid_dt.date().isoformat(),
                    "days_early": days_early,
                    "amount": amount,
                },
                recommendation="Validate approval trail and beneficiary before releasing early high-value payments.",
                document_name=invoice.invoice_number or "invoice",
                amount_involved=amount,
            )

        grace_days = 15
        if paid_dt > due_dt + timedelta(days=grace_days):
            days_late = (paid_dt - due_dt).days
            return self._build_violation(
                rule_name="payment_terms_late_payment",
                severity="MEDIUM",
                description=(
                    f"Payment was made {days_late} day(s) after due date under Net {net_days} terms."
                ),
                evidence={
                    "invoice_number": invoice.invoice_number,
                    "payment_terms": invoice.payment_terms,
                    "invoice_date": invoice_dt.date().isoformat(),
                    "due_date": due_dt.date().isoformat(),
                    "payment_date": paid_dt.date().isoformat(),
                    "days_late": days_late,
                    "grace_days": grace_days,
                    "amount": amount,
                },
                recommendation="Review delayed payment process to prevent interest and penalty exposure.",
                document_name=invoice.invoice_number or "invoice",
                amount_involved=amount,
            )

        return None

    def check_related_party(self, vendor_name: str, employee_list: list[str]) -> PolicyViolation | None:
        """Flag potential related-party transactions by vendor-employee name similarity."""
        vendor = (vendor_name or "").strip()
        if not vendor:
            return None

        best_similarity = 0.0
        best_employee = ""

        for employee in employee_list or []:
            candidate = (employee or "").strip()
            if not candidate:
                continue
            similarity = SequenceMatcher(None, vendor.lower(), candidate.lower()).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_employee = candidate

        if best_similarity <= 0.8:
            return None

        return self._build_violation(
            rule_name="related_party_vendor",
            severity="HIGH",
            description=(
                f"Vendor name '{vendor}' is highly similar to employee '{best_employee}' "
                f"(similarity {best_similarity:.2f})."
            ),
            evidence={
                "vendor_name": vendor,
                "employee_name": best_employee,
                "similarity": round(best_similarity, 4),
                "threshold": 0.8,
            },
            recommendation="Escalate for conflict-of-interest review before releasing payment.",
            document_name=vendor,
            amount_involved=None,
        )

    def check_near_duplicate_invoice(self, invoice: InvoiceEntities, history: list[InvoiceEntities], days_window: int = 30) -> PolicyViolation | None:
        current_number = (invoice.invoice_number or "").strip()
        current_vendor = (invoice.vendor_name or "").strip().lower()
        current_amount = float(invoice.total_amount or invoice.subtotal or 0.0)
        current_date = self._parse_date(invoice.invoice_date)
        if not current_number or not current_vendor:
            return None
        for old in history:
            old_number = (old.invoice_number or "").strip()
            old_vendor = (old.vendor_name or "").strip().lower()
            if old_vendor != current_vendor or not old_number:
                continue
            if old_number == current_number:
                continue
            similarity = SequenceMatcher(None, current_number.lower(), old_number.lower()).ratio()
            if similarity < 0.8:
                continue
            old_amount = float(old.total_amount or old.subtotal or 0.0)
            amount_match = abs(current_amount - old_amount) <= max(1.0, 0.01 * max(current_amount, old_amount))
            old_date = self._parse_date(old.invoice_date)
            date_match = (
                current_date is not None and old_date is not None
                and abs((current_date - old_date).days) <= days_window
            )
            if amount_match and date_match:
                return self._build_violation(
                    rule_name="near_duplicate_invoice",
                    severity="HIGH",
                    description=f"Invoice '{current_number}' is near-duplicate of '{old_number}' for vendor '{invoice.vendor_name}' — similar number, same amount, within {days_window} days.",
                    evidence={"invoice_number": current_number, "matched_invoice": old_number, "vendor": invoice.vendor_name, "amount": current_amount, "similarity": round(similarity, 3)},
                    recommendation="Hold payment and verify whether this is a resubmission.",
                    document_name=current_number,
                    amount_involved=current_amount,
                )
        return None

    def check_same_amount_vendor_duplicate(self, invoice: InvoiceEntities, history: list[InvoiceEntities]) -> PolicyViolation | None:
        current_number = (invoice.invoice_number or "").strip()
        current_vendor = (invoice.vendor_name or "").strip().lower()
        current_amount = float(invoice.total_amount or invoice.subtotal or 0.0)
        if not current_vendor or current_amount <= 0:
            return None
        for old in history:
            old_number = (old.invoice_number or "").strip()
            old_vendor = (old.vendor_name or "").strip().lower()
            if old_vendor != current_vendor or old_number == current_number:
                continue
            old_amount = float(old.total_amount or old.subtotal or 0.0)
            if abs(current_amount - old_amount) <= max(1.0, 0.01 * max(current_amount, old_amount)):
                return self._build_violation(
                    rule_name="same_amount_vendor_duplicate",
                    severity="MEDIUM",
                    description=f"Vendor '{invoice.vendor_name}' has two invoices with same amount {current_amount:.2f} but different numbers: '{current_number}' and '{old_number}'.",
                    evidence={"invoice_number": current_number, "matched_invoice": old_number, "vendor": invoice.vendor_name, "amount": current_amount},
                    recommendation="Verify both invoices represent distinct deliveries before payment.",
                    document_name=current_number,
                    amount_involved=current_amount,
                )
        return None

    def check_gstin_name_mismatch(self, invoice: InvoiceEntities, history: list[InvoiceEntities]) -> PolicyViolation | None:
        current_gstin = (invoice.vendor_gst or "").strip().upper()
        current_name = (invoice.vendor_name or "").strip().lower()
        if not current_gstin or not current_name:
            return None
        for old in history:
            old_gstin = (old.vendor_gst or "").strip().upper()
            old_name = (old.vendor_name or "").strip().lower()
            if old_gstin != current_gstin or not old_name:
                continue
            similarity = SequenceMatcher(None, current_name, old_name).ratio()
            if similarity < 0.85:
                return self._build_violation(
                    rule_name="gstin_name_mismatch",
                    severity="HIGH",
                    description=f"GSTIN '{current_gstin}' appears with different vendor names: '{invoice.vendor_name}' vs '{old.vendor_name}'.",
                    evidence={"gstin": current_gstin, "name_1": invoice.vendor_name, "name_2": old.vendor_name, "similarity": round(similarity, 3)},
                    recommendation="Verify GSTIN ownership — possible identity fraud or data error.",
                    document_name=invoice.invoice_number or "invoice",
                    amount_involved=float(invoice.total_amount or invoice.subtotal or 0.0) or None,
                )
        return None

    def check_vendor_sudden_activity(self, invoice: InvoiceEntities, history: list[InvoiceEntities], window_days: int = 30, min_invoices: int = 3) -> PolicyViolation | None:
        current_vendor = (invoice.vendor_name or "").strip().lower()
        if not current_vendor:
            return None
        current_date = self._parse_date(invoice.invoice_date)
        if current_date is None:
            return None
        window_start = current_date - timedelta(days=window_days)
        vendor_invoices = [
            old for old in history
            if (old.vendor_name or "").strip().lower() == current_vendor
            and self._parse_date(old.invoice_date) is not None
            and self._parse_date(old.invoice_date) >= window_start
        ]
        if len(vendor_invoices) + 1 >= min_invoices:
            older_history = [
                old for old in history
                if (old.vendor_name or "").strip().lower() == current_vendor
                and self._parse_date(old.invoice_date) is not None
                and self._parse_date(old.invoice_date) < window_start
            ]
            if not older_history:
                total = len(vendor_invoices) + 1
                return self._build_violation(
                    rule_name="vendor_sudden_activity",
                    severity="MEDIUM",
                    description=f"Vendor '{invoice.vendor_name}' has {total} invoices in {window_days} days with no prior history.",
                    evidence={"vendor": invoice.vendor_name, "invoice_count": total, "window_days": window_days},
                    recommendation="Perform vendor due diligence before processing further payments.",
                    document_name=invoice.invoice_number or "invoice",
                    amount_involved=float(invoice.total_amount or invoice.subtotal or 0.0) or None,
                )
        return None

    def check_one_time_vendor(self, invoice: InvoiceEntities, history: list[InvoiceEntities], high_value_threshold: float = 50000.0) -> PolicyViolation | None:
        current_vendor = (invoice.vendor_name or "").strip().lower()
        current_gstin = (invoice.vendor_gst or "").strip()
        current_amount = float(invoice.total_amount or invoice.subtotal or 0.0)
        if not current_vendor or current_amount <= high_value_threshold:
            return None
        prior = [old for old in history if (old.vendor_name or "").strip().lower() == current_vendor]
        if prior:
            return None
        if current_gstin:
            return None
        return self._build_violation(
            rule_name="one_time_vendor",
            severity="HIGH",
            description=f"Vendor '{invoice.vendor_name}' appears for the first time with high-value invoice {current_amount:.2f} and no GSTIN.",
            evidence={"vendor": invoice.vendor_name, "amount": current_amount, "gstin": current_gstin or None, "threshold": high_value_threshold},
            recommendation="Require vendor registration, GSTIN, and manager approval before payment.",
            document_name=invoice.invoice_number or "invoice",
            amount_involved=current_amount,
        )

    def check_frequency_anomaly(self, invoice: InvoiceEntities, history: list[InvoiceEntities], spike_multiplier: float = 3.0) -> PolicyViolation | None:
        current_vendor = (invoice.vendor_name or "").strip().lower()
        current_date = self._parse_date(invoice.invoice_date)
        if not current_vendor or current_date is None:
            return None
        current_month = current_date.strftime("%Y-%m")
        monthly_counts: dict[str, int] = defaultdict(int)
        for old in history:
            if (old.vendor_name or "").strip().lower() != current_vendor:
                continue
            old_date = self._parse_date(old.invoice_date)
            if old_date is None:
                continue
            monthly_counts[old_date.strftime("%Y-%m")] += 1
        monthly_counts[current_month] += 1
        current_count = monthly_counts[current_month]
        other_counts = [v for k, v in monthly_counts.items() if k != current_month]
        if not other_counts:
            return None
        avg = sum(other_counts) / len(other_counts)
        if avg > 0 and current_count >= spike_multiplier * avg:
            return self._build_violation(
                rule_name="invoice_frequency_anomaly",
                severity="MEDIUM",
                description=f"Vendor '{invoice.vendor_name}' has {current_count} invoices in {current_month} vs average {avg:.1f}/month — {spike_multiplier}x spike.",
                evidence={"vendor": invoice.vendor_name, "current_month": current_month, "current_count": current_count, "historical_avg": round(avg, 2), "multiplier": spike_multiplier},
                recommendation="Review invoice burst for signs of fraudulent billing.",
                document_name=invoice.invoice_number or "invoice",
                amount_involved=float(invoice.total_amount or invoice.subtotal or 0.0) or None,
            )
        return None

    def check_amount_clustering(self, invoice: InvoiceEntities, history: list[InvoiceEntities], threshold_buffer: float = 0.05) -> PolicyViolation | None:
        approval_threshold = float(self.policy.amount_thresholds.vendor_payment_approval_above)
        current_vendor = (invoice.vendor_name or "").strip().lower()
        current_amount = float(invoice.total_amount or invoice.subtotal or 0.0)
        lower_bound = approval_threshold * (1.0 - threshold_buffer)
        if not current_vendor or not (lower_bound <= current_amount < approval_threshold):
            return None
        clustered = [
            old for old in history
            if (old.vendor_name or "").strip().lower() == current_vendor
            and lower_bound <= float(old.total_amount or old.subtotal or 0.0) < approval_threshold
        ]
        if len(clustered) >= 1:
            count = len(clustered) + 1
            return self._build_violation(
                rule_name="amount_clustering",
                severity="HIGH",
                description=f"Vendor '{invoice.vendor_name}' has {count} invoices clustered just below approval threshold {approval_threshold:.2f}.",
                evidence={"vendor": invoice.vendor_name, "invoice_count": count, "approval_threshold": approval_threshold, "current_amount": current_amount, "buffer_pct": threshold_buffer * 100},
                recommendation="Aggregate invoices and enforce single approval — likely threshold avoidance.",
                document_name=invoice.invoice_number or "invoice",
                amount_involved=current_amount,
            )
        return None

    def check_unusually_high_invoice(self, invoice: InvoiceEntities, history: list[InvoiceEntities], z_threshold: float = 2.5) -> PolicyViolation | None:
        import statistics
        current_amount = float(invoice.total_amount or invoice.subtotal or 0.0)
        if current_amount <= 0:
            return None
        amounts = [
            float(old.total_amount or old.subtotal or 0.0)
            for old in history
            if float(old.total_amount or old.subtotal or 0.0) > 0
        ]
        if len(amounts) < 3:
            return None
        mean = statistics.mean(amounts)
        stdev = statistics.stdev(amounts)
        if stdev == 0:
            return None
        z = (current_amount - mean) / stdev
        if z >= z_threshold:
            return self._build_violation(
                rule_name="unusually_high_invoice",
                severity="HIGH",
                description=f"Invoice amount {current_amount:.2f} is {z:.1f} standard deviations above mean {mean:.2f} for vendor '{invoice.vendor_name}'.",
                evidence={"vendor": invoice.vendor_name, "amount": current_amount, "mean": round(mean, 2), "stdev": round(stdev, 2), "z_score": round(z, 3), "threshold": z_threshold},
                recommendation="Request additional approval and supporting documentation for outlier invoice.",
                document_name=invoice.invoice_number or "invoice",
                amount_involved=current_amount,
            )
        return None

    def check_cgst_sgst_split(self, invoice: InvoiceEntities) -> PolicyViolation | None:
        cgst = invoice.cgst_amount
        sgst = invoice.sgst_amount
        if cgst is None or sgst is None:
            return None
        if cgst <= 0 and sgst <= 0:
            return None
        tolerance = max(1.0, 0.01 * max(abs(cgst), abs(sgst)))
        if abs(cgst - sgst) > tolerance:
            return self._build_violation(
                rule_name="cgst_sgst_split_mismatch",
                severity="HIGH",
                description=f"CGST {cgst:.2f} and SGST {sgst:.2f} are not equal — they must be equal for intra-state supply.",
                evidence={"invoice_number": invoice.invoice_number, "vendor": invoice.vendor_name, "cgst": cgst, "sgst": sgst, "difference": round(abs(cgst - sgst), 2)},
                recommendation="Correct tax split before filing — CGST must equal SGST for intra-state transactions.",
                document_name=invoice.invoice_number or "invoice",
                amount_involved=float(invoice.total_amount or invoice.subtotal or 0.0) or None,
            )
        return None

    def check_interstate_tax_type(self, invoice: InvoiceEntities) -> PolicyViolation | None:
        vendor_gstin = (invoice.vendor_gst or "").strip().upper()
        place_of_supply = (invoice.place_of_supply or "").strip().upper()
        if not vendor_gstin or not place_of_supply or len(vendor_gstin) < 2:
            return None

        vendor_state = vendor_gstin[:2]
        pos_match = re.search(r"\b(\d{2})\b", place_of_supply)
        if pos_match:
            supply_state = pos_match.group(1)
        else:
            supply_state = place_of_supply[:2]

        if not supply_state or len(supply_state) != 2 or not supply_state.isdigit():
            return None

        is_interstate = vendor_state != supply_state
        has_igst = (invoice.igst_amount or 0.0) > 0
        has_cgst_sgst = (invoice.cgst_amount or 0.0) > 0 or (invoice.sgst_amount or 0.0) > 0

        if is_interstate and has_cgst_sgst and not has_igst:
            return self._build_violation(
                rule_name="interstate_tax_type_mismatch",
                severity="HIGH",
                description=(
                    f"Invoice '{invoice.invoice_number or 'invoice'}' appears interstate based on GSTIN '{vendor_gstin}' and place of supply '{place_of_supply}', but CGST/SGST are used instead of IGST."
                ),
                evidence={
                    "vendor_gstin": vendor_gstin,
                    "place_of_supply": place_of_supply,
                    "vendor_state": vendor_state,
                    "supply_state": supply_state,
                    "igst_amount": invoice.igst_amount,
                    "cgst_amount": invoice.cgst_amount,
                    "sgst_amount": invoice.sgst_amount,
                },
                recommendation="Use IGST for interstate supply and correct the GST breakdown.",
                document_name=invoice.invoice_number or "invoice",
                amount_involved=float(invoice.total_amount or invoice.subtotal or 0.0) or None,
            )

        if not is_interstate and has_igst and not has_cgst_sgst:
            return self._build_violation(
                rule_name="intrastate_tax_type_mismatch",
                severity="HIGH",
                description=(
                    f"Invoice '{invoice.invoice_number or 'invoice'}' appears intra-state based on GSTIN '{vendor_gstin}' and place of supply '{place_of_supply}', but IGST is used instead of CGST/SGST."
                ),
                evidence={
                    "vendor_gstin": vendor_gstin,
                    "place_of_supply": place_of_supply,
                    "vendor_state": vendor_state,
                    "supply_state": supply_state,
                    "igst_amount": invoice.igst_amount,
                    "cgst_amount": invoice.cgst_amount,
                    "sgst_amount": invoice.sgst_amount,
                },
                recommendation="Use CGST and SGST for intra-state supply and correct the GST breakdown.",
                document_name=invoice.invoice_number or "invoice",
                amount_involved=float(invoice.total_amount or invoice.subtotal or 0.0) or None,
            )

        return None

    def evaluate_invoice(self, invoice: InvoiceEntities, history: list[InvoiceEntities], approval_threshold: float | None = None) -> list[PolicyViolation]:
        """Evaluate invoice-level policy checks."""
        violations: list[PolicyViolation] = []

        duplicate = self.check_duplicate_invoice(invoice, history)
        if duplicate:
            violations.append(duplicate)

        near_dup = self.check_near_duplicate_invoice(invoice, history)
        if near_dup:
            violations.append(near_dup)

        same_amount_dup = self.check_same_amount_vendor_duplicate(invoice, history)
        if same_amount_dup:
            violations.append(same_amount_dup)

        gstin_mismatch = self.check_gstin_name_mismatch(invoice, history)
        if gstin_mismatch:
            violations.append(gstin_mismatch)

        sudden_activity = self.check_vendor_sudden_activity(invoice, history)
        if sudden_activity:
            violations.append(sudden_activity)

        one_time = self.check_one_time_vendor(invoice, history)
        if one_time:
            violations.append(one_time)

        freq_anomaly = self.check_frequency_anomaly(invoice, history)
        if freq_anomaly:
            violations.append(freq_anomaly)

        clustering = self.check_amount_clustering(invoice, history)
        if clustering:
            violations.append(clustering)

        high_invoice = self.check_unusually_high_invoice(invoice, history)
        if high_invoice:
            violations.append(high_invoice)

        gst_violation = self.check_gst_consistency(invoice)
        if gst_violation:
            violations.append(gst_violation)

        cgst_sgst = self.check_cgst_sgst_split(invoice)
        if cgst_sgst:
            violations.append(cgst_sgst)

        interstate = self.check_interstate_tax_type(invoice)
        if interstate:
            violations.append(interstate)

        return violations

    def evaluate_transaction(
        self,
        transaction: TransactionEntity,
        known_vendors: list[str],
        history: list[Any],
        approval_threshold: float | None = None,
    ) -> list[PolicyViolation]:
        """Evaluate transaction-level policy checks."""
        _ = history  # reserved for future transaction history rules

        violations: list[PolicyViolation] = []

        category = transaction.category or ""
        amount_violation = self.check_amount_threshold(transaction, category)
        if amount_violation:
            violations.append(amount_violation)

        round_violation = self.check_round_number(transaction)
        if round_violation:
            violations.append(round_violation)

        vendor_violation = self.check_new_vendor(transaction.party_name or "", transaction, known_vendors)
        if vendor_violation:
            violations.append(vendor_violation)

        return violations

    def evaluate_all(
        self,
        invoices: list[InvoiceEntities],
        transactions: list[TransactionEntity] | list[dict[str, Any]],
        known_vendors: list[str],
        approval_threshold: float | None = None,
    ) -> list[PolicyViolation]:
        """Run invoice and transaction evaluators and return deduplicated violations.
        
        Args:
            invoices: List of invoices to evaluate
            transactions: List of transactions to evaluate
            known_vendors: List of known vendor names
            approval_threshold: Optional override for payment approval threshold
        """
        all_violations: list[PolicyViolation] = []

        normalized_transactions = [
            self._transaction_from_record(transaction) if isinstance(transaction, dict) else transaction
            for transaction in transactions
        ]

        invoice_history: list[InvoiceEntities] = []
        for invoice in invoices:
            all_violations.extend(self.evaluate_invoice(invoice, invoice_history, approval_threshold))
            invoice_history.append(invoice)

        transaction_history: list[TransactionEntity] = []
        for transaction in normalized_transactions:
            all_violations.extend(self.evaluate_transaction(transaction, known_vendors, transaction_history, approval_threshold))
            transaction_history.append(transaction)

        vendor_groups: dict[str, list[TransactionEntity]] = defaultdict(list)
        for txn in normalized_transactions:
            vendor_key = self._extract_vendor_token(txn.party_name or txn.description or "")
            vendor_groups[vendor_key].append(txn)

        for vendor_txns in vendor_groups.values():
            split_violation = self.check_split_billing(vendor_txns, approval_threshold)
            if split_violation:
                all_violations.append(split_violation)

        return self._deduplicate_violations(all_violations)

    def _transaction_from_record(self, record: dict[str, Any]) -> TransactionEntity:
        amount = self._to_float(record.get("amount"))
        debit = record.get("debit")
        credit = record.get("credit")

        # Convert None/NaN values to 0 before numeric comparisons.
        try:
            debit = float(debit) if debit is not None and str(debit) != "nan" else 0.0
        except (ValueError, TypeError):
            debit = 0.0

        try:
            credit = float(credit) if credit is not None and str(credit) != "nan" else 0.0
        except (ValueError, TypeError):
            credit = 0.0

        if credit > 0 and credit >= debit:
            transaction_type = "credit"
            amount = credit
        else:
            transaction_type = "debit"
            amount = debit if debit > 0 else amount

        return TransactionEntity(
            date=str(record.get("date") or ""),
            description=str(record.get("description") or ""),
            party_name=(str(record.get("party_name") or "").strip() or None),
            amount=float(amount),
            transaction_type=transaction_type,
            reference_id=(str(record.get("transaction_id") or "").strip() or None),
            upi_details=None,
            bank_name=(str(record.get("bank_name") or record.get("document_name") or "").strip() or None),
            category=(str(record.get("category") or "").strip() or None),
        )

    def _to_float(self, value: Any) -> float:
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

    def _deduplicate_violations(self, violations: list[PolicyViolation]) -> list[PolicyViolation]:
        seen: dict[tuple[str, str, str], PolicyViolation] = {}

        severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        for violation in violations:
            key = (
                violation.rule_name,
                violation.document_name,
                violation.description,
            )
            current = seen.get(key)
            if current is None:
                seen[key] = violation
                continue

            if severity_rank.get(violation.severity, 0) > severity_rank.get(current.severity, 0):
                seen[key] = violation

        ordered = list(seen.values())
        ordered.sort(key=lambda v: (severity_rank.get(v.severity, 0), v.rule_name), reverse=True)
        return ordered

    def _build_violation(
        self,
        rule_name: str,
        severity: str,
        description: str,
        evidence: dict[str, Any],
        recommendation: str,
        document_name: str,
        amount_involved: float | None,
    ) -> PolicyViolation:
        return PolicyViolation(
            violation_id=str(uuid.uuid4()),
            rule_name=rule_name,
            severity=severity,
            description=description,
            evidence=evidence,
            recommendation=recommendation,
            document_name=document_name,
            amount_involved=amount_involved,
        )

    def _parse_date(self, value: str | None) -> datetime | None:
        if not value:
            return None
        return dateparser.parse(
            value,
            settings={"DATE_ORDER": "DMY", "PREFER_LOCALE_DATE_ORDER": False, "STRICT_PARSING": False},
        )

    def _extract_vendor_token(self, text: str) -> str:
        value = str(text or "").strip().upper()
        if not value:
            return "unknown"

        def _clean_token(token: str) -> str:
            return re.sub(r"[^A-Z]", "", token.strip().upper())

        if "UPI" in value:
            for token in reversed([part.strip() for part in value.split("/") if part.strip()]):
                cleaned = _clean_token(token)
                if len(cleaned) >= 3:
                    return cleaned

        if "NEFT" in value or "IMPS" in value:
            for token in reversed([part.strip() for part in value.split() if part.strip()]):
                cleaned = _clean_token(token)
                if len(cleaned) >= 3:
                    return cleaned

        if "ACH" in value:
            return "ACH"

        if "CHEQUE" in value:
            return "CHEQUE"

        first_token = value.split()[0].strip()
        cleaned_first = _clean_token(first_token)
        if len(cleaned_first) >= 3:
            return cleaned_first

        return "unknown"

    def _load_blacklisted_vendors(self) -> list[dict[str, Any]]:
        blacklist_path = Path(__file__).resolve().parent.parent / "config" / "blacklisted_vendors.yaml"
        if not blacklist_path.exists():
            return []

        with blacklist_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or []

        if isinstance(loaded, dict):
            maybe_list = loaded.get("blacklisted_vendors", [])
            return maybe_list if isinstance(maybe_list, list) else []

        return loaded if isinstance(loaded, list) else []

    def _rule_map(self, rules: Any, keys: list[str], default: dict[str, Any]) -> dict[str, Any]:
        if isinstance(rules, dict):
            for key in keys:
                value = rules.get(key)
                if isinstance(value, dict):
                    return value
            expense_rules = rules.get("expense_limits")
            if isinstance(expense_rules, dict):
                for key in keys:
                    value = expense_rules.get(key)
                    if isinstance(value, dict):
                        return value
            return default

        for key in keys:
            value = getattr(rules, key, None)
            if isinstance(value, dict):
                return value
        expense_rules = getattr(rules, "expense_limits", None)
        if expense_rules is not None:
            for key in keys:
                value = getattr(expense_rules, key, None)
                if isinstance(value, dict):
                    return value
        return default

    def _rule_number(self, rules: Any, keys: list[str], default: float) -> float:
        if isinstance(rules, dict):
            for key in keys:
                value = self._to_float(rules.get(key))
                if value is not None:
                    return value
            expense_rules = rules.get("expense_limits")
            if isinstance(expense_rules, dict):
                for key in keys:
                    value = self._to_float(expense_rules.get(key))
                    if value is not None:
                        return value
            return default

        for key in keys:
            value = self._to_float(getattr(rules, key, None))
            if value is not None:
                return value
        expense_rules = getattr(rules, "expense_limits", None)
        if expense_rules is not None:
            for key in keys:
                value = self._to_float(getattr(expense_rules, key, None))
                if value is not None:
                    return value
        return default

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            text = str(value).strip().replace(",", "")
            if not text:
                return None
            try:
                return float(text)
            except Exception:
                return None

    def _to_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"yes", "y", "true", "1", "attached"}
