"""PDF report generation utilities for audit outputs."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any
import io
from xhtml2pdf import pisa

try:
    from ..extraction.models import AuditReport
except ImportError:  # pragma: no cover - fallback for script-style execution
    from extraction.models import AuditReport


class ReportGenerator:
    """Generate a professional HTML-based audit PDF report."""

    def clean_text(self, text) -> str:
        if not text:
            return ""
        text = str(text)
        # Replace ALL currency symbols
        text = text.replace("₹", "Rs.")
        text = text.replace("\u20b9", "Rs.")
        text = text.replace("Rs.", "Rs.")
        # Replace dashes with spaced ASCII dash
        text = text.replace("\u2014", " - ")
        text = text.replace("\u2013", " - ")
        text = text.replace("—", " - ")
        text = text.replace("–", " - ")
        # Replace other problem chars
        text = text.replace("\u2019", "'")
        text = text.replace("\u2018", "'")
        text = text.replace("\u201c", '"')
        text = text.replace("\u201d", '"')
        # Remove anything still non-ASCII
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.strip()
        return text

    def _safe(self, value: Any) -> str:
        return escape(self.clean_text(value), quote=True)

    def _as_dict(self, report: dict[str, Any] | AuditReport) -> dict[str, Any]:
        if isinstance(report, dict):
            return dict(report)

        return {
            "risk_score": getattr(report, "risk_score", 0),
            "summary": getattr(report, "summary", ""),
            "documents_processed": getattr(report, "documents_processed", 0),
            "total_transactions": getattr(report, "total_transactions", 0),
            "anomalies": list(getattr(report, "anomalies", [])),
            "policy_violations": list(getattr(report, "policy_violations", [])),
        }

    def _findings_list(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        anomalies = report.get("anomalies", []) or []
        violations = report.get("policy_violations", []) or []

        findings: list[dict[str, Any]] = []

        for anomaly in anomalies:
            finding = dict(anomaly) if isinstance(anomaly, dict) else {
                "severity": getattr(anomaly, "severity", "LOW"),
                "finding_type": getattr(anomaly, "finding_type", ""),
                "human_readable_reason": getattr(anomaly, "human_readable_reason", ""),
                "amount_involved": getattr(anomaly, "amount_involved", None),
                "description": getattr(anomaly, "description", ""),
                "rule_name": getattr(anomaly, "rule_name", ""),
                "evidence": getattr(anomaly, "evidence", {}),
            }
            findings.append(finding)

        for violation in violations:
            finding = dict(violation) if isinstance(violation, dict) else {
                "severity": getattr(violation, "severity", "LOW"),
                "rule_name": getattr(violation, "rule_name", ""),
                "description": getattr(violation, "description", ""),
                "amount_involved": getattr(violation, "amount_involved", None),
                "human_readable_reason": getattr(violation, "description", ""),
                "evidence": getattr(violation, "evidence", {}),
            }
            findings.append(finding)

        return findings

    def generate_pdf(self, report: dict | AuditReport) -> bytes:
        html = self.generate_html(report)
        buffer = io.BytesIO()
        pisa.CreatePDF(
            html.encode("utf-8"),
            dest=buffer,
            encoding="utf-8",
        )
        return buffer.getvalue()

    def generate_html(self, report: dict | AuditReport) -> str:
        report_dict = self._as_dict(report)

        anomalies_cleaned: list[dict[str, Any]] = []
        for anomaly in report_dict.get("anomalies", []) or []:
            cleaned = dict(anomaly) if isinstance(anomaly, dict) else {
                "severity": getattr(anomaly, "severity", "LOW"),
                "finding_type": getattr(anomaly, "finding_type", ""),
                "human_readable_reason": getattr(anomaly, "human_readable_reason", ""),
                "amount_involved": getattr(anomaly, "amount_involved", None),
                "description": getattr(anomaly, "description", ""),
                "evidence": getattr(anomaly, "evidence", {}),
            }
            cleaned["severity"] = self.clean_text(cleaned.get("severity", "LOW"))
            cleaned["human_readable_reason"] = self.clean_text(cleaned.get("human_readable_reason", ""))
            cleaned["finding_type"] = self.clean_text(cleaned.get("finding_type", ""))
            anomalies_cleaned.append(cleaned)

        violations_cleaned: list[dict[str, Any]] = []
        for violation in report_dict.get("policy_violations", []) or []:
            cleaned = dict(violation) if isinstance(violation, dict) else {
                "severity": getattr(violation, "severity", "LOW"),
                "rule_name": getattr(violation, "rule_name", ""),
                "description": getattr(violation, "description", ""),
                "amount_involved": getattr(violation, "amount_involved", None),
                "evidence": getattr(violation, "evidence", {}),
            }
            cleaned["severity"] = self.clean_text(cleaned.get("severity", "LOW"))
            cleaned["description"] = self.clean_text(cleaned.get("description", ""))
            cleaned["rule_name"] = self.clean_text(cleaned.get("rule_name", ""))
            violations_cleaned.append(cleaned)

        report_dict = dict(report_dict)
        report_dict["anomalies"] = anomalies_cleaned
        report_dict["policy_violations"] = violations_cleaned
        report_dict["summary"] = self.clean_text(report_dict.get("summary", ""))

        risk = float(report_dict.get("risk_score", 0) or 0)
        risk_color = "#16A34A" if risk <= 30 else "#F97316" if risk <= 60 else "#EF4444"

        all_findings = self._findings_list(report_dict)
        high = sum(1 for f in all_findings if str(f.get("severity", "")).upper() == "HIGH")
        medium = sum(1 for f in all_findings if str(f.get("severity", "")).upper() == "MEDIUM")
        low = sum(1 for f in all_findings if str(f.get("severity", "")).upper() == "LOW")

        summary = self.clean_text(report_dict.get("summary", "No summary available"))
        date = datetime.now().strftime("%d %B %Y")

        from collections import defaultdict
        import re

        type_severity = defaultdict(lambda: {"count": 0, "severity": "LOW"})
        for f in all_findings:
            ftype = self.clean_text(
                f.get("finding_type", f.get("rule_name", "unknown"))
            )
            sev = str(f.get("severity", "LOW")).upper()
            type_severity[ftype]["count"] += 1
            if sev == "HIGH":
                type_severity[ftype]["severity"] = "HIGH"
            elif sev == "MEDIUM" and type_severity[ftype]["severity"] != "HIGH":
                type_severity[ftype]["severity"] = "MEDIUM"

        findings_by_type_rows = ""
        for ftype, data in sorted(
            type_severity.items(), key=lambda x: x[1]["count"], reverse=True
        ):
            sev = data["severity"]
            color = {"HIGH": "#EF4444", "MEDIUM": "#F97316", "LOW": "#EAB308"}.get(sev, "black")
            findings_by_type_rows += f"""
    <tr>
        <td>{ftype}</td>
        <td>{data['count']}</td>
        <td style="color:{color};font-weight:bold">{sev}</td>
    </tr>
    """

        total_flagged_amount = sum(
            f.get("amount_involved", 0) or 0
            for f in all_findings
        )
        high_amount = sum(
            f.get("amount_involved", 0) or 0
            for f in all_findings if f.get("severity") == "HIGH"
        )

        transaction_summary_rows = f"""
<tr><td>Total Transactions Reviewed</td><td>{report_dict.get('total_transactions', 0)}</td></tr>
<tr><td>Flagged Transactions</td><td>{len(all_findings)}</td></tr>
<tr><td>Total Amount Involved in Findings</td><td>Rs.{total_flagged_amount:,.0f}</td></tr>
<tr><td>Amount in High Severity Findings</td><td>Rs.{high_amount:,.0f}</td></tr>
<tr><td>Risk Score</td><td>{risk}/100</td></tr>
<tr><td>Documents Reviewed</td><td>{report_dict.get('documents_processed', 0)}</td></tr>
"""

        vendors = {}
        for f in all_findings:
            reason = f.get("human_readable_reason", "")
            amount = f.get("amount_involved", 0) or 0
            match = re.search(r"Vendor '([^']+)'", reason)
            if match:
                vendor = self.clean_text(match.group(1))
                if vendor not in vendors:
                    vendors[vendor] = {"amount": 0, "count": 0}
                vendors[vendor]["amount"] += amount
                vendors[vendor]["count"] += 1

        vendor_rows = ""
        for vendor, data in vendors.items():
            vendor_rows += f"<tr><td>{vendor}</td><td>Rs.{data['amount']:,.0f}</td><td>{data['count']}</td></tr>"

        if not vendor_rows:
            vendor_rows = "<tr><td colspan='3' style='text-align:center'>No vendor-specific findings detected</td></tr>"

        risk_level = "HIGH" if risk > 60 else "MODERATE" if risk > 30 else "LOW"
        full_summary = (
            f"This financial audit analyzed {report_dict.get('documents_processed', 0)} "
            f"document(s) containing {report_dict.get('total_transactions', 0)} transactions. "
            f"The overall risk level is {risk_level} with a score of {risk}/100. "
            f"A total of {len(all_findings)} findings were identified: "
            f"{high} high severity, {medium} medium severity, and {low} low severity. "
            f"{self.clean_text(report_dict.get('summary', ''))}"
        )

        high_findings = [f for f in all_findings if str(f.get("severity", "")).upper() == "HIGH"][:3]
        concerns_list = ""
        for f in high_findings:
            reason = self.clean_text(f.get("human_readable_reason", ""))[:100]
            concerns_list += f"<li>{reason}</li>"
        if not concerns_list:
            concerns_list = "<li>No high severity concerns found</li>"

        findings_rows = ""
        for finding in all_findings:
            sev = self.clean_text(finding.get("severity", "LOW")) or "LOW"
            color = {"HIGH": "#EF4444", "MEDIUM": "#F97316", "LOW": "#EAB308"}.get(sev.upper(), "#808080")
            reason = self.clean_text(
                finding.get("human_readable_reason")
                or finding.get("description")
                or finding.get("explanation")
                or ""
            )
            ftype = self.clean_text(finding.get("finding_type") or finding.get("rule_name") or "")
            amount = self._amount(
                finding.get("amount_involved")
                or finding.get("amount")
                or (finding.get("evidence", {}) or {}).get("amount")
                or (finding.get("evidence", {}) or {}).get("invoice_value")
                or (finding.get("evidence", {}) or {}).get("tax_amount")
            )

            findings_rows += f"""
            <tr>
                <td style="color:{color};font-weight:bold">{self._safe(sev)}</td>
                <td>{self._safe(ftype)}</td>
                <td>{self._safe(reason)}</td>
                <td>Rs.{amount:,.0f}</td>
            </tr>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
<style>
    @page {{
        size: A4;
        margin: 40px;
    }}
    body {{ font-family: Arial, Helvetica, sans-serif; color: #1E293B; font-size: 10pt; }}
    h1 {{ color: #0F172A; font-size: 28px; margin: 0 0 12px 0; }}
    h2 {{ color: #0F172A; font-size: 18px; border-bottom: 2px solid #0F172A; padding-bottom: 5px; margin-top: 0; }}
    .cover {{ text-align: center; padding: 60px 0 0 0; page-break-after: always; }}
    .risk-score {{ font-size: 48px; font-weight: bold; color: {risk_color}; margin: 10px 0; }}
    .metrics {{ display: flex; gap: 12px; margin: 20px 0; flex-wrap: wrap; }}
    .metric-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 15px; flex: 1 1 120px; text-align: center; min-width: 120px; }}
    .metric-value {{ font-size: 24px; font-weight: bold; color: #0F172A; }}
    .metric-label {{ font-size: 12px; color: #64748B; }}
    .summary-box {{ background: #F0F9FF; border-left: 4px solid #3B82F6; padding: 15px; margin: 16px 0 20px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 12px; table-layout: fixed; }}
    th {{ background: #0F172A; color: white; padding: 8px; text-align: left; vertical-align: top; }}
    td {{ padding: 8px; border-bottom: 1px solid #E2E8F0; vertical-align: top; word-wrap: break-word; overflow-wrap: break-word; }}
    tr:nth-child(even) {{ background: #F8FAFC; }}
    .page-break {{ page-break-after: always; }}
    .high {{ color: #EF4444; }}
    .medium {{ color: #F97316; }}
    .low {{ color: #EAB308; }}
</style>
</head>
<body>

<div class="cover">
    <h1>FINANCIAL AUDIT REPORT</h1>
    <p>Generated: {self._safe(date)}</p>
    <div class="risk-score">{risk:.0f}/100</div>
    <p>Risk Score</p>
    <p>Prepared by AuditAI</p>
</div>

<h2>Executive Summary</h2>
<div class="summary-box">{self._safe(summary)}</div>

<div class="metrics">
    <div class="metric-card">
        <div class="metric-value">{int(report_dict.get('documents_processed', 0) or 0)}</div>
        <div class="metric-label">Documents</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{int(report_dict.get('total_transactions', 0) or 0)}</div>
        <div class="metric-label">Transactions</div>
    </div>
    <div class="metric-card">
        <div class="metric-value high">{high}</div>
        <div class="metric-label">High Risk</div>
    </div>
    <div class="metric-card">
        <div class="metric-value medium">{medium}</div>
        <div class="metric-label">Medium Risk</div>
    </div>
    <div class="metric-card">
        <div class="metric-value low">{low}</div>
        <div class="metric-label">Low Risk</div>
    </div>
</div>

<div class="page-break"></div>

<h2>Analytics</h2>

<h3>Findings by Type</h3>
<table>
    <tr>
        <th style="width:50%">Finding Type</th>
        <th style="width:25%">Count</th>
        <th style="width:25%">Severity</th>
    </tr>
    {findings_by_type_rows}
</table>

<h3>Transaction Summary</h3>
<table>
    <tr>
        <th style="width:50%">Metric</th>
        <th style="width:50%">Value</th>
    </tr>
    {transaction_summary_rows}
</table>

<h3>Flagged Vendors</h3>
<table>
    <tr>
        <th style="width:50%">Vendor</th>
        <th style="width:25%">Total Amount</th>
        <th style="width:25%">Findings</th>
    </tr>
    {vendor_rows}
</table>

<div class="page-break"></div>

<h2>Detailed Summary</h2>

<div class="summary-box">
{full_summary}
</div>

<h3>Key Concerns</h3>
<ul>
{concerns_list}
</ul>

<h2>Audit Findings</h2>
<table>
    <tr>
        <th style="width:8%">Severity</th>
        <th style="width:15%">Type</th>
        <th style="width:60%">Description</th>
        <th style="width:17%">Amount</th>
    </tr>
    {findings_rows if findings_rows else '<tr><td>LOW</td><td>No findings</td><td>No findings available</td><td></td></tr>'}
</table>

</body>
</html>"""

    def _amount(self, value: Any) -> float:
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
