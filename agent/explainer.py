"""LLM-backed explanation generation for anomalies, violations, and summaries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
from typing import Any, Generator

import requests

try:
    from groq import Groq
except ImportError:  # pragma: no cover - groq is optional at runtime
    Groq = None  # type: ignore[assignment]

try:
    from ..extraction.models import AnomalyFinding, AuditReport, PolicyViolation
except ImportError:  # pragma: no cover - fallback for script-style execution
    from extraction.models import AnomalyFinding, AuditReport, PolicyViolation

LOGGER = logging.getLogger(__name__)

# Maximum characters allowed in any evidence/transaction JSON blob sent to the LLM.
# Long evidence dicts were the primary cause of timeouts — the model was receiving
# 2000+ token prompts for simple findings. 600 chars covers all meaningful fields.
_EVIDENCE_MAX_CHARS = 600

ANOMALY_PROMPT = """
You are a financial auditor reviewing a suspicious transaction.
Explain in 2-3 plain English sentences why this is suspicious.
Then write "Investigate:" followed by 2 specific action items.
Do not speculate about intent. State only observable facts.

Transaction details:
{transaction}

Anomaly type: {anomaly_type}
Evidence: {evidence}

Response:
"""

VIOLATION_PROMPT = """
A business policy was violated. Explain what happened in 1-2 sentences
a non-accountant would understand. Then write the recommended action.

Rule violated: {rule_name}
Evidence: {evidence}
Policy: {policy_text}

Response:
"""

EXPENSE_PROMPT = """
You are a corporate expense auditor. Review this expense claim.
Identify the specific policy concern in 2 sentences.
State exactly what documentation should be checked.
Never speculate about intent.

Expense: {expense}
Policy concern: {concern}
Employee: {employee}

Response:
"""

GST_PROMPT = """
You are a GST compliance officer in India.
Explain this GST discrepancy in simple terms.
State the compliance risk and recommended action.
Reference relevant GST rules if applicable.

Discrepancy: {discrepancy}
Amount involved: {amount}

Response:
"""

VENDOR_RISK_PROMPT = """
You are a vendor due diligence analyst.
Explain why this vendor payment needs review.
List 3 specific verification steps.

Vendor: {vendor}
Concern: {concern}
Transaction: {transaction}

Response:
"""

SUMMARY_PROMPT = """
You are a financial auditor. Write a 3 sentence executive summary.
Sentence 1: Overall risk level (score: {risk_score:.0f}/100) and why.
Sentence 2: Most critical finding.
Sentence 3: Single most important action to take.
Be direct. No hedging.

Findings: {top_findings}

Response:
"""


def _truncate_json(obj: Any, max_chars: int = _EVIDENCE_MAX_CHARS) -> str:
    """Serialize *obj* to JSON and hard-truncate to *max_chars* characters.

    This prevents bloated evidence dicts from blowing out the prompt length and
    causing Ollama to queue for a very long time before responding.
    """
    serialised = json.dumps(obj, ensure_ascii=False, default=str)
    if len(serialised) <= max_chars:
        return serialised
    truncated = serialised[:max_chars]
    # Close the JSON string gracefully so it is at least readable.
    return truncated + '... [truncated]"}'


class Explainer:
    """Generate natural language explanations using Ollama or Groq."""

    # -------------------------------------------------------------------
    # Timeout constants — raised from the original 60 s which was not
    # enough for llama3.1:8b on CPU-only machines under load.
    # OLLAMA_CALL_TIMEOUT_SECONDS is now actually used in call_ollama().
    # -------------------------------------------------------------------
    OLLAMA_CALL_TIMEOUT_SECONDS = 180   # was 60 and was never applied
    OLLAMA_HEALTH_TIMEOUT_SECONDS = 3

    # Ollama is single-threaded for generation; running more than 1
    # parallel call just makes every call slower. Use 1 worker for Ollama,
    # more for Groq which is a remote API.
    _OLLAMA_BATCH_WORKERS = 1
    _GROQ_BATCH_WORKERS = 5

    def __init__(self) -> None:
        self.backend = "none"
        self._anomaly_cache: dict[str, str] = {}
        self._violation_cache: dict[str, str] = {}
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()

        ollama_available = self._is_ollama_running()
        groq_available = bool(self.groq_api_key and Groq is not None)

        if ollama_available:
            self.backend = "ollama"
        elif groq_available:
            self.backend = "groq"
        else:
            self.backend = "none"

        LOGGER.info("Explainer backend active: %s", self.backend)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _is_ollama_running(self) -> bool:
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=self.OLLAMA_HEALTH_TIMEOUT_SECONDS,
            )
            return response.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Backend calls
    # ------------------------------------------------------------------

    def call_ollama(self, prompt: str) -> str:
        """Call Ollama HTTP API — non-streaming, with correct timeout applied."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 200,   # hard cap keeps latency predictable
                    },
                },
                timeout=self.OLLAMA_CALL_TIMEOUT_SECONDS,  # was hardcoded 60, never used constant
            )
            if response.status_code == 200:
                data = response.json()
                return self._clean_response(data.get("response", "Explanation unavailable"))
            return f"Explanation unavailable (HTTP {response.status_code})"
        except requests.Timeout:
            LOGGER.warning(
                "Ollama timed out after %s s — consider increasing OLLAMA_CALL_TIMEOUT_SECONDS "
                "or switching to Groq for faster responses.",
                self.OLLAMA_CALL_TIMEOUT_SECONDS,
            )
            return "Explanation timed out — Ollama is taking too long. Try again or switch to Groq."
        except Exception as exc:
            LOGGER.error("Ollama call failed: %s", exc)
            return f"Explanation unavailable: {exc}"

    def call_ollama_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream tokens from Ollama one chunk at a time.

        Yields each token string as it arrives. The caller is responsible for
        accumulating them (e.g. for a Streamlit st.write_stream consumer).
        This gives instant feedback in the UI instead of a long silent wait.
        """
        try:
            with requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": 0.2, "num_predict": 200},
                },
                timeout=self.OLLAMA_CALL_TIMEOUT_SECONDS,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except requests.Timeout:
            yield "\n[Explanation timed out]"
        except Exception as exc:
            yield f"\n[Stream error: {exc}]"

    def call_groq(self, prompt: str) -> str:
        """Call Groq API and return model text."""
        if Groq is None:
            return "Groq error: groq library is not installed"
        if not self.groq_api_key:
            return "Groq error: GROQ_API_KEY is not configured"

        try:
            client = Groq(api_key=self.groq_api_key)
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a concise financial audit assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = completion.choices[0].message.content or ""
            return self._clean_response(content)
        except Exception as exc:
            LOGGER.error("Groq call failed: %s", exc)
            return f"Groq error: {exc}"

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Route generation to the active backend."""
        if self.backend == "ollama":
            return self.call_ollama(prompt)
        if self.backend == "groq":
            return self.call_groq(prompt)
        return "Explanation unavailable - LLM not configured"

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream generation — only supported for Ollama; falls back to single call for Groq."""
        if self.backend == "ollama":
            yield from self.call_ollama_stream(prompt)
        elif self.backend == "groq":
            # Groq is fast enough that streaming is not critical; yield full response
            yield self.call_groq(prompt)
        else:
            yield "Explanation unavailable - LLM not configured"

    def explain_anomaly(self, finding: AnomalyFinding) -> str:
        """Generate or fetch cached explanation for an anomaly finding."""
        if finding.finding_id in self._anomaly_cache:
            return self._anomaly_cache[finding.finding_id]

        # Keep the transaction payload lean — the model does not need raw IDs
        transaction_payload: dict[str, Any] = {
            "document_name": finding.document_name,
            "severity": finding.severity,
            "score": finding.score,
            "reason": finding.human_readable_reason,
        }

        prompt = self._build_anomaly_prompt(finding, transaction_payload)
        response = self.generate(prompt)
        cleaned = self._clean_response(response)
        self._anomaly_cache[finding.finding_id] = cleaned
        return cleaned

    def explain_anomaly_stream(self, finding: AnomalyFinding) -> Generator[str, None, None]:
        """Stream explanation for an anomaly finding, populating cache on completion."""
        if finding.finding_id in self._anomaly_cache:
            yield self._anomaly_cache[finding.finding_id]
            return

        transaction_payload: dict[str, Any] = {
            "document_name": finding.document_name,
            "severity": finding.severity,
            "score": finding.score,
            "reason": finding.human_readable_reason,
        }
        prompt = self._build_anomaly_prompt(finding, transaction_payload)

        accumulated: list[str] = []
        for token in self.generate_stream(prompt):
            accumulated.append(token)
            yield token
        self._anomaly_cache[finding.finding_id] = self._clean_response("".join(accumulated))

    def explain_violation(self, violation: PolicyViolation) -> str:
        """Generate or fetch cached explanation for a policy violation."""
        if violation.violation_id in self._violation_cache:
            return self._violation_cache[violation.violation_id]

        policy_text = (
            f"Rule={violation.rule_name}; Severity={violation.severity}; "
            f"Recommendation={violation.recommendation}"
        )
        prompt = VIOLATION_PROMPT.format(
            rule_name=violation.rule_name,
            evidence=_truncate_json(violation.evidence),   # was full json.dumps — could be huge
            policy_text=policy_text,
        )

        response = self.generate(prompt)
        cleaned = self._clean_response(response)
        self._violation_cache[violation.violation_id] = cleaned
        return cleaned

    def generate_summary(self, report: AuditReport) -> str:
        """Generate executive summary for an audit report."""
        if self.backend == "none":
            return self._rule_based_summary(report)

        top_findings_list: list[str] = []
        for anomaly in report.anomalies[:3]:
            top_findings_list.append(f"• {anomaly.finding_type}: {anomaly.human_readable_reason}")
        for violation in report.policy_violations[:2]:
            top_findings_list.append(f"• {violation.rule_name}: {violation.description}")

        top_findings = "\n".join(top_findings_list) if top_findings_list else "No findings"

        prompt = SUMMARY_PROMPT.format(
            risk_score=report.risk_score,
            top_findings=top_findings,
        )

        response = self.generate(prompt)
        cleaned = self._clean_response(response)
        _failed = (
            cleaned.startswith("Explanation unavailable")
            or cleaned == "Explanation timed out — Ollama is taking too long. Try again or switch to Groq."
            or cleaned == "Explanation unavailable - LLM not configured"
        )
        if _failed:
            return self._rule_based_summary(report)
        return cleaned

    def explain_batch(self, findings: list[Any]) -> list[str]:
        """Explain a list of findings, respecting backend concurrency limits.

        Ollama is a single-threaded local process — firing 5 simultaneous
        requests just queues them all and each one races against the timeout.
        We use 1 worker for Ollama and up to 5 for Groq (remote API).
        """
        if not findings:
            return []

        max_workers = (
            self._OLLAMA_BATCH_WORKERS
            if self.backend == "ollama"
            else min(self._GROQ_BATCH_WORKERS, len(findings))
        )

        def _explain_one(item: Any) -> str:
            if isinstance(item, AnomalyFinding):
                return self.explain_anomaly(item)
            if isinstance(item, PolicyViolation):
                return self.explain_violation(item)

            # Dict-style finding (legacy path)
            finding_type = str(item.get("finding_type", "unknown")) if isinstance(item, dict) else "unknown"
            prompt = ANOMALY_PROMPT.format(
                transaction=_truncate_json(item if isinstance(item, dict) else {"value": str(item)}),
                anomaly_type=finding_type,
                evidence=_truncate_json(item if isinstance(item, dict) else {}),
            )
            return self.generate(prompt)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_explain_one, findings))

        return [self._clean_response(text) for text in results]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_response(self, text: str) -> str:
        cleaned = (text or "").replace("\r\n", "\n").strip()
        cleaned = self.strip_ansi(cleaned)
        # Decode any \uXXXX escapes the LLM echoed back (e.g. \u20b9 → ₹)
        cleaned = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), cleaned)
        cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
        return cleaned

    @staticmethod
    def strip_ansi(text: str) -> str:
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def _rule_based_summary(self, report: AuditReport) -> str:
        risk_score = float(report.risk_score)
        if risk_score >= 60:
            risk_level = "High"
            recommendation = (
                "Prioritize immediate review of the highest-risk findings and pause "
                "further processing until they are resolved."
            )
        elif risk_score >= 30:
            risk_level = "Medium"
            recommendation = (
                "Review the flagged findings, validate supporting documents, and close "
                "the gaps before the next audit cycle."
            )
        else:
            risk_level = "Low"
            recommendation = (
                "Continue routine monitoring and recheck the small number of flagged "
                "items for completeness."
            )

        high_findings = [
            f"• {f.finding_type}: {f.human_readable_reason}"
            for f in report.anomalies
            if f.severity == "HIGH"
        ]
        high_findings.extend(
            f"• {v.rule_name}: {v.description}"
            for v in report.policy_violations
            if str(v.severity).upper() == "HIGH"
        )

        medium_findings = [
            f"• {f.finding_type}: {f.human_readable_reason}"
            for f in report.anomalies
            if f.severity == "MEDIUM"
        ]
        medium_findings.extend(
            f"• {v.rule_name}: {v.description}"
            for v in report.policy_violations
            if str(v.severity).upper() == "MEDIUM"
        )

        low_count = sum(
            1 for f in report.anomalies if f.severity == "LOW"
        ) + sum(
            1 for v in report.policy_violations if str(v.severity).upper() == "LOW"
        )

        summary_parts = [
            f"Overall risk level: {risk_level} with a final risk score of {risk_score:.0f}/100.",
            f"Documents analyzed: {report.documents_processed}. Total transactions: {report.total_transactions}.",
            f"High risk findings: {len(high_findings)}. Medium risk findings: {len(medium_findings)}. Low risk findings: {low_count}.",
        ]

        if high_findings:
            summary_parts.append("High risk issues: " + " ".join(high_findings))
        if medium_findings:
            summary_parts.append("Medium risk issues: " + " ".join(medium_findings))

        summary_parts.append(f"Recommendation: {recommendation}")
        return " ".join(summary_parts)

    def _build_anomaly_prompt(
        self, finding: AnomalyFinding, transaction_payload: dict[str, Any]
    ) -> str:
        finding_type = (finding.finding_type or "").lower()
        reason = finding.human_readable_reason or ""
        evidence = finding.evidence or {}

        if "expense" in finding_type:
            expense_snapshot = {
                "document_name": finding.document_name,
                "severity": finding.severity,
                "score": finding.score,
                # Evidence is the main bloat source — truncate here
                "evidence": _truncate_json(evidence),
            }
            employee = str(
                evidence.get("submitted_by")
                or evidence.get("employee")
                or evidence.get("employee_name")
                or "unknown"
            )
            return EXPENSE_PROMPT.format(
                expense=_truncate_json(expense_snapshot),
                concern=reason,
                employee=employee,
            )

        if "gst" in finding_type:
            amount = (
                evidence.get("amount")
                or evidence.get("invoice_value")
                or evidence.get("tax_amount")
                or evidence.get("taxable_value")
                or "unknown"
            )
            return GST_PROMPT.format(discrepancy=reason, amount=amount)

        if "vendor" in finding_type or "related_party" in finding_type:
            vendor = str(evidence.get("vendor") or evidence.get("vendor_name") or "unknown")
            return VENDOR_RISK_PROMPT.format(
                vendor=vendor,
                concern=reason,
                transaction=_truncate_json(transaction_payload),
            )

        return ANOMALY_PROMPT.format(
            transaction=_truncate_json(transaction_payload),
            anomaly_type=finding.finding_type,
            evidence=_truncate_json(evidence),   # was raw json.dumps — primary timeout cause
        )