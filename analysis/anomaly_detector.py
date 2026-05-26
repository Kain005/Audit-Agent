"""Anomaly detection module for financial audit transactions."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

try:
    from ..extraction.models import AnomalyFinding
except ImportError:  # pragma: no cover - fallback for script-style execution
    from extraction.models import AnomalyFinding


@dataclass(frozen=True)
class _SplitBillingRules:
    min_invoices_to_flag: int
    within_days: int
    all_below_threshold: float


class AnomalyDetector:
    """Detect anomalies across amount, timing, frequency, split-billing, and multivariate patterns."""

    HOLIDAY_MONTH_DAYS = {
        "01-01",  # New Year
        "01-26",  # Republic Day
        "03-14",  # Holi (approx)
        "03-31",  # Eid (approx)
        "08-15",  # Independence Day
        "10-02",  # Gandhi Jayanti
        "10-20",  # Diwali (approx)
        "12-25",  # Christmas
    }

    # Transaction type patterns for categorization
    TRANSACTION_PATTERNS = {
        "SALARY": ["salary", "sal cr", "payroll"],
        "EMI": ["emi", "loan", "equated"],
        "REFUND": ["refund", "rev", "reversal", "cancelled"],
        "TRANSFER": ["trf", "transfer", "self"],
        "INTEREST": ["int cr", "interest credit"],
        "TDS": ["tds", "tax deducted"],
        "CHARGES": ["charges", "fee", "annual fee"],
    }

    def tag_transaction_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag transaction rows into coarse categories based on description text."""
        if "description" not in df.columns:
            return df

        df = df.copy()
        desc = df["description"].astype(str).str.lower()

        df["transaction_category"] = "other"

        df.loc[
            desc.str.contains("salary|sal cr|payroll|ctc", na=False),
            "transaction_category",
        ] = "SALARY"

        df.loc[
            desc.str.contains("emi|loan|equated|repayment", na=False),
            "transaction_category",
        ] = "EMI"

        df.loc[
            desc.str.contains("refund|rev|reversal|cancelled|cancel", na=False),
            "transaction_category",
        ] = "REFUND"

        df.loc[
            desc.str.contains("trf|transfer|self|own account", na=False),
            "transaction_category",
        ] = "TRANSFER"

        df.loc[
            desc.str.contains("int cr|interest credit|interest earned", na=False),
            "transaction_category",
        ] = "INTEREST"

        df.loc[
            desc.str.contains("tds|tax deducted|tax deduction", na=False),
            "transaction_category",
        ] = "TDS"

        df.loc[
            desc.str.contains("charges|fee|annual fee|service charge", na=False),
            "transaction_category",
        ] = "CHARGES"

        return df

    def filter_known_transaction_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction_category column based on description keywords.
        
        Categories: SALARY, EMI, REFUND, TRANSFER, INTEREST, TDS, CHARGES, UNKNOWN
        
        Args:
            df: Transaction dataframe
            
        Returns:
            DataFrame with added 'transaction_category' column
        """
        if "description" in df.columns:
            return self.tag_transaction_categories(df)

        df = df.copy()
        
        # Find description column
        description_cols = [col for col in df.columns if any(
            keyword in str(col).lower() for keyword in ["description", "narration", "remarks", "particulars"]
        )]
        
        if not description_cols:
            df["transaction_category"] = "UNKNOWN"
            return df
        
        description_col = description_cols[0]
        
        def categorize_transaction(desc):
            if pd.isna(desc) or not str(desc).strip():
                return "UNKNOWN"
            
            desc_lower = str(desc).lower()
            
            # Check each category pattern
            for category, patterns in self.TRANSACTION_PATTERNS.items():
                for pattern in patterns:
                    if pattern in desc_lower:
                        return category
            
            return "UNKNOWN"
        
        df["transaction_category"] = df[description_col].apply(categorize_transaction)
        return df

    def detect_amount_outliers(self, df: pd.DataFrame, category_col: str = "category") -> list[AnomalyFinding]:
        """Detect amount outliers using z-score and IQR by category.
        
        Skips SALARY transactions (legitimate recurring income).
        """
        data = self._prepare_dataframe(df)

        # Exclude transaction categories not useful for outlier checks.
        if "transaction_category" in data.columns:
            data = data[~data["transaction_category"].isin(["SALARY", "INTEREST", "CHARGES"])].copy()
        
        findings: list[AnomalyFinding] = []

        if data.empty:
            return findings

        if category_col not in data.columns:
            data[category_col] = "uncategorized"

        grouped = data.groupby(category_col, dropna=False)

        for category, group in grouped:
            if len(group) < 3:
                continue

            amounts = group["amount"].astype(float)
            if amounts.nunique() <= 1:
                continue

            z_scores = np.abs(stats.zscore(amounts, nan_policy="omit"))
            q1 = amounts.quantile(0.25)
            q3 = amounts.quantile(0.75)
            iqr = q3 - q1
            iqr_threshold = q3 + (3 * iqr)

            for idx, row in group.iterrows():
                z = float(z_scores[group.index.get_loc(idx)]) if not np.isnan(z_scores[group.index.get_loc(idx)]) else 0.0
                amount = float(row["amount"])

                z_flag = z > 3
                iqr_flag = amount > iqr_threshold

                if not z_flag and not iqr_flag:
                    continue

                severity = "HIGH" if z >4 else "MEDIUM"
                score = min(1.0, max(0.0, z / 6.0)) if z_flag else 0.7

                findings.append(
                    self._build_finding(
                        row=row,
                        finding_type="amount_outlier",
                        severity=severity,
                        score=score,
                        evidence={
                            "category": self._safe_value(category),
                            "amount": amount,
                            "z_score": round(z, 4),
                            "iqr_threshold": round(float(iqr_threshold), 2),
                            "z_method_flag": z_flag,
                            "iqr_method_flag": iqr_flag,
                        },
                        reason=(
                            "Transaction amount is anomalous within category based on "
                            f"{'z-score' if z_flag else ''}{' and ' if z_flag and iqr_flag else ''}{'IQR' if iqr_flag else ''}."
                        ),
                    )
                )

        return findings

    def detect_timing_anomalies(self, df: pd.DataFrame) -> list[AnomalyFinding]:
        """Detect outside-hours, weekend, and holiday transactions.
        
        Skips TDS transactions (system-initiated tax deductions).
        """
        data = self._prepare_dataframe(df)

        # Exclude system-driven entries from timing anomaly checks.
        if "transaction_category" in data.columns:
            data = data[~data["transaction_category"].isin(["TDS", "CHARGES", "INTEREST"])].copy()
        
        findings: list[AnomalyFinding] = []

        if data.empty:
            return findings

        # Check if time data actually exists.
        # If all hours are 0, the CSV has no time component — skip hour check.
        data["parsed_date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
        valid_dates = data["parsed_date"].dropna()
        if len(valid_dates) == 0:
            has_time_data = False
        else:
            all_midnight = (valid_dates.dt.hour == 0).all()
            midnight_pct = (valid_dates.dt.hour == 0).sum() / len(valid_dates)
            has_time_data = (not all_midnight) and (midnight_pct < 0.9)

        weekend_amount_threshold = 5000.0
        large_amount_threshold = data["amount"].quantile(0.90) if len(data) > 10 else data["amount"].max()

        for _, row in data.iterrows():
            txn_date = row["parsed_date"]
            day_name = txn_date.strftime("%A")
            hour = int(txn_date.hour)
            amount = float(row["amount"])
            day_of_week = txn_date.weekday()
            # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
            is_weekend = day_of_week in [5, 6]
            month_day = txn_date.strftime("%m-%d")
            is_holiday = month_day in self.HOLIDAY_MONTH_DAYS
            outside_hours = has_time_data and (hour < 9 or hour > 18)
            if is_weekend and amount < 10000:
                continue
            weekend_flag = is_weekend and amount >= 10000

            if not (outside_hours or weekend_flag or is_holiday):
                continue

            severe = weekend_flag and amount >= float(large_amount_threshold)
            severity = "HIGH" if severe else "MEDIUM"
            score = 0.9 if severe else 0.65

            if is_holiday and is_weekend:
                reason = f"Payment of Rs.{amount:,.0f} made on {day_name} -- public holiday and weekend"
            elif is_holiday:
                reason = f"Payment of Rs.{amount:,.0f} made on {day_name} -- public holiday"
            elif is_weekend:
                reason = f"Payment of Rs.{amount:,.0f} made on {day_name} -- weekend payment above Rs.10,000"
            elif outside_hours:
                reason = f"Payment of Rs.{amount:,.0f} made at {hour}:00 -- outside business hours (9am-6pm)"
            else:
                reason = f"Payment of Rs.{amount:,.0f} made on {day_name} at {hour}:00 -- unusual timing"

            findings.append(
                self._build_finding(
                    row=row,
                    finding_type="timing_anomaly",
                    severity=severity,
                    score=score,
                    evidence={
                        "hour": hour,
                        "outside_business_hours": outside_hours,
                        "is_weekend": is_weekend,
                        "weekend_flag": weekend_flag,
                        "is_public_holiday": is_holiday,
                        "amount": amount,
                        "large_amount_threshold": float(large_amount_threshold),
                    },
                    reason=reason,
                )
            )

        return findings

    def detect_frequency_anomalies(self, df: pd.DataFrame) -> list[AnomalyFinding]:
        """Detect vendors with unusual current-month transaction frequency."""
        data = self._prepare_dataframe(df)

        if "transaction_category" in data.columns:
            data = data[~data["transaction_category"].isin(["SALARY", "EMI", "INTEREST"])].copy()

        findings: list[AnomalyFinding] = []

        if data.empty:
            return findings

        vendor_col = self._resolve_vendor_column(data)
        data["vendor_key"] = data[vendor_col].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        data["year_month"] = data["parsed_date"].dt.to_period("M")

        if data["year_month"].empty:
            return findings

        current_month = data["year_month"].max()
        monthly_counts = data.groupby(["vendor_key", "year_month"]).size().rename("count").reset_index()

        vendor_stats = monthly_counts.groupby("vendor_key")["count"].agg(["mean", "std"]).fillna(0.0)
        current_counts = (
            monthly_counts[monthly_counts["year_month"] == current_month]
            .set_index("vendor_key")["count"]
            .to_dict()
        )

        for vendor, current_count in current_counts.items():
            mean_count = float(vendor_stats.loc[vendor, "mean"])
            std_count = float(vendor_stats.loc[vendor, "std"])
            threshold_medium = mean_count + (2 * std_count)
            threshold_high = mean_count + (3 * std_count)

            if current_count <= threshold_medium:
                continue

            severity = "HIGH" if current_count > threshold_high else "MEDIUM"
            score = min(1.0, 0.6 + ((current_count - threshold_medium) / (max(1.0, threshold_medium) + 1)))

            matching_rows = data[
                (data["vendor_key"] == vendor) & (data["year_month"] == current_month)
            ]

            findings.append(
                self._build_group_finding(
                    rows=matching_rows,
                    finding_type="frequency_anomaly",
                    severity=severity,
                    score=score,
                    evidence={
                        "vendor": vendor,
                        "current_month": str(current_month),
                        "current_count": int(current_count),
                        "mean_count": round(mean_count, 2),
                        "std_count": round(std_count, 2),
                        "threshold_medium": round(threshold_medium, 2),
                        "threshold_high": round(threshold_high, 2),
                    },
                    reason="Vendor transaction frequency in current month exceeds historical trend.",
                )
            )

        return findings

    def detect_split_billing(self, df: pd.DataFrame, rules: Any) -> list[AnomalyFinding]:
        """Detect split billing attempts within rolling windows by vendor.
        
        Skips INTEREST and CHARGES transactions (legitimate bank fees and interest).
        """
        data = self._prepare_dataframe(df)

        # Exclude categories that are usually expected repetitive/system postings.
        if "transaction_category" in data.columns:
            data = data[~data["transaction_category"].isin(["EMI", "TRANSFER", "CHARGES"])].copy()
        
        findings: list[AnomalyFinding] = []

        if data.empty:
            return findings

        split_rules = self._parse_split_rules(rules)
        vendor_col = self._resolve_vendor_column(data)
        data["vendor_key"] = data[vendor_col].fillna("unknown").astype(str).str.strip().replace("", "unknown")

        for vendor, group in data.groupby("vendor_key", dropna=False):
            ordered = group.sort_values("parsed_date").copy()
            records = list(ordered.itertuples(index=False))

            for start_idx in range(len(records)):
                start_record = records[start_idx]
                window_end = start_record.parsed_date + pd.Timedelta(days=split_rules.within_days)
                window_rows = [
                    rec
                    for rec in records[start_idx:]
                    if rec.parsed_date <= window_end
                ]

                if len(window_rows) < split_rules.min_invoices_to_flag:
                    continue

                amounts = [float(rec.amount) for rec in window_rows]
                if not all(amount < split_rules.all_below_threshold for amount in amounts):
                    continue

                combined_amount = float(sum(amounts))
                rows_df = pd.DataFrame(window_rows)
                count = len(window_rows)
                days = split_rules.within_days
                total = combined_amount
                threshold = split_rules.all_below_threshold

                findings.append(
                    self._build_group_finding(
                        rows=rows_df,
                        finding_type="split_billing",
                        severity="HIGH",
                        score=0.95,
                        evidence={
                            "vendor": vendor,
                            "window_days": split_rules.within_days,
                            "transaction_count": len(window_rows),
                            "approval_threshold": split_rules.all_below_threshold,
                            "combined_amount": round(combined_amount, 2),
                        },
                        reason=(
                            f"Vendor '{vendor}' received {count} payments in {days} days "
                            f"totaling ₹{total:,.0f}, all below ₹{threshold:,.0f} approval threshold"
                        ),
                    )
                )
                break

        return findings

    def detect_multivariate_anomalies(self, df: pd.DataFrame) -> list[AnomalyFinding]:
        """Detect anomalies using IsolationForest on transaction behavior features."""
        data = self._prepare_dataframe(df)

        if "transaction_category" in data.columns:
            data = data[~data["transaction_category"].isin(["SALARY", "EMI", "REFUND"])].copy()

        findings: list[AnomalyFinding] = []

        if len(data) < 10:
            return findings

        vendor_col = self._resolve_vendor_column(data)
        vendor_frequency = data[vendor_col].fillna("unknown").astype(str).value_counts()

        features = pd.DataFrame(
            {
                "amount": data["amount"].astype(float),
                "hour": data["parsed_date"].dt.hour.astype(float),
                "day_of_week": data["parsed_date"].dt.weekday.astype(float),
                "vendor_frequency": data[vendor_col]
                .fillna("unknown")
                .astype(str)
                .map(vendor_frequency)
                .fillna(1)
                .astype(float),
            }
        )

        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(features)

        predictions = model.predict(features)
        raw_scores = model.decision_function(features)

        anomaly_rows = data[predictions == -1].copy()
        anomaly_scores = raw_scores[predictions == -1]

        for (_, row), raw_score in zip(anomaly_rows.iterrows(), anomaly_scores):
            magnitude = max(0.0, min(1.0, -float(raw_score)))
            severity = "HIGH" if magnitude > 0.25 else "MEDIUM"
            score = min(1.0, 0.55 + magnitude)

            findings.append(
                self._build_finding(
                    row=row,
                    finding_type="multivariate_anomaly",
                    severity=severity,
                    score=score,
                    evidence={
                        "anomaly_score": round(float(raw_score), 5),
                        "score_magnitude": round(magnitude, 5),
                        "amount": float(row["amount"]),
                        "hour": int(row["parsed_date"].hour),
                        "day_of_week": int(row["parsed_date"].weekday()),
                        "vendor_frequency": int(vendor_frequency.get(str(row.get(vendor_col, "unknown")), 1)),
                    },
                    reason="Transaction deviates from multivariate behavioral baseline.",
                )
            )

        return findings

    def detect_expense_policy_violations(self, expense_df: pd.DataFrame) -> list[AnomalyFinding]:
        """Detect policy violations in employee expense submissions."""
        findings: list[AnomalyFinding] = []
        if expense_df is None or expense_df.empty:
            return findings

        data = expense_df.copy()
        data["expense_date"] = pd.to_datetime(data.get("date"), dayfirst=True, errors="coerce")
        data["amount_value"] = pd.to_numeric(
            data.get("amount", pd.Series(index=data.index, dtype=object)).astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        submitted_col = self._find_existing_column(data, ["submitted_by", "employee_name", "employee", "user"])
        data["submitted_by_key"] = (
            data[submitted_col].fillna("unknown").astype(str).str.strip().replace("", "unknown")
            if submitted_col
            else "unknown"
        )

        submitted_date_col = self._find_existing_column(
            data,
            ["submitted_date", "submission_date", "submitted_at", "created_time", "created_date"],
        )
        if submitted_date_col:
            data["submitted_date"] = pd.to_datetime(data[submitted_date_col], dayfirst=True, errors="coerce")
        else:
            data["submitted_date"] = pd.NaT

        tx_id_col = self._find_existing_column(data, ["expense_id", "transaction_id", "id"])
        if tx_id_col:
            data["transaction_id"] = data[tx_id_col].fillna("").astype(str)
        else:
            data["transaction_id"] = data.index.astype(str)
        missing_tx = data["transaction_id"].str.strip() == ""
        data.loc[missing_tx, "transaction_id"] = data.index[missing_tx].astype(str)

        if "document_name" not in data.columns:
            data["document_name"] = "unknown"

        weekend_mask = (data["expense_date"].dt.weekday.isin([5, 6])) & (data["amount_value"] > 5000)
        for _, row in data[weekend_mask].iterrows():
            findings.append(
                self._build_finding(
                    row=row,
                    finding_type="expense_policy_weekend_high_amount",
                    severity="MEDIUM",
                    score=0.75,
                    evidence={
                        "submitted_by": str(row.get("submitted_by_key", "unknown")),
                        "expense_date": str(row.get("expense_date")),
                        "amount": float(row.get("amount_value", 0.0) or 0.0),
                        "threshold": 5000,
                    },
                    reason="Weekend expense amount exceeds policy threshold of INR 5000.",
                )
            )

        valid_dates = data.dropna(subset=["expense_date"]).copy()
        valid_dates["expense_day"] = valid_dates["expense_date"].dt.normalize()
        for employee, group in valid_dates.groupby("submitted_by_key", dropna=False):
            unique_days = sorted(group["expense_day"].dropna().unique())
            if len(unique_days) < 3:
                continue

            streak: list[pd.Timestamp] = [unique_days[0]]
            for day in unique_days[1:]:
                prev_day = streak[-1]
                if (day - prev_day).days == 1:
                    streak.append(day)
                else:
                    if len(streak) >= 3:
                        streak_rows = group[group["expense_day"].isin(streak)]
                        findings.append(
                            self._build_group_finding(
                                rows=streak_rows,
                                finding_type="expense_policy_consecutive_days",
                                severity="MEDIUM",
                                score=0.72,
                                evidence={
                                    "submitted_by": employee,
                                    "consecutive_days": len(streak),
                                    "start_date": str(streak[0]),
                                    "end_date": str(streak[-1]),
                                },
                                reason="Same employee submitted expenses for 3 or more consecutive days.",
                            )
                        )
                        break
                    streak = [day]

            if len(streak) >= 3:
                streak_rows = group[group["expense_day"].isin(streak)]
                findings.append(
                    self._build_group_finding(
                        rows=streak_rows,
                        finding_type="expense_policy_consecutive_days",
                        severity="MEDIUM",
                        score=0.72,
                        evidence={
                            "submitted_by": employee,
                            "consecutive_days": len(streak),
                            "start_date": str(streak[0]),
                            "end_date": str(streak[-1]),
                        },
                        reason="Same employee submitted expenses for 3 or more consecutive days.",
                    )
                )

        late_mask = (
            data["submitted_date"].notna()
            & data["expense_date"].notna()
            & ((data["submitted_date"] - data["expense_date"]).dt.days > 30)
        )
        for _, row in data[late_mask].iterrows():
            days_late = int((row["submitted_date"] - row["expense_date"]).days)
            findings.append(
                self._build_finding(
                    row=row,
                    finding_type="expense_policy_late_submission",
                    severity="MEDIUM",
                    score=0.7,
                    evidence={
                        "submitted_by": str(row.get("submitted_by_key", "unknown")),
                        "expense_date": str(row.get("expense_date")),
                        "submitted_date": str(row.get("submitted_date")),
                        "days_late": days_late,
                    },
                    reason="Expense was submitted more than 30 days after expense date.",
                )
            )

        same_day_groups = valid_dates.groupby(["submitted_by_key", "expense_day"], dropna=False)
        for (employee, expense_day), group in same_day_groups:
            if len(group) < 2:
                continue
            findings.append(
                self._build_group_finding(
                    rows=group,
                    finding_type="expense_policy_possible_split",
                    severity="HIGH" if len(group) >= 3 else "MEDIUM",
                    score=0.78 if len(group) >= 3 else 0.68,
                    evidence={
                        "submitted_by": employee,
                        "expense_day": str(expense_day),
                        "expense_count": len(group),
                        "total_amount": float(group["amount_value"].fillna(0).sum()),
                    },
                    reason="Multiple expenses submitted by same employee on same day, possible splitting.",
                )
            )

        round_mask = (data["amount_value"] >= 10000) & (data["amount_value"] % 5000 == 0)
        for _, row in data[round_mask].iterrows():
            findings.append(
                self._build_finding(
                    row=row,
                    finding_type="expense_policy_round_large_amount",
                    severity="MEDIUM",
                    score=0.66,
                    evidence={
                        "submitted_by": str(row.get("submitted_by_key", "unknown")),
                        "amount": float(row.get("amount_value", 0.0) or 0.0),
                        "round_rule": "amount >= 10000 and multiple of 5000",
                    },
                    reason="Large expense has a round amount pattern that may indicate manual splitting or estimation.",
                )
            )

        return findings

    def detect_gst_anomalies(self, gst_df: pd.DataFrame) -> list[AnomalyFinding]:
        """Detect GST data inconsistencies and suspicious filing patterns."""
        findings: list[AnomalyFinding] = []
        if gst_df is None or gst_df.empty:
            return findings

        data = gst_df.copy()
        gst_columns = ["gstin", "buyer_gstin", "invoice_number", "taxable_value"]
        if not any(col in data.columns for col in gst_columns):
            return findings

        GST_PATTERN = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"

        gstin_col = "gstin" if "gstin" in data.columns else "buyer_gstin"
        if gstin_col not in data.columns:
            return findings

        for _, row in data.iterrows():
            gstin = str(row.get(gstin_col, ""))
            if gstin and gstin != "nan":
                if not re.match(GST_PATTERN, gstin):
                    findings.append(
                        self._build_finding(
                            row=row,
                            finding_type="invalid_gstin",
                            severity="HIGH",
                            score=0.95,
                            evidence={"gstin": gstin, "invoice": str(row.get("invoice_number", ""))},
                            reason=f"Invalid GSTIN format '{gstin}' on invoice {row.get('invoice_number', '')} — does not match GST number pattern",
                        )
                    )

        if "invoice_number" in data.columns:
            invoice_counts = data["invoice_number"].value_counts()
            duplicates = invoice_counts[invoice_counts > 1]
            for inv_no, count in duplicates.items():
                dup_rows = data[data["invoice_number"] == inv_no]
                total = float(dup_rows["total"].sum() if "total" in data.columns else 0)
                findings.append(
                    self._build_finding(
                        row=dup_rows.iloc[0],
                        finding_type="duplicate_invoice",
                        severity="HIGH",
                        score=0.98,
                        evidence={"invoice_number": str(inv_no), "count": int(count), "total": total},
                        reason=f"Invoice {inv_no} appears {count} times — possible duplicate billing of Rs.{total/count:,.0f}",
                    )
                )

        for _, row in data.iterrows():
            taxable = float(row.get("taxable_value", 0) or 0)
            igst = float(row.get("igst", 0) or 0)
            cgst = float(row.get("cgst", 0) or 0)
            sgst = float(row.get("sgst", 0) or 0)
            total = float(row.get("total", 0) or 0)

            if taxable > 0 and total > 0:
                expected_total = taxable + igst + cgst + sgst
                if abs(expected_total - total) > total * 0.01:
                    findings.append(
                        self._build_finding(
                            row=row,
                            finding_type="gst_math_error",
                            severity="MEDIUM",
                            score=0.85,
                            evidence={
                                "taxable": taxable,
                                "tax": igst + cgst + sgst,
                                "expected_total": expected_total,
                                "actual_total": total,
                            },
                            reason=f"GST calculation error on invoice {row.get('invoice_number', '')} — expected Rs.{expected_total:,.0f} but total is Rs.{total:,.0f}",
                        )
                    )

        return findings

    def detect_invoice_anomalies(self, invoice_df: pd.DataFrame, rules: Any) -> list[AnomalyFinding]:
        """Detect invoice-specific anomalies such as duplicates, tax mismatches, and missing vendors."""
        findings: list[AnomalyFinding] = []
        if invoice_df is None or invoice_df.empty:
            return findings

        data = invoice_df.copy()
        required_columns = {"invoice_number", "date", "vendor", "amount", "tax", "total"}
        if not any(column in data.columns for column in required_columns):
            return findings

        amount_threshold = self._get_invoice_amount_threshold(rules)
        data["invoice_number_key"] = data.get("invoice_number", pd.Series(index=data.index, dtype=object)).apply(
            self._normalize_text_key
        )
        data["vendor_key"] = data.get("vendor", pd.Series(index=data.index, dtype=object)).apply(
            self._normalize_text_key
        )
        data["date_parsed"] = pd.to_datetime(data.get("date"), dayfirst=True, errors="coerce")

        if "invoice_number_key" in data.columns:
            duplicates = data[data["invoice_number_key"] != ""].groupby("invoice_number_key")
            for invoice_number, group in duplicates:
                count = len(group)
                if count <= 1:
                    continue

                findings.append(
                    self._build_group_finding(
                        rows=group,
                        finding_type="duplicate_invoice",
                        severity="HIGH",
                        score=0.98,
                        evidence={
                            "invoice_number": invoice_number,
                            "count": int(count),
                            "source_files": sorted({str(value) for value in group.get("source_file", pd.Series(dtype=object)).dropna().tolist()}),
                        },
                        reason=f"Invoice {invoice_number} appears {count} times — possible duplicate billing",
                    )
                )

        for _, row in data.iterrows():
            invoice_number = self._normalize_text_key(row.get("invoice_number"))
            amount = float(row.get("amount") or 0)
            tax = float(row.get("tax") or 0)
            total = float(row.get("total") or 0)
            vendor = self._normalize_text_key(row.get("vendor"))
            parsed_date = row.get("date_parsed")

            if not vendor:
                findings.append(
                    self._build_finding(
                        row=row,
                        finding_type="missing_vendor",
                        severity="HIGH",
                        score=0.97,
                        evidence={"invoice_number": invoice_number, "vendor": None if pd.isna(row.get("vendor")) else row.get("vendor")},
                        reason=f"Invoice {invoice_number or 'unknown'} has no vendor name — cannot verify source",
                    )
                )

            if amount > amount_threshold:
                findings.append(
                    self._build_finding(
                        row=row,
                        finding_type="high_invoice_amount",
                        severity="HIGH",
                        score=0.9,
                        evidence={"invoice_number": invoice_number, "amount": amount, "threshold": float(amount_threshold)},
                        reason=f"Invoice {invoice_number or 'unknown'} amount Rs.{amount:,.0f} exceeds approval threshold",
                    )
                )

            if amount > 0 and self._is_round_number(amount, 10000):
                findings.append(
                    self._build_finding(
                        row=row,
                        finding_type="round_invoice_amount",
                        severity="MEDIUM",
                        score=0.72,
                        evidence={"invoice_number": invoice_number, "amount": amount},
                        reason=f"Invoice {invoice_number or 'unknown'} has suspiciously round amount Rs.{amount:,.0f}",
                    )
                )

            if amount > 0 or tax > 0 or total > 0:
                expected_total = amount + tax
                if abs(expected_total - total) > 1:
                    findings.append(
                        self._build_finding(
                            row=row,
                            finding_type="invoice_tax_mismatch",
                            severity="MEDIUM",
                            score=0.84,
                            evidence={
                                "invoice_number": invoice_number,
                                "amount": amount,
                                "tax": tax,
                                "expected_total": expected_total,
                                "actual_total": total,
                            },
                            reason=f"Invoice {invoice_number or 'unknown'} tax calculation mismatch — expected {expected_total:,.0f} got {total:,.0f}",
                        )
                    )

            if pd.notna(parsed_date):
                day_name = parsed_date.strftime("%A")
                if parsed_date.weekday() >= 5 and amount > 10000:
                    findings.append(
                        self._build_finding(
                            row=row,
                            finding_type="weekend_invoice",
                            severity="MEDIUM",
                            score=0.78,
                            evidence={"invoice_number": invoice_number, "day": day_name, "amount": amount},
                            reason=f"Invoice {invoice_number or 'unknown'} dated on {day_name} — weekend invoice above threshold",
                        )
                    )

        return findings

    def detect_suspicious_payment_patterns(self, transactions_df: pd.DataFrame) -> list[AnomalyFinding]:
        """Detect suspicious payment behavior around thresholds, account usage, and reversals."""
        data = self._prepare_dataframe(transactions_df)
        findings: list[AnomalyFinding] = []

        if data.empty:
            return findings

        beneficiary_col = self._find_existing_column(
            data,
            ["beneficiary_account", "beneficiary_acct", "to_account", "account_number", "beneficiary"],
        )
        source_col = self._find_existing_column(
            data,
            ["company_account", "from_account", "debit_account", "account_from", "source_account"],
        )
        if not source_col:
            source_col = self._find_existing_column(data, ["account_number", "account_no"])

        if beneficiary_col:
            data["beneficiary_key"] = data[beneficiary_col].fillna("").astype(str).str.strip()
        else:
            data["beneficiary_key"] = ""
        if source_col:
            data["source_account_key"] = data[source_col].fillna("").astype(str).str.strip()
        else:
            data["source_account_key"] = ""

        if beneficiary_col and source_col:
            for beneficiary, group in data[data["beneficiary_key"] != ""].groupby("beneficiary_key"):
                source_count = group["source_account_key"].replace("", pd.NA).dropna().nunique()
                if source_count >= 2:
                    findings.append(
                        self._build_group_finding(
                            rows=group,
                            finding_type="payment_multi_source_to_same_beneficiary",
                            severity="HIGH",
                            score=0.82,
                            evidence={
                                "beneficiary_account": beneficiary,
                                "source_accounts": sorted(group["source_account_key"].replace("", pd.NA).dropna().unique().tolist()),
                                "source_count": int(source_count),
                            },
                            reason="Same beneficiary account receives payments from multiple company accounts.",
                        )
                    )

        if beneficiary_col:
            personal_mask = data["beneficiary_key"].str.fullmatch(r"\d{10,12}", na=False)
            for _, row in data[personal_mask].iterrows():
                findings.append(
                    self._build_finding(
                        row=row,
                        finding_type="payment_possible_personal_account",
                        severity="MEDIUM",
                        score=0.7,
                        evidence={
                            "beneficiary_account": str(row.get("beneficiary_key", "")),
                            "amount": float(row.get("amount", 0.0) or 0.0),
                        },
                        reason="Beneficiary account number pattern resembles personal account format.",
                    )
                )

        invoice_date_col = self._find_existing_column(data, ["invoice_date", "bill_date"])
        if "invoice_date" not in data.columns and "bill_date" not in data.columns:
            return findings
        if not invoice_date_col:
            return findings

        data["invoice_date_parsed"] = pd.to_datetime(data[invoice_date_col], dayfirst=True, errors="coerce")
        large_payment_threshold = max(100000.0, float(data["amount"].quantile(0.90)))
        same_day_large = (
            data["invoice_date_parsed"].notna()
            & (data["invoice_date_parsed"].dt.normalize() == data["parsed_date"].dt.normalize())
            & (data["amount"] >= large_payment_threshold)
        )
        for _, row in data[same_day_large].iterrows():
            findings.append(
                self._build_finding(
                    row=row,
                    finding_type="payment_unusually_fast_after_invoice",
                    severity="MEDIUM",
                    score=0.76,
                    evidence={
                        "invoice_date": str(row.get("invoice_date_parsed")),
                        "payment_date": str(row.get("parsed_date")),
                        "amount": float(row.get("amount", 0.0) or 0.0),
                        "large_payment_threshold": large_payment_threshold,
                    },
                    reason="Large payment was made on the same day as invoice date.",
                )
            )

        signed_series = self._signed_amount_series(data)
        data["signed_amount"] = signed_series
        ordered = data.sort_values("parsed_date")

        threshold_value = 100000.0
        lower_bound = threshold_value * 0.95
        upper_bound = threshold_value * 0.99
        just_below_mask = data["amount"].between(lower_bound, upper_bound, inclusive="both")
        just_below_df = data[just_below_mask].copy()
        if not just_below_df.empty:
            if beneficiary_col:
                repeated_counts = just_below_df.groupby("beneficiary_key")["beneficiary_key"].transform("size")
            else:
                repeated_counts = pd.Series([1] * len(just_below_df), index=just_below_df.index)

            just_below_df["repeat_count"] = repeated_counts
            for _, row in just_below_df.iterrows():
                repeat_count = int(row.get("repeat_count", 1) or 1)
                severity = "HIGH" if repeat_count >= 2 else "LOW"
                findings.append(
                    self._build_finding(
                        row=row,
                        finding_type="payment_just_below_threshold",
                        severity=severity,
                        score=0.85 if severity == "HIGH" else 0.45,
                        evidence={
                            "amount": float(row.get("amount", 0.0) or 0.0),
                            "threshold": threshold_value,
                            "beneficiary_account": str(row.get("beneficiary_key", "")),
                            "repeat_count": repeat_count,
                        },
                        reason="Payment amount is just below INR 100000 approval threshold.",
                    )
                )

        for beneficiary, group in ordered[ordered["beneficiary_key"] != ""].groupby("beneficiary_key"):
            credits = group[group["signed_amount"] > 0]
            debits = group[group["signed_amount"] < 0]
            if credits.empty or debits.empty:
                continue

            for _, credit_row in credits.iterrows():
                matched_debits = debits[
                    (debits["parsed_date"] > credit_row["parsed_date"])
                    & ((debits["parsed_date"] - credit_row["parsed_date"]).dt.days <= 7)
                    & ((debits["signed_amount"].abs() - abs(float(credit_row["signed_amount"]))).abs() <= 1.0)
                ]
                if matched_debits.empty:
                    continue

                pair_rows = pd.concat([pd.DataFrame([credit_row]), matched_debits.head(1)], ignore_index=False)
                findings.append(
                    self._build_group_finding(
                        rows=pair_rows,
                        finding_type="payment_reversed_and_reissued",
                        severity="HIGH",
                        score=0.9,
                        evidence={
                            "beneficiary_account": beneficiary,
                            "credit_amount": float(credit_row["signed_amount"]),
                            "matched_debit_amount": float(matched_debits.iloc[0]["signed_amount"]),
                            "credit_date": str(credit_row["parsed_date"]),
                            "debit_date": str(matched_debits.iloc[0]["parsed_date"]),
                        },
                        reason="Payment appears reversed (credit) and re-issued (debit) with same amount.",
                    )
                )
                break

        # --- Tag-aware bank statement checks ---
        try:
            # Determine salary baseline if available: prefer explicit column, else infer from SALARY category credits
            salary = None
            if "salary_amount" in data.columns:
                try:
                    salary = float(data["salary_amount"].dropna().astype(float).max())
                except Exception:
                    salary = None

            if salary is None and "transaction_category" in data.columns:
                sal_rows = data[data["transaction_category"] == "SALARY"]
                if not sal_rows.empty:
                    try:
                        salary = float(sal_rows["amount"].dropna().astype(float).max())
                    except Exception:
                        salary = None

            # 1) Large cash withdrawals
            if "tags" in data.columns:
                for _, row in data.iterrows():
                    tags = row.get("tags") or {}
                    try:
                        is_cash = bool(tags.get("is_cash")) if isinstance(tags, dict) else False
                    except Exception:
                        is_cash = False
                    debit_amt = float(row.get("debit") or 0.0)
                    if is_cash and debit_amt > 0 and salary is not None and debit_amt > (salary * 0.5):
                        findings.append(
                            self._build_finding(
                                row=row,
                                finding_type="large_cash_withdrawal",
                                severity="MEDIUM",
                                score=0.7,
                                evidence={
                                    "amount": debit_amt,
                                    "salary_threshold": salary,
                                },
                                reason=(f"Large cash withdrawal of ₹{debit_amt:,.0f} detected"),
                            )
                        )

            # 2) Round number suspicious transactions
            for _, row in data.iterrows():
                amount = float(row.get("amount") or 0.0)
                if amount <= 50000:
                    continue
                if not self._is_round_number(amount, 10000):
                    continue
                tags = row.get("tags") or {}
                try:
                    pmode = tags.get("payment_mode") if isinstance(tags, dict) else None
                except Exception:
                    pmode = None
                if pmode and str(pmode).upper() in {"SI", "EMI"}:
                    continue
                findings.append(
                    self._build_finding(
                        row=row,
                        finding_type="round_number_transaction",
                        severity="LOW",
                        score=0.45,
                        evidence={"amount": amount, "payment_mode": pmode},
                        reason=(f"Suspiciously round transaction amount ₹{amount:,.0f} via {pmode or 'unknown'}"),
                    )
                )

            # 3) High frequency merchant payments (more than 5 times in 30 days)
            if "tags" in data.columns:
                merchants = []
                merchant_groups = {}
                for _, row in data.iterrows():
                    tags = row.get("tags") or {}
                    try:
                        is_merchant = bool(tags.get("is_merchant")) if isinstance(tags, dict) else False
                        receiver = tags.get("receiver") if isinstance(tags, dict) else None
                    except Exception:
                        is_merchant = False
                        receiver = None
                    if is_merchant and receiver:
                        merchant_groups.setdefault(receiver, []).append(row["parsed_date"])

                for receiver, dates in merchant_groups.items():
                    dates_sorted = sorted([d for d in dates if pd.notna(d)])
                    # sliding window count
                    max_count = 0
                    for i, dt in enumerate(dates_sorted):
                        window_end = dt + pd.Timedelta(days=30)
                        count = sum(1 for d in dates_sorted if d >= dt and d <= window_end)
                        if count > max_count:
                            max_count = count
                    if max_count > 5:
                        # build a pseudo-row for evidence
                        pseudo = pd.Series({"transaction_id": f"merchant-{receiver}", "description": receiver})
                        findings.append(
                            self._build_finding(
                                row=pseudo,
                                finding_type="frequent_merchant_payments",
                                severity="LOW",
                                score=0.5,
                                evidence={"receiver": receiver, "count_30d": int(max_count)},
                                reason=(f"Frequent payments to {receiver}: {max_count} times in 30 days"),
                            )
                        )

            # 4) EMI count anomaly
            emi_rows = []
            if "tags" in data.columns:
                for _, row in data.iterrows():
                    tags = row.get("tags") or {}
                    try:
                        if isinstance(tags, dict) and bool(tags.get("is_emi")):
                            emi_rows.append(float(row.get("amount") or 0.0))
                    except Exception:
                        continue
            emi_count = len(emi_rows)
            avg_emi = float(pd.Series(emi_rows).mean()) if emi_count > 0 else 0.0
            total_emi = emi_count * avg_emi
            if emi_count > 0 and salary is not None and (total_emi > (salary * 0.5)):
                pseudo = pd.Series({"transaction_id": "emi-summary", "description": "EMI summary"})
                findings.append(
                    self._build_finding(
                        row=pseudo,
                        finding_type="emi_obligation_exceeds_salary",
                        severity="HIGH",
                        score=0.95,
                        evidence={"emi_count": emi_count, "avg_emi": avg_emi, "total_emi": total_emi, "salary": salary},
                        reason=(f"Total EMI obligations ₹{total_emi:,.0f} exceed 50% of salary (₹{salary:,.0f})"),
                    )
                )

            # 5) Refund without matching debit in prior 30 days
            if "tags" in data.columns:
                refunds = []
                for idx, row in data.iterrows():
                    tags = row.get("tags") or {}
                    try:
                        if isinstance(tags, dict) and bool(tags.get("is_refund")):
                            refunds.append((idx, row))
                    except Exception:
                        continue

                for idx, refund_row in refunds:
                    r_tags = refund_row.get("tags") or {}
                    try:
                        receiver = r_tags.get("receiver") if isinstance(r_tags, dict) else None
                    except Exception:
                        receiver = None
                    refund_date = refund_row.get("parsed_date")
                    if not receiver or pd.isna(refund_date):
                        continue
                    window_start = refund_date - pd.Timedelta(days=30)
                    # look for prior debit to same receiver
                    prior = data[
                        (data["parsed_date"] >= window_start)
                        & (data["parsed_date"] < refund_date)
                        & (data["debit"] > 0)
                    ]
                    # match by tags.receiver where available
                    matched = False
                    for _, cand in prior.iterrows():
                        ctags = cand.get("tags") or {}
                        try:
                            creceiver = ctags.get("receiver") if isinstance(ctags, dict) else None
                        except Exception:
                            creceiver = None
                        if creceiver and receiver and creceiver.strip().upper() == receiver.strip().upper():
                            matched = True
                            break
                    if not matched:
                        findings.append(
                            self._build_finding(
                                row=refund_row,
                                finding_type="refund_without_prior_debit",
                                severity="LOW",
                                score=0.4,
                                evidence={"receiver": receiver, "refund_date": str(refund_date)},
                                reason=(f"Refund from {receiver} with no matching prior payment found"),
                            )
                        )
        except Exception:
            # Do not let tagging checks break overall detection
            pass

        return findings

            # Note: the following tag-aware bank statement checks are appended below.

    def detect_vendor_risk(self, transactions_df: pd.DataFrame, invoices: list) -> list[AnomalyFinding]:
        """Detect vendor-level risk signals from transactions and invoice metadata.
        
        Skips TRANSFER transactions (internal account transfers, not vendor payments).
        """
        # Skip entirely if no GST columns present (bank statements)
        gst_columns = ['gstin', 'gst_number', 'vendor_gst', 'GSTIN']
        if not any(col in transactions_df.columns for col in gst_columns):
            return []
        
        # Skip TRANSFER transactions for vendor risk analysis
        analysis_df = transactions_df.copy()
        if "transaction_category" in analysis_df.columns:
            analysis_df = analysis_df[analysis_df["transaction_category"] != "TRANSFER"].copy()
        
        findings: list[AnomalyFinding] = []

        data = self._prepare_dataframe(analysis_df) if analysis_df is not None and not analysis_df.empty else pd.DataFrame()
        invoice_records = [self._invoice_to_dict(inv) for inv in (invoices or [])]

        if not data.empty:
            vendor_col = self._resolve_vendor_column(data)
            gst_col = self._find_existing_column(data, ["vendor_gst", "gstin", "supplier_gstin", "buyer_gstin"])
            account_col = self._find_existing_column(data, ["beneficiary_account", "account_number", "account_no", "bank_account"])

            data["vendor_key"] = data[vendor_col].fillna("unknown").astype(str).apply(self._extract_vendor_token)
            data["gst_key"] = data[gst_col].fillna("").astype(str).str.strip().str.upper() if gst_col else ""

            no_gst_high = data[(data["amount"] > 20000) & (data["gst_key"].astype(str).str.strip() == "")]
            for _, row in no_gst_high.iterrows():
                amount = float(row.get("amount", 0.0) or 0.0)
                vendor = str(row.get("vendor_key", "unknown"))
                findings.append(
                    self._build_finding(
                        row=row,
                        finding_type="vendor_no_gst_high_payment",
                        severity="HIGH",
                        score=0.88,
                        evidence={
                            "vendor": vendor,
                            "amount": amount,
                            "threshold": 20000,
                        },
                        reason=(
                            f"Payment of ₹{amount:,.0f} to vendor '{vendor}' "
                            f"with no GST number on record"
                        ),
                    )
                )

            if account_col:
                data["account_key"] = data[account_col].fillna("").astype(str).str.strip()
                grouped = data[data["account_key"] != ""].groupby("account_key")
                for account, group in grouped:
                    vendors = sorted(set(group["vendor_key"].tolist()))
                    if len(vendors) >= 2:
                        findings.append(
                            self._build_group_finding(
                                rows=group,
                                finding_type="vendor_shared_bank_account",
                                severity="HIGH",
                                score=0.9,
                                evidence={"account_number": account, "vendors": vendors},
                                reason=(
                                    f"Multiple vendors ({', '.join(vendors[:3])}) use the same bank account "
                                    f"'{account}' for payments—sign of vendor consolidation or fraud."
                                ),
                            )
                        )

        employee_names = self._collect_employee_names(data)

        for inv in invoice_records:
            vendor_name = str(inv.get("vendor_name") or "").strip()
            vendor_gst = str(inv.get("vendor_gst") or "").strip()
            vendor_address = str(inv.get("vendor_address") or "").strip()
            total_amount = float(self._safe_float(inv.get("total_amount")) or 0.0)
            invoice_number = str(inv.get("invoice_number") or "").strip()
            invoice_date = str(inv.get("invoice_date") or "")

            pseudo_row = pd.Series(
                {
                    "transaction_id": invoice_number or str(uuid.uuid4()),
                    "document_name": "invoice",
                    "vendor_key": vendor_name or "unknown",
                }
            )

            if not vendor_address:
                findings.append(
                    self._build_finding(
                        row=pseudo_row,
                        finding_type="vendor_missing_address",
                        severity="MEDIUM",
                        score=0.7,
                        evidence={"vendor": vendor_name, "invoice_number": invoice_number, "invoice_date": invoice_date},
                        reason=(
                            f"Invoice {invoice_number} from vendor '{vendor_name}' is missing address "
                            f"information—cannot verify vendor legitimacy."
                        ),
                    )
                )

            if total_amount > 20000 and not vendor_gst:
                findings.append(
                    self._build_finding(
                        row=pseudo_row,
                        finding_type="vendor_no_gst_high_invoice",
                        severity="HIGH",
                        score=0.86,
                        evidence={"vendor": vendor_name, "invoice_number": invoice_number, "amount": total_amount},
                        reason=(
                            f"Invoice {invoice_number} totaling ₹{total_amount:,.0f} from vendor '{vendor_name}' "
                            f"has no GST number—high-value transaction requires GST registration verification."
                        ),
                    )
                )

            if vendor_name and employee_names:
                for employee in employee_names:
                    similarity = SequenceMatcher(None, vendor_name.lower(), employee.lower()).ratio()
                    if similarity >= 0.88 and vendor_name.lower() != employee.lower():
                        findings.append(
                            self._build_finding(
                                row=pseudo_row,
                                finding_type="vendor_name_similar_to_employee",
                                severity="HIGH",
                                score=min(1.0, 0.75 + (similarity - 0.88)),
                                evidence={
                                    "vendor_name": vendor_name,
                                    "employee_name": employee,
                                    "similarity": round(similarity, 4),
                                },
                                reason=(
                                    f"Vendor '{vendor_name}' name matches employee '{employee}' "
                                    f"({round(similarity * 100)}% similar)—possible fraudulent relationship."
                                ),
                            )
                        )
                        break

        return findings

    def infer_doc_type(self, df: pd.DataFrame) -> str:
        """Infer source document type from available columns."""
        if "gstin" in df.columns:
            return "gst"
        if "submitted_by" in df.columns:
            return "expense"
        if "debit" in df.columns and "credit" in df.columns:
            return "bank_statement"
        return "unknown"

    def detect_all(
        self,
        df: pd.DataFrame,
        rules: Any,
        expense_df: pd.DataFrame | None = None,
        gst_df: pd.DataFrame | None = None,
        transactions_df: pd.DataFrame | None = None,
        invoices: list | None = None,
        exclude_amounts: list[float] | None = None,
    ) -> list[AnomalyFinding]:
        """Run all detectors, merge duplicates, and return HIGH-first ordered findings.
        
        Args:
            df: Transaction dataframe to analyze
            rules: Policy rules for evaluation
            expense_df: Optional expense sheet dataframe
            gst_df: Optional GST document dataframe
            transactions_df: Optional additional transactions dataframe
            invoices: Optional list of invoices
            exclude_amounts: Optional list of known regular payment amounts to exclude
        """
        # Tag categories before running any detectors.
        analysis_df = self.tag_transaction_categories(df)
        
        # Exclude known regular payments before running detectors
        if exclude_amounts:
            for amount in exclude_amounts:
                if amount > 0:
                    tolerance = amount * 0.02
                    mask = analysis_df["debit"].between(amount - tolerance, amount + tolerance)
                    excluded = int(mask.sum())
                    if excluded > 0:
                        analysis_df = analysis_df[~mask]
                        print(f"Excluded {excluded} rows matching ₹{amount:,.0f}")
        
        findings: list[AnomalyFinding] = []
        findings.extend(self.detect_amount_outliers(analysis_df))
        findings.extend(self.detect_timing_anomalies(analysis_df))
        findings.extend(self.detect_frequency_anomalies(analysis_df))
        findings.extend(self.detect_split_billing(analysis_df, rules))
        findings.extend(self.detect_multivariate_anomalies(analysis_df))

        invoice_source: pd.DataFrame | None = None
        if invoices:
            invoice_source = pd.DataFrame(invoices)
        elif any(column in analysis_df.columns for column in ["invoice_number", "vendor", "tax", "total", "source_file"]):
            invoice_source = analysis_df

        if invoice_source is not None:
            findings.extend(self.detect_invoice_anomalies(invoice_source, rules))

        expense_source = expense_df if expense_df is not None else pd.DataFrame()
        findings.extend(self.detect_expense_policy_violations(expense_source))

        gst_source = gst_df if gst_df is not None else pd.DataFrame()
        if any(col in analysis_df.columns for col in ["gstin", "buyer_gstin", "invoice_number", "taxable_value"]):
            findings.extend(self.detect_gst_anomalies(analysis_df))
        findings.extend(self.detect_gst_anomalies(gst_source))

        transaction_source = transactions_df if transactions_df is not None else analysis_df
        findings.extend(self.detect_suspicious_payment_patterns(transaction_source))

        vendor_source = transaction_source if "date" in transaction_source.columns else pd.DataFrame()
        findings.extend(self.detect_vendor_risk(vendor_source, invoices or []))

        merged = self._merge_findings(findings)
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        merged.sort(key=lambda f: (severity_order.get(f.severity, 3), -f.score))
        return merged

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if "debit" not in data.columns:
            if "taxable_value" in data.columns:
                data["debit"] = pd.to_numeric(data["taxable_value"], errors="coerce").fillna(0)
            elif "amount" in data.columns:
                data["debit"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0)
            else:
                data["debit"] = 0.0
        else:
            data["debit"] = pd.to_numeric(data["debit"], errors="coerce").fillna(0)

        if "credit" not in data.columns:
            data["credit"] = 0.0
        else:
            data["credit"] = pd.to_numeric(data["credit"], errors="coerce").fillna(0)

        if "amount" not in data.columns:
            data["amount"] = data["debit"].where(data["debit"] > 0, data["credit"])
        else:
            amount_text = data["amount"].astype(str).str.replace(",", "", regex=False).str.strip()
            data["amount"] = pd.to_numeric(amount_text, errors="coerce")

        if "date" not in data.columns:
            if "invoice_date" in data.columns:
                data["date"] = data["invoice_date"]
            else:
                data["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        data["parsed_date"] = self._parse_mixed_sbi_dates(data["date"])
        data = data.dropna(subset=["parsed_date", "amount"]).copy()

        if "description" not in data.columns:
            if "invoice_number" in data.columns:
                data["description"] = data["invoice_number"]
            else:
                data["description"] = "Unknown"

        if "transaction_id" not in data.columns:
            data["transaction_id"] = data.index.astype(str)
        else:
            data["transaction_id"] = data["transaction_id"].fillna("").astype(str)
            missing = data["transaction_id"].str.strip() == ""
            data.loc[missing, "transaction_id"] = data.index[missing].astype(str)

        if "document_name" not in data.columns:
            if "source_file" in data.columns:
                data["document_name"] = data["source_file"].fillna("unknown").astype(str)
            else:
                data["document_name"] = "unknown"

        return data

    def _parse_mixed_sbi_dates(self, values: pd.Series) -> pd.Series:
        """Parse SBI OCR date strings across the observed statement formats."""
        raw_values = values.astype(str).str.strip()
        raw_values = raw_values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

        parsed = pd.Series(pd.NaT, index=raw_values.index, dtype="datetime64[ns]")

        for date_format in ("%d-%m-%Y", "%d %b %Y", "%d-%m-%y"):
            candidate = pd.to_datetime(raw_values, format=date_format, errors="coerce")
            parsed = parsed.fillna(candidate)

        remaining_mask = parsed.isna() & raw_values.notna()
        if remaining_mask.any():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*infer.*format.*",
                    category=UserWarning,
                )
                inferred = pd.to_datetime(raw_values[remaining_mask], dayfirst=True, errors="coerce")
            parsed.loc[remaining_mask] = inferred

        return parsed

    def _get_invoice_amount_threshold(self, rules: Any) -> float:
        default_threshold = 100000.0
        if isinstance(rules, dict):
            amount_thresholds = rules.get("amount_thresholds", {})
            return float(amount_thresholds.get("vendor_payment_approval_above", default_threshold))

        amount_thresholds = getattr(rules, "amount_thresholds", None)
        if amount_thresholds is None:
            return default_threshold
        return float(getattr(amount_thresholds, "vendor_payment_approval_above", default_threshold))

    def _normalize_text_key(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    def _resolve_document_name(self, row: pd.Series) -> str:
        document_name = self._normalize_text_key(row.get("document_name"))
        if document_name:
            return document_name

        source_file = self._normalize_text_key(row.get("source_file"))
        if source_file:
            return source_file

        return "unknown"

    def _is_round_number(self, amount: float, divisor: int) -> bool:
        return divisor > 0 and abs(amount - round(amount / divisor) * divisor) < 0.01

    def _resolve_vendor_column(self, data: pd.DataFrame) -> str:
        for candidate in ["vendor", "vendor_name", "party_name", "description"]:
            if candidate in data.columns:
                return candidate
        data["party_name"] = "unknown"
        return "party_name"

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

    def _parse_split_rules(self, rules: Any) -> _SplitBillingRules:
        # Supports dict-based config and dataclass/object-based config.
        min_invoices = 3
        within_days = 7
        threshold = 100000.0

        if isinstance(rules, dict):
            split = rules.get("split_billing", {})
            amounts = rules.get("amount_thresholds", {})
            min_invoices = int(split.get("min_invoices_to_flag", min_invoices))
            within_days = int(split.get("within_days", within_days))
            threshold = float(
                split.get(
                    "all_below_threshold",
                    amounts.get("vendor_payment_approval_above", threshold),
                )
            )
        else:
            split_obj = getattr(rules, "split_billing", None)
            amount_obj = getattr(rules, "amount_thresholds", None)
            if split_obj is not None:
                min_invoices = int(getattr(split_obj, "min_invoices_to_flag", min_invoices))
                within_days = int(getattr(split_obj, "within_days", within_days))
                threshold = float(getattr(split_obj, "all_below_threshold", threshold))
            if amount_obj is not None and threshold == 100000.0:
                threshold = float(getattr(amount_obj, "vendor_payment_approval_above", threshold))

        return _SplitBillingRules(
            min_invoices_to_flag=max(1, min_invoices),
            within_days=max(1, within_days),
            all_below_threshold=max(0.0, threshold),
        )

    def _build_finding(
        self,
        row: pd.Series,
        finding_type: str,
        severity: str,
        score: float,
        evidence: dict[str, Any],
        reason: str,
    ) -> AnomalyFinding:
        transaction_id = str(row.get("transaction_id", row.name))
        document_name = self._resolve_document_name(row)
        return AnomalyFinding(
            finding_id=str(uuid.uuid4()),
            document_name=document_name,
            finding_type=finding_type,
            severity=severity,
            score=float(max(0.0, min(1.0, score))),
            evidence=evidence,
            human_readable_reason=reason,
            explanation=None,
            transaction_ids=[transaction_id],
        )

    def _build_group_finding(
        self,
        rows: pd.DataFrame,
        finding_type: str,
        severity: str,
        score: float,
        evidence: dict[str, Any],
        reason: str,
    ) -> AnomalyFinding:
        tx_ids = [str(txn_id) for txn_id in rows.get("transaction_id", pd.Series([], dtype=object)).tolist()]
        tx_ids = list(dict.fromkeys([txn for txn in tx_ids if txn]))
        document_name = self._resolve_document_name(rows.iloc[0]) if not rows.empty else "unknown"

        return AnomalyFinding(
            finding_id=str(uuid.uuid4()),
            document_name=document_name,
            finding_type=finding_type,
            severity=severity,
            score=float(max(0.0, min(1.0, score))),
            evidence=evidence,
            human_readable_reason=reason,
            explanation=None,
            transaction_ids=tx_ids,
        )

    def _merge_findings(self, findings: list[AnomalyFinding]) -> list[AnomalyFinding]:
        merged_groups: list[dict[str, Any]] = []
        severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

        for finding in findings:
            tx_set = set(finding.transaction_ids)
            match_index = None

            for index, group in enumerate(merged_groups):
                if tx_set and tx_set.intersection(group["tx_ids"]):
                    match_index = index
                    break

            if match_index is None:
                merged_groups.append({"tx_ids": tx_set, "finding": finding})
                continue

            current = merged_groups[match_index]["finding"]
            selected = current

            if severity_rank.get(finding.severity, 0) > severity_rank.get(current.severity, 0):
                selected = finding
            elif severity_rank.get(finding.severity, 0) == severity_rank.get(current.severity, 0):
                if finding.score > current.score:
                    selected = finding

            merged_reason = selected.human_readable_reason or current.human_readable_reason or finding.human_readable_reason

            merged_evidence = dict(current.evidence)
            merged_evidence.setdefault("merged_finding_types", [])
            merged_types = set(merged_evidence["merged_finding_types"])
            merged_types.update([current.finding_type, finding.finding_type])
            merged_evidence["merged_finding_types"] = sorted(merged_types)

            combined_ids = sorted(set(current.transaction_ids + finding.transaction_ids))
            merged_groups[match_index]["tx_ids"] = set(combined_ids)
            merged_groups[match_index]["finding"] = AnomalyFinding(
                finding_id=selected.finding_id,
                document_name=selected.document_name,
                finding_type=selected.finding_type,
                severity=selected.severity,
                score=max(current.score, finding.score),
                evidence=merged_evidence,
                human_readable_reason=merged_reason,
                explanation=None,
                transaction_ids=combined_ids,
            )

        return [group["finding"] for group in merged_groups]

    def _safe_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _find_existing_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        lower_lookup = {str(col).strip().lower(): str(col) for col in df.columns}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in lower_lookup:
                return lower_lookup[key]

        for candidate in candidates:
            key = candidate.strip().lower()
            for col in df.columns:
                col_key = str(col).strip().lower()
                if key in col_key or col_key in key:
                    return str(col)

        return None

    def _signed_amount_series(self, data: pd.DataFrame) -> pd.Series:
        if "debit" in data.columns or "credit" in data.columns:
            debit = pd.to_numeric(data.get("debit", 0), errors="coerce").fillna(0.0)
            credit = pd.to_numeric(data.get("credit", 0), errors="coerce").fillna(0.0)
            return credit - debit

        tx_type_col = self._find_existing_column(data, ["transaction_type", "type"])
        if tx_type_col:
            tx_type = data[tx_type_col].fillna("").astype(str).str.lower().str.strip()
            sign = tx_type.map({"credit": 1, "cr": 1, "debit": -1, "dr": -1}).fillna(-1)
            return data["amount"].astype(float) * sign.astype(float)

        return -data["amount"].astype(float)

    def _invoice_to_dict(self, invoice: Any) -> dict[str, Any]:
        if invoice is None:
            return {}
        if isinstance(invoice, dict):
            return invoice
        if hasattr(invoice, "model_dump"):
            return invoice.model_dump()
        if hasattr(invoice, "dict"):
            return invoice.dict()
        return {k: getattr(invoice, k) for k in dir(invoice) if not k.startswith("_") and not callable(getattr(invoice, k))}

    def _collect_employee_names(self, data: pd.DataFrame) -> set[str]:
        if data is None or data.empty:
            return set()
        names: set[str] = set()
        for col in ["employee_name", "submitted_by", "approved_by", "user"]:
            if col in data.columns:
                values = data[col].fillna("").astype(str).str.strip()
                names.update({val for val in values.tolist() if val})
        return names

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            text = str(value).strip().replace(",", "")
            if text == "":
                return None
            try:
                return float(text)
            except Exception:
                return None
