"""Policy configuration loader and validator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AmountThresholds:
    office_supplies: int
    travel: int
    software: int
    hardware: int
    vendor_payment_approval_above: int


@dataclass(frozen=True)
class DuplicateDetection:
    same_vendor_same_amount_within_days: int
    similar_amount_tolerance_percent: float


@dataclass(frozen=True)
class Holiday:
    name: str
    date: str


@dataclass(frozen=True)
class TimingRules:
    business_hours_start: int
    business_hours_end: int
    flag_weekends: bool
    flag_public_holidays: bool
    indian_holidays: list[Holiday]


@dataclass(frozen=True)
class SplitBilling:
    min_invoices_to_flag: int
    within_days: int
    all_below_threshold: int


@dataclass(frozen=True)
class RoundNumberBias:
    suspicious_round_amounts: list[int]
    tolerance: int


@dataclass(frozen=True)
class NewVendor:
    scrutiny_above_amount: int
    require_gst_verification: bool


@dataclass(frozen=True)
class GstRules:
    format_regex: str
    amount_tolerance_percent: float


@dataclass(frozen=True)
class PolicyConfig:
    amount_thresholds: AmountThresholds
    duplicate_detection: DuplicateDetection
    timing_rules: TimingRules
    split_billing: SplitBilling
    round_number_bias: RoundNumberBias
    new_vendor: NewVendor
    gst_rules: GstRules


_POLICY_SINGLETON: PolicyConfig | None = None


def _require_dict(value: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{field_path}' to be a mapping/dict.")
    return value


def _require_field(data: dict[str, Any], key: str, field_path: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required field: '{field_path}.{key}'")
    return data[key]


def _parse_holidays(data: list[Any]) -> list[Holiday]:
    holidays: list[Holiday] = []
    for index, item in enumerate(data):
        item_path = f"timing_rules.indian_holidays[{index}]"
        item_dict = _require_dict(item, item_path)
        name = _require_field(item_dict, "name", item_path)
        date = _require_field(item_dict, "date", item_path)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Expected '{item_path}.name' to be a non-empty string.")
        if not isinstance(date, str) or not date.strip():
            raise ValueError(f"Expected '{item_path}.date' to be a non-empty string.")
        holidays.append(Holiday(name=name.strip(), date=date.strip()))
    return holidays


def _build_policy_config(raw: dict[str, Any]) -> PolicyConfig:
    amount_raw = _require_dict(_require_field(raw, "amount_thresholds", "root"), "amount_thresholds")
    dup_raw = _require_dict(_require_field(raw, "duplicate_detection", "root"), "duplicate_detection")
    timing_raw = _require_dict(_require_field(raw, "timing_rules", "root"), "timing_rules")
    split_raw = _require_dict(_require_field(raw, "split_billing", "root"), "split_billing")
    round_raw = _require_dict(_require_field(raw, "round_number_bias", "root"), "round_number_bias")
    vendor_raw = _require_dict(_require_field(raw, "new_vendor", "root"), "new_vendor")
    gst_raw = _require_dict(_require_field(raw, "gst_rules", "root"), "gst_rules")

    holidays_raw = _require_field(timing_raw, "indian_holidays", "timing_rules")
    if not isinstance(holidays_raw, list):
        raise ValueError("Expected 'timing_rules.indian_holidays' to be a list.")

    return PolicyConfig(
        amount_thresholds=AmountThresholds(
            office_supplies=int(_require_field(amount_raw, "office_supplies", "amount_thresholds")),
            travel=int(_require_field(amount_raw, "travel", "amount_thresholds")),
            software=int(_require_field(amount_raw, "software", "amount_thresholds")),
            hardware=int(_require_field(amount_raw, "hardware", "amount_thresholds")),
            vendor_payment_approval_above=int(
                _require_field(amount_raw, "vendor_payment_approval_above", "amount_thresholds")
            ),
        ),
        duplicate_detection=DuplicateDetection(
            same_vendor_same_amount_within_days=int(
                _require_field(dup_raw, "same_vendor_same_amount_within_days", "duplicate_detection")
            ),
            similar_amount_tolerance_percent=float(
                _require_field(dup_raw, "similar_amount_tolerance_percent", "duplicate_detection")
            ),
        ),
        timing_rules=TimingRules(
            business_hours_start=int(_require_field(timing_raw, "business_hours_start", "timing_rules")),
            business_hours_end=int(_require_field(timing_raw, "business_hours_end", "timing_rules")),
            flag_weekends=bool(_require_field(timing_raw, "flag_weekends", "timing_rules")),
            flag_public_holidays=bool(_require_field(timing_raw, "flag_public_holidays", "timing_rules")),
            indian_holidays=_parse_holidays(holidays_raw),
        ),
        split_billing=SplitBilling(
            min_invoices_to_flag=int(_require_field(split_raw, "min_invoices_to_flag", "split_billing")),
            within_days=int(_require_field(split_raw, "within_days", "split_billing")),
            all_below_threshold=int(_require_field(split_raw, "all_below_threshold", "split_billing")),
        ),
        round_number_bias=RoundNumberBias(
            suspicious_round_amounts=[int(value) for value in _require_field(
                round_raw, "suspicious_round_amounts", "round_number_bias"
            )],
            tolerance=int(_require_field(round_raw, "tolerance", "round_number_bias")),
        ),
        new_vendor=NewVendor(
            scrutiny_above_amount=int(_require_field(vendor_raw, "scrutiny_above_amount", "new_vendor")),
            require_gst_verification=bool(
                _require_field(vendor_raw, "require_gst_verification", "new_vendor")
            ),
        ),
        gst_rules=GstRules(
            format_regex=str(_require_field(gst_raw, "format_regex", "gst_rules")),
            amount_tolerance_percent=float(
                _require_field(gst_raw, "amount_tolerance_percent", "gst_rules")
            ),
        ),
    )


def load_policy_config(policy_file: Path | None = None) -> PolicyConfig:
    """Load and validate policy configuration from YAML."""
    config_path = policy_file or (Path(__file__).parent / "policy_rules.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Policy rules file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}

    if not isinstance(parsed, dict):
        raise ValueError("Policy YAML root must be a mapping/dict.")

    return _build_policy_config(parsed)


def get_policy() -> PolicyConfig:
    """Return singleton policy config loaded from disk once."""
    global _POLICY_SINGLETON
    if _POLICY_SINGLETON is None:
        _POLICY_SINGLETON = load_policy_config()
    return _POLICY_SINGLETON
