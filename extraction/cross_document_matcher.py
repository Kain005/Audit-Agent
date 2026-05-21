"""Cross-document matching utilities for invoices and transactions."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import chromadb
import dateparser
import numpy as np
from sentence_transformers import SentenceTransformer

from .models import InvoiceEntities, TransactionEntity


class CrossDocumentMatcher:
    """Match invoices with payments and detect duplicates across documents."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        project_root = Path(__file__).parent.parent
        persist_path = project_root / "data" / "chroma_db"
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection(
            name="vendor_names",
            metadata={"hnsw:space": "cosine"},
        )

    def embed_vendor_names(self, vendors: list[str]) -> None:
        """Embed and persist vendor names for fuzzy matching."""
        cleaned_vendors = [vendor.strip() for vendor in vendors if isinstance(vendor, str) and vendor.strip()]
        if not cleaned_vendors:
            return

        unique_vendors = list(dict.fromkeys(cleaned_vendors))
        embeddings = self.model.encode(unique_vendors, normalize_embeddings=True)

        ids = [self._vendor_id(vendor) for vendor in unique_vendors]
        metadatas = [{"vendor": vendor} for vendor in unique_vendors]

        # upsert keeps the collection idempotent when same vendor list is embedded multiple times.
        self.collection.upsert(
            ids=ids,
            documents=unique_vendors,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def find_similar_vendor(self, vendor_name: str, threshold: float = 0.85) -> list[str]:
        """Return vendor names from ChromaDB above similarity threshold."""
        query_name = (vendor_name or "").strip()
        if not query_name:
            return []

        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode([query_name], normalize_embeddings=True)
        query_result = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=25,
            include=["distances", "documents"],
        )

        matched: list[str] = []
        docs = query_result.get("documents", [[]])
        distances = query_result.get("distances", [[]])

        if not docs or not distances:
            return matched

        for doc, distance in zip(docs[0], distances[0]):
            if doc is None or distance is None:
                continue
            similarity = 1.0 - float(distance)
            if similarity >= threshold and doc not in matched:
                matched.append(str(doc))

        return matched

    def match_invoice_to_payment(
        self,
        invoice: InvoiceEntities,
        transactions: list[TransactionEntity],
    ) -> dict[str, Any]:
        """Match one invoice to the best candidate payment transaction."""
        invoice_amount = self._invoice_amount(invoice)
        invoice_date = self._parse_date(invoice.invoice_date)
        invoice_vendor = (invoice.vendor_name or "").strip()

        if invoice_amount is None:
            return {
                "matched": False,
                "transaction": None,
                "confidence": 0.0,
                "match_reason": "Invoice amount missing",
            }

        best_transaction: TransactionEntity | None = None
        best_confidence = 0.0
        best_reason = "No qualifying transaction found"

        for txn in transactions:
            amount_score = self._amount_similarity(invoice_amount, txn.amount)
            if amount_score <= 0.0:
                continue

            vendor_candidate = (txn.party_name or txn.description or "").strip()
            vendor_similarity = self._vendor_similarity(invoice_vendor, vendor_candidate)
            if vendor_similarity < 0.85:
                continue

            date_score, date_ok = self._date_proximity_score(invoice_date, self._parse_date(txn.date))
            if not date_ok:
                continue

            confidence = (0.45 * amount_score) + (0.40 * vendor_similarity) + (0.15 * date_score)

            if confidence > best_confidence:
                best_confidence = confidence
                best_transaction = txn
                best_reason = (
                    "Amount within 2%, vendor similarity "
                    f"{vendor_similarity:.2f}, date proximity score {date_score:.2f}"
                )

        return {
            "matched": best_transaction is not None,
            "transaction": best_transaction,
            "confidence": round(best_confidence, 4),
            "match_reason": best_reason,
        }

    def find_duplicate_invoices(self, invoices: list[InvoiceEntities]) -> list[dict[str, Any]]:
        """Find exact and fuzzy duplicate invoice groups."""
        duplicates: list[dict[str, Any]] = []

        exact_groups: dict[tuple[str, str], list[int]] = {}
        for index, invoice in enumerate(invoices):
            vendor = (invoice.vendor_name or "").strip().lower()
            number = (invoice.invoice_number or "").strip().lower()
            if not vendor or not number:
                continue
            exact_groups.setdefault((number, vendor), []).append(index)

        for key, idxs in exact_groups.items():
            if len(idxs) > 1:
                duplicates.append(
                    {
                        "type": "exact",
                        "invoice_key": {"invoice_number": key[0], "vendor_name": key[1]},
                        "invoice_indexes": idxs,
                        "similarity": 1.0,
                    }
                )

        total = len(invoices)
        for i in range(total):
            for j in range(i + 1, total):
                first = invoices[i]
                second = invoices[j]

                vendor_similarity = self._vendor_similarity(first.vendor_name or "", second.vendor_name or "")
                if vendor_similarity <= 0.9:
                    continue

                first_amount = self._invoice_amount(first)
                second_amount = self._invoice_amount(second)
                if first_amount is None or second_amount is None:
                    continue
                if abs(first_amount - second_amount) > 0.01:
                    continue

                first_date = self._parse_date(first.invoice_date)
                second_date = self._parse_date(second.invoice_date)
                if first_date is None or second_date is None:
                    continue
                if abs((first_date - second_date).days) > 7:
                    continue

                duplicates.append(
                    {
                        "type": "fuzzy",
                        "invoice_indexes": [i, j],
                        "similarity": round(vendor_similarity, 4),
                        "amount": round(first_amount, 2),
                    }
                )

        return duplicates

    def reconcile(
        self,
        invoices: list[InvoiceEntities],
        transactions: list[TransactionEntity],
    ) -> dict[str, Any]:
        """Match invoices to payments and return reconciliation summary."""
        vendor_values = [invoice.vendor_name or "" for invoice in invoices]
        vendor_values.extend(txn.party_name or "" for txn in transactions)
        self.embed_vendor_names(vendor_values)

        matched_pairs: list[dict[str, Any]] = []
        unmatched_invoices: list[InvoiceEntities] = []
        unmatched_payments: list[TransactionEntity] = []
        used_transaction_indexes: set[int] = set()

        for invoice in invoices:
            available_transactions = [
                txn for idx, txn in enumerate(transactions) if idx not in used_transaction_indexes
            ]
            outcome = self.match_invoice_to_payment(invoice, available_transactions)

            if outcome["matched"] and outcome["transaction"] is not None:
                matched_txn = outcome["transaction"]
                matched_index = self._find_transaction_index(transactions, matched_txn)
                if matched_index is not None:
                    used_transaction_indexes.add(matched_index)
                matched_pairs.append(
                    {
                        "invoice": invoice,
                        "transaction": matched_txn,
                        "confidence": outcome["confidence"],
                        "match_reason": outcome["match_reason"],
                    }
                )
            else:
                unmatched_invoices.append(invoice)

        for idx, txn in enumerate(transactions):
            if idx not in used_transaction_indexes:
                unmatched_payments.append(txn)

        return {
            "matched_pairs": matched_pairs,
            "unmatched_invoices": unmatched_invoices,
            "unmatched_payments": unmatched_payments,
        }

    def _vendor_id(self, vendor: str) -> str:
        normalized = " ".join(vendor.lower().split())
        return f"vendor::{normalized}"

    def _invoice_amount(self, invoice: InvoiceEntities) -> float | None:
        if invoice.total_amount is not None:
            return float(invoice.total_amount)
        if invoice.subtotal is not None:
            return float(invoice.subtotal)
        return None

    def _parse_date(self, value: str | None) -> datetime | None:
        if not value:
            return None
        parsed = dateparser.parse(
            value,
            settings={"DATE_ORDER": "DMY", "PREFER_LOCALE_DATE_ORDER": False, "STRICT_PARSING": False},
        )
        return parsed

    def _vendor_similarity(self, first: str, second: str) -> float:
        left = (first or "").strip()
        right = (second or "").strip()
        if not left or not right:
            return 0.0

        embeddings = self.model.encode([left, right], normalize_embeddings=True)
        left_vector = np.array(embeddings[0], dtype=float)
        right_vector = np.array(embeddings[1], dtype=float)
        similarity = float(np.dot(left_vector, right_vector))
        return max(0.0, min(1.0, similarity))

    def _amount_similarity(self, invoice_amount: float, txn_amount: float) -> float:
        baseline = abs(float(invoice_amount))
        candidate = abs(float(txn_amount))

        if baseline <= 0:
            return 0.0

        diff_ratio = abs(baseline - candidate) / baseline
        if diff_ratio <= 0.0:
            return 1.0
        if diff_ratio <= 0.02:
            return 1.0 - (diff_ratio / 0.02) * 0.1
        return 0.0

    def _date_proximity_score(
        self,
        invoice_date: datetime | None,
        transaction_date: datetime | None,
    ) -> tuple[float, bool]:
        if invoice_date is None or transaction_date is None:
            return (0.0, False)

        min_date = invoice_date
        max_date = invoice_date + timedelta(days=30)

        if transaction_date < min_date or transaction_date > max_date:
            return (0.0, False)

        days = (transaction_date - invoice_date).days
        score = 1.0 - (days / 30.0)
        return (max(0.0, min(1.0, score)), True)

    def _find_transaction_index(
        self,
        transactions: list[TransactionEntity],
        target: TransactionEntity,
    ) -> int | None:
        target_ref = (target.reference_id or "").strip()
        target_amount = float(target.amount)
        target_date = (target.date or "").strip()

        for idx, txn in enumerate(transactions):
            ref = (txn.reference_id or "").strip()
            if target_ref and ref and ref == target_ref:
                return idx

        for idx, txn in enumerate(transactions):
            if float(txn.amount) == target_amount and (txn.date or "").strip() == target_date:
                return idx

        return None
