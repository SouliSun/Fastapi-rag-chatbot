"""Data cleaning, normalization, conflict resolution, and deterministic audit tracking."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

LOCAL_CACHE_PATH = Path("cleaned_faqs.csv")


class ConflictResolution(str, Enum):
    """Policies for handling identical questions with conflicting answers."""
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    DROP = "drop"
    ERROR = "error"


@dataclass
class CleaningReport:
    """Detailed audit metrics from the data cleaning pipeline."""
    total_loaded: int = 0
    missing_values_removed: int = 0
    exact_duplicates_removed: int = 0
    conflicts_detected: int = 0
    conflict_rows_pruned: int = 0
    total_retained: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"CleaningReport(loaded={self.total_loaded}, "
            f"null_or_blank_removed={self.missing_values_removed}, "
            f"exact_duplicates_removed={self.exact_duplicates_removed}, "
            f"conflicts_detected={self.conflicts_detected}, "
            f"conflict_rows_pruned={self.conflict_rows_pruned}, "
            f"retained={self.total_retained})"
        )


def normalize_text(text: Any) -> str:
    """Convert value to string and normalize contiguous whitespace."""
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).strip().split())


def compute_faq_id(question: str) -> str:
    """Generate a stable, deterministic ID based on the normalized question text."""
    normalized = normalize_text(question)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"faq_{digest}"


def clean_faq_dataframe(
    raw_df: pd.DataFrame,
    conflict_resolution: ConflictResolution = ConflictResolution.KEEP_FIRST,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Clean, validate, and deduplicate a FAQ dataframe deterministically.

    Audit pipeline logic:
      1. Validate existence of required columns ('question', 'answer').
      2. Normalize text and filter null / blank values.
      3. Remove exact duplicates (identical normalized question AND answer).
      4. Detect conflicting questions (same normalized question with differing answers).
      5. Apply conflict resolution policy deterministically.
      6. Assign deterministic IDs and sort canonically.

    Args:
        raw_df: Input pandas DataFrame containing 'question' and 'answer' columns.
        conflict_resolution: Policy to resolve questions associated with multiple answers.

    Returns:
        Tuple of (cleaned_df, cleaning_report)

    Raises:
        ValueError: If required columns are missing or if conflict_resolution='error'
                    and conflicting answers are encountered.
    """
    report = CleaningReport(total_loaded=len(raw_df))

    # 1. Validate required columns
    required_cols = {"question", "answer"}
    missing_cols = required_cols - set(raw_df.columns)
    if missing_cols:
        raise ValueError(
            f"DataFrame missing required column(s): {sorted(missing_cols)}"
        )

    # 2. Extract and copy required columns only
    work_df = raw_df[["question", "answer"]].copy()

    # 3. Handle non-string, null, or empty values safely
    work_df["question"] = work_df["question"].apply(normalize_text)
    work_df["answer"] = work_df["answer"].apply(normalize_text)

    # Filter out empty or whitespace-only records
    valid_mask = (work_df["question"] != "") & (work_df["answer"] != "")
    report.missing_values_removed = int((~valid_mask).sum())
    valid_df = work_df[valid_mask].copy()

    # 4. Remove EXACT duplicates (same question AND same answer)
    exact_dedup_df = valid_df.drop_duplicates(subset=["question", "answer"], keep="first")
    report.exact_duplicates_removed = len(valid_df) - len(exact_dedup_df)

    # 5. Detect conflicting questions on unique (question, answer) pairs
    grouped = exact_dedup_df.groupby("question")["answer"].unique()
    conflict_questions = grouped[grouped.apply(len) > 1]

    for q, answers in conflict_questions.items():
        conflict_info = {
            "question": q,
            "distinct_answers": [str(a) for a in answers],
            "occurrences": int(len(answers)),
        }
        report.conflicts.append(conflict_info)
        logger.warning(
            "Conflicting answers detected for question '%s': %d distinct answers",
            q,
            len(answers),
        )

    report.conflicts_detected = len(report.conflicts)

    # 6. Apply conflict resolution policy
    if report.conflicts_detected > 0:
        if conflict_resolution == ConflictResolution.ERROR:
            raise ValueError(
                f"Found {report.conflicts_detected} conflicting FAQ questions: "
                f"{report.conflicts}"
            )
        elif conflict_resolution == ConflictResolution.DROP:
            conflict_set = set(conflict_questions.index)
            conflict_mask = exact_dedup_df["question"].isin(conflict_set)
            report.conflict_rows_pruned = int(conflict_mask.sum())
            resolved_df = exact_dedup_df[~conflict_mask].copy()
        elif conflict_resolution == ConflictResolution.KEEP_FIRST:
            resolved_df = exact_dedup_df.drop_duplicates(subset=["question"], keep="first")
            report.conflict_rows_pruned = len(exact_dedup_df) - len(resolved_df)
        elif conflict_resolution == ConflictResolution.KEEP_LAST:
            resolved_df = exact_dedup_df.drop_duplicates(subset=["question"], keep="last")
            report.conflict_rows_pruned = len(exact_dedup_df) - len(resolved_df)
    else:
        resolved_df = exact_dedup_df.copy()
        report.conflict_rows_pruned = 0

    # 7. Assign deterministic FAQ IDs
    resolved_df["faq_id"] = resolved_df["question"].apply(compute_faq_id)

    # 8. Sort deterministically by faq_id / question so ordering is immutable
    final_df = resolved_df.sort_values(by=["faq_id", "question"]).reset_index(drop=True)

    report.total_retained = len(final_df)
    return final_df, report


def load_and_clean_faqs(
    dataset_name: str = "MakTek/Customer_support_faqs_dataset",
    split: str = "train",
    conflict_resolution: ConflictResolution = ConflictResolution.KEEP_FIRST,
    cache_path: Optional[Path] = LOCAL_CACHE_PATH,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Load FAQ dataset, clean it, and persist/use a local CSV cache for offline resilience."""
    # If local cache exists, use it to avoid network calls
    if cache_path and cache_path.exists():
        try:
            logger.info("Loading cached FAQs from '%s'...", cache_path)
            cached_df = pd.read_csv(cache_path)
            if {"question", "answer", "faq_id"}.issubset(cached_df.columns):
                report = CleaningReport(
                    total_loaded=len(cached_df),
                    total_retained=len(cached_df),
                )
                return cached_df, report
        except Exception as exc:
            logger.warning("Failed to read local cache '%s': %s. Re-fetching.", cache_path, exc)

    # Fetch from Hugging Face
    from datasets import load_dataset

    logger.info("Fetching dataset '%s' (split=%s) from Hugging Face...", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)
    raw_df = ds.to_pandas()
    cleaned_df, report = clean_faq_dataframe(
        raw_df, conflict_resolution=conflict_resolution
    )

    if cache_path:
        try:
            cleaned_df.to_csv(cache_path, index=False)
            logger.info("Saved cleaned FAQs to local cache '%s'.", cache_path)
        except Exception as exc:
            logger.warning("Failed to save local cache '%s': %s", cache_path, exc)

    logger.info("Cleaning finished: %s", report.summary())
    return cleaned_df, report


# Lazy singleton pattern: data is NOT loaded upon module import
_CACHED_DF: Optional[pd.DataFrame] = None


def get_clean_df() -> pd.DataFrame:
    """Access the cleaned FAQ DataFrame singleton lazily on first call."""
    global _CACHED_DF
    if _CACHED_DF is None:
        _CACHED_DF, report = load_and_clean_faqs()
    return _CACHED_DF


class _LazyDFProxy:
    """Transparent proxy that defers DataFrame loading until an attribute or item is accessed."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_clean_df(), name)

    def __getitem__(self, item: Any) -> Any:
        return get_clean_df()[item]

    def __len__(self) -> int:
        return len(get_clean_df())

    def __iter__(self):
        return iter(get_clean_df())

    def __repr__(self) -> str:
        return repr(get_clean_df())


df = _LazyDFProxy()


if __name__ == "__main__":
    cleaned, rep = load_and_clean_faqs()
    print("\nCleaning report:")
    print(rep.summary())
    if rep.conflicts:
        print(f"Conflicts ({len(rep.conflicts)}):", rep.conflicts[:3])
    print(f"\nCleaned sample (first 3 rows):\n{cleaned.head(3)}")