"""
CSV validation helpers for notebook/app ingestion.
"""

from typing import Dict, List, Tuple
import re

import pandas as pd


FORMULA_PREFIX_RE = re.compile(r"^\s*[=+\-@]")

BASE_REQUIRED_COLUMNS = [
    "Breed",
    "Country of Origin",
    "Fur Color",
    "Height (in)",
    "Color of Eyes",
    "Longevity (yrs)",
    "Character Traits",
    "Common Health Problems",
]

RANKING_REQUIRED_COLUMNS = [
    "Breed",
    "score",
    "popularity ranking",
    "intelligence",
]


def _find_formula_risk_cells(df: pd.DataFrame) -> List[Tuple[str, int]]:
    risky = []
    for col in df.columns:
        if df[col].dtype == object:
            count = df[col].fillna("").astype(str).str.match(FORMULA_PREFIX_RE).sum()
            if count:
                risky.append((col, int(count)))
    return risky


def validate_csv(df: pd.DataFrame, required_columns: List[str], name: str) -> Dict[str, object]:
    missing = [c for c in required_columns if c not in df.columns]
    duplicate_rows = int(df.duplicated().sum())
    duplicate_breed = int(df["Breed"].duplicated().sum()) if "Breed" in df.columns else 0
    null_counts = {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0}
    formula_risk = _find_formula_risk_cells(df)

    return {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing,
        "duplicate_rows": duplicate_rows,
        "duplicate_breed_values": duplicate_breed,
        "null_counts": null_counts,
        "formula_risk_cells": formula_risk,
        "is_valid_schema": len(missing) == 0,
    }


def validate_base_and_ranking(base_df: pd.DataFrame, ranking_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    return {
        "base_csv": validate_csv(base_df, BASE_REQUIRED_COLUMNS, "base_csv"),
        "ranking_csv": validate_csv(ranking_df, RANKING_REQUIRED_COLUMNS, "ranking_csv"),
    }
