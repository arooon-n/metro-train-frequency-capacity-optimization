"""Small shared helpers."""

from __future__ import annotations

import re
import unicodedata


def norm_key(value: object) -> str:
    """Normalised comparison key: lowercase, ascii-folded, punctuation stripped."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFKD", str(value))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def display_label(values) -> str:
    """Pick the most frequent non-null original label for a group."""
    s = [str(v) for v in values if v is not None and str(v).strip() and str(v) != "nan"]
    if not s:
        return ""
    return max(set(s), key=s.count)


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""
