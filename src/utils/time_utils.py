"""Time-band normalisation.

NUMBAT time labels look like ``04:00-04:15``, ``0400-0415`` or free text
('AM Peak'). Parsing is attempted; unparseable labels are preserved verbatim in
``time_period`` and flagged via ``time_parsed`` — never rewritten to invented
values. The canonical operating day assumed for gap detection is 04:00-01:00
(21 hours => 84 fifteen-minute periods), configurable via ``EXPECTED_PERIODS``.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

PERIOD_MINUTES = 15
DAY_START = 4 * 60      # 04:00
DAY_END = 25 * 60       # 01:00 next day (25:00)

_RANGE_RE = re.compile(
    r"^\s*(\d{1,2}):?(\d{2})\s*[-\u2013\u2014to]+\s*(\d{1,2}):?(\d{2})\s*$",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"^\s*(\d{1,2}):?(\d{2})\s*$")


def expected_time_periods() -> list[str]:
    """All 84 canonical 15-minute labels of the 04:00-01:00 operating day."""
    out = []
    for m in range(DAY_START, DAY_END, PERIOD_MINUTES):
        out.append(f"{label_from_minutes(m)}-{label_from_minutes(m + PERIOD_MINUTES)}")
    return out


def label_from_minutes(total_minutes: int) -> str:
    total_minutes %= 24 * 60
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def parse_timeband(value: object) -> tuple[str | None, int | None, int | None, bool]:
    """Return (canonical_label, start_minute, end_minute, parsed_ok)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None, None, False
    if isinstance(value, pd.Timestamp):
        start = value.hour * 60 + value.minute
        end = start + PERIOD_MINUTES
        return f"{label_from_minutes(start)}-{label_from_minutes(end)}", start, end, True

    s = str(value).strip()
    m = _RANGE_RE.match(s)
    if m:
        h1, m1, h2, m2 = (int(g) for g in m.groups())
        if not (0 <= m1 < 60 and 0 <= m2 < 60 and 0 <= h1 <= 25 and 0 <= h2 <= 25):
            return s, None, None, False
        start = (h1 % 24) * 60 + m1
        end = (h2 % 24) * 60 + m2
        if end < start:              # crosses midnight (e.g. 23:45-00:00)
            end += 24 * 60
        elif end == start:           # zero-length label: treat as one period
            end = start + PERIOD_MINUTES
        return f"{label_from_minutes(start)}-{label_from_minutes(end)}", start, end, True

    m = _CLOCK_RE.match(s)
    if m:
        h1, m1 = int(m.group(1)), int(m.group(2))
        if 0 <= m1 < 60 and 0 <= h1 <= 25:
            start = (h1 % 24) * 60 + m1
            end = start + PERIOD_MINUTES
            return f"{label_from_minutes(start)}-{label_from_minutes(end)}", start, end, True

    return s, None, None, False


def normalise_time_column(series: pd.Series) -> pd.DataFrame:
    """Expand a raw time column into canonical label + start/end minutes."""
    parsed = series.map(parse_timeband)
    return pd.DataFrame(
        {
            "time_period": [p[0] for p in parsed],
            "time_start": [p[1] for p in parsed],
            "time_end": [p[2] for p in parsed],
            "time_parsed": [p[3] for p in parsed],
        },
        index=series.index,
    )


def missing_periods(observed: Iterable[str]) -> list[str]:
    obs = {str(p) for p in observed}
    return [p for p in expected_time_periods() if p not in obs]
