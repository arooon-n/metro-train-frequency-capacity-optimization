#!/usr/bin/env python
"""Generate an HTML report (outputs/reports/report_YYYYMMDD_HHMMSS.html) from
the latest model results, validation report, scenario summary and figures.

Usage:
    python scripts/generate_report.py
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import config as cfg
from src.utils.io import load_profile

DISCLAIMER = (
    "Academic decision-support model. This is NOT an TfL operational timetable. "
    "Optimised frequencies are model recommendations under the configured "
    "assumptions, not deployable service plans."
)


def _table(df: pd.DataFrame | None, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "<p><em>No data.</em></p>"
    show = df.head(max_rows)
    return show.to_html(index=False, border=0, classes="tbl", float_format=lambda v: f"{v:,.3f}")


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _validation_html(path: Path) -> str:
    if not path.exists():
        return "<p><em>No validation report found.</em></p>"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for c in data.get("checks", []):
        mark = "PASS" if c["passed"] else ("WARN" if c.get("severity") == "warning" else "FAIL")
        rows.append(
            f"<tr><td>{mark}</td><td>{html.escape(c['name'])}</td>"
            f"<td>{html.escape(str(c['detail']))}</td></tr>"
        )
    overall = "PASSED" if data.get("overall_passed") else "FAILED"
    expl = data.get("infeasible_explanation")
    expl_html = f"<pre class='err'>{html.escape(expl)}</pre>" if expl else ""
    return (
        f"<p>Solver status: <b>{html.escape(str(data.get('solver_status')))}</b> — "
        f"overall <b>{overall}</b></p>{expl_html}"
        f"<table class='tbl'><tr><th>Result</th><th>Check</th><th>Detail</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def main() -> int:
    cfg.ensure_dirs()
    out = cfg.OUTPUTS_DIR
    profile = load_profile()
    m1 = _read_csv(out / "model1_results.csv")
    m2 = _read_csv(out / "model2_results.csv")
    m3 = _read_csv(out / "model3_results.csv")
    scen = _read_csv(out / "scenario_results.csv")
    val = out / "validation_report.json"

    figs = sorted((cfg.FIGURES_DIR).glob("*.html")) + sorted((cfg.FIGURES_DIR).glob("*.png"))
    fig_html = "".join(
        f"<div class='fig'><h4>{html.escape(f.stem)}</h4>"
        + (
            f"<iframe src='file:///{f.resolve().as_posix()}' width='100%' height='480' "
            "frameborder='0'></iframe>"
            if f.suffix == ".html"
            else f"<img src='file:///{f.resolve().as_posix()}' style='max-width:100%'>"
        )
        + "</div>"
        for f in figs
    ) or "<p><em>No figures yet — run the models / EDA first.</em></p>"

    src_html = (
        f"<ul><li>Workbook: {html.escape(str(profile.get('workbook'))) if profile else 'n/a'}</li>"
        f"<li>Source: {cfg.NUMBAT_SOURCE_URL}</li>"
        f"<li>Lines: {', '.join(profile.get('cleaned_summary', {}).get('lines', [])) if profile else 'n/a'}</li>"
        f"<li>Day types: {', '.join(profile.get('cleaned_summary', {}).get('day_types', [])) if profile else 'n/a'}</li>"
        "</ul>"
    )

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>NUMBAT Rail Optimisation Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:2rem;color:#222;}}
h1{{border-bottom:3px solid #1f77b4;padding-bottom:.3rem;}}
h2{{margin-top:2rem;color:#1f77b4;}}
.tbl{{border-collapse:collapse;width:100%;font-size:.85rem;}}
.tbl td,.tbl th{{border:1px solid #ddd;padding:.3rem .5rem;text-align:left;}}
.tbl tr:nth-child(even){{background:#f7f7f7;}}
.err{{background:#fff3f3;border:1px solid #e0a0a0;padding:.7rem;white-space:pre-wrap;}}
.warn{{background:#fff8e1;border:1px solid #e0c060;padding:.7rem;}}
.fig{{border:1px solid #eee;margin:1rem 0;padding:.5rem;}}
</style></head><body>
<h1>Data-Driven Urban Rail Train Frequency and Capacity Optimization</h1>
<p class="warn"><b>Disclaimer:</b> {html.escape(DISCLAIMER)}</p>
<p>Generated: {datetime.now().isoformat(timespec='seconds')}</p>
<h2>1. Data source</h2>{src_html}
<h2>2. Validation</h2>{_validation_html(val)}
<h2>3. Model 1 — minimum service frequency</h2>
{_table(m1, 30) if m1 is not None else "<p><em>Not run.</em></p>"}
<h2>4. Model 2 — capacity-constrained allocation</h2>
{_table(m2, 30) if m2 is not None else "<p><em>Not run.</em></p>"}
<h2>5. Model 3 — weighted goal programming</h2>
{_table(m3, 30) if m3 is not None else "<p><em>Not run.</em></p>"}
<h2>6. Scenarios</h2>
{_table(scen) if scen is not None else "<p><em>Not run.</em></p>"}
<h2>7. Figures</h2>{fig_html}
</body></html>"""

    reports = cfg.REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    path.write_text(doc, encoding="utf-8")
    latest = reports / "report_latest.html"
    latest.write_text(doc, encoding="utf-8")
    print(f"Report written: {path}")
    print(f"Latest copy   : {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
