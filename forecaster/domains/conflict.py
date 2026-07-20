"""Read the US-Iran tracker's state file and render it as bot evidence.

David's daily loop + apply_updates.py already maintain a canonical, machine-
readable snapshot at BayesianPredictor/tracker_state.json. Its threads[] carry
the current posterior for each standing question (initial base rate updated by
cumulative likelihood ratios from dated reporting). Per the chosen design
("bot reads tracker"), the bot reads THAT file — never the lock-prone .xlsx —
and renders the active threads as a timestamped Bayesian evidence block for
related conflict questions.

This module is the bot's read side only; the human-approved daily loop remains
the single source of truth. Pure stdlib, tests under Python 3.10.

tracker_state.json threads[] schema:
    {id, question, latest_posterior, resolution_date, resolved}   # resolved: 1/0 or null
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def load_tracker_state(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def active_threads(state: dict, as_of: date | None = None) -> list[dict]:
    """Threads still open: no resolved outcome and resolution date not passed."""
    today = as_of or date.today()
    out: list[dict] = []
    for t in state.get("threads", []):
        if t.get("resolved"):  # 1/0 present => resolved; null/absent => open
            continue
        rd = t.get("resolution_date")
        if rd:
            try:
                if date.fromisoformat(rd) < today:
                    continue
            except (ValueError, TypeError):
                pass
        out.append(t)
    return out


def render_conflict_evidence(
    state: dict, max_threads: int = 12, as_of: date | None = None
) -> str:
    """Render active tracker threads as a Bayesian evidence block for research."""
    threads = active_threads(state, as_of=as_of)[:max_threads]
    if not threads:
        return ""
    lines = [
        f"CONFLICT BAYESIAN THREADS (US-Iran tracker, as of {state.get('as_of', '?')}):",
        "Reasoned, sequentially-updated priors (initial base rate x cumulative "
        "likelihood ratios from dated reporting). Use these as informed priors for "
        "related questions; do not anchor to any community/market number.",
    ]
    for t in threads:
        p = t.get("latest_posterior")
        p_s = f"{round(p * 100, 1)}%" if isinstance(p, (int, float)) else "n/a"
        line = f"- [{t.get('id')}] {t.get('question')}: current posterior ~{p_s}"
        if t.get("resolution_date"):
            line += f", resolves by {t['resolution_date']}"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Preview the conflict evidence block from tracker_state.json")
    p.add_argument("state", help="Path to BayesianPredictor/tracker_state.json")
    args = p.parse_args()
    state = load_tracker_state(args.state)
    block = render_conflict_evidence(state)
    print(block or "(no active threads)")


if __name__ == "__main__":
    main()
