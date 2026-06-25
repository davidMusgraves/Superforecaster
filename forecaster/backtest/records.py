"""ResolvedRecord — one resolved binary question, frozen for offline scoring.

This is the on-disk unit of the backtest. The loader (metaculus_loader.py)
fetches resolved questions from Metaculus and writes a JSON list of these; the
harness (harness.py) reads them back and scores. Keeping the record a pure,
SDK-free dataclass means the *scoring* half of the backtest runs anywhere
(including the Python 3.10 dev sandbox), while only the *fetch* half needs the
forecasting_tools SDK and Python 3.11.

Binary-only for now (the calibrator and scoring map cleanly onto binary);
numeric/MC come later.

No-look-ahead note: ``community_prob`` is whatever snapshot the loader captured.
With the SDK loader that is the *latest* community prediction (recency-weighted
center at access time), i.e. the value just before resolution — an optimistic,
near-omniscient baseline. The honest lead-time snapshot (community prediction as
of question open / a fixed horizon) needs the aggregation history and arrives
with the HTTP loader. ``cp_is_final`` records which kind this is so the harness
can label the baseline truthfully.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ResolvedRecord:
    question_id: int                 # Metaculus post id
    question_text: str
    url: str
    outcome: int                     # 1 if resolved YES, 0 if resolved NO
    community_prob: float | None     # community prediction snapshot (see module docstring)
    cp_is_final: bool                # True => latest/near-resolution CP (optimistic baseline)
    category: str = "global"         # coarse category key (per-category keying comes with B1/B6)
    num_forecasters: int | None = None
    open_time: str | None = None     # ISO 8601 strings; kept as text so the record is pure JSON
    close_time: str | None = None
    resolve_time: str | None = None
    source: str = "metaculus"

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {self.outcome!r}")
        if self.community_prob is not None and not 0.0 <= self.community_prob <= 1.0:
            raise ValueError(
                f"community_prob must be in [0,1], got {self.community_prob!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResolvedRecord":
        # Tolerate extra keys from future loader versions.
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


def save_records(records: list[ResolvedRecord], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "resolved_record/v1",
        "count": len(records),
        "records": [r.to_dict() for r in records],
    }
    p.write_text(json.dumps(payload, indent=2))


def load_records(path: str | Path) -> list[ResolvedRecord]:
    p = Path(path)
    data = json.loads(p.read_text())
    rows = data["records"] if isinstance(data, dict) else data
    return [ResolvedRecord.from_dict(r) for r in rows]
