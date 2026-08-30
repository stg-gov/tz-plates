"""Temporal OCR aggregation (spec §17).

For one vehicle track, plate readings from many frames are fused with
OCR-probability-weighted voting: every frame contributes to a per-character
score, weighted by that frame's confidence and the OCR posterior for the
character. The winning string is re-validated against the Tanzania rules, and the
aggregated confidence is a noisy-OR over the frames that agree with it — so
several confident, agreeing frames yield a higher confidence than any single one
(e.g. T331E8G/T331EBG/T331EBG/T331E8G -> T331EBG @ 0.97).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import exp

from tz_alpr.country_rules.tanzania import TanzaniaRules
from tz_alpr.ocr.ctc_decode import CharPosterior
from tz_alpr.postprocessing.confidence import ConfidenceModel

_MAX_AGG_CONF = 0.995


@dataclass
class PlateObservation:
    frame_idx: int
    timestamp: float
    raw_ocr: str
    normalized_text: str
    positions: list[CharPosterior]
    seq_confidence: float
    final_confidence: float
    plate_det_conf: float
    plate_type: str
    corrections: list[str] = field(default_factory=list)
    vehicle_type: str = "unknown"


@dataclass
class TrackAggregate:
    track_id: int
    plate: str
    raw_ocr: str
    confidence: float
    plate_type: str
    review_status: str
    vehicle_type: str
    corrections: list[str]
    n_frames: int
    first_frame: int
    last_frame: int
    first_timestamp: float
    last_timestamp: float
    per_frame: list[tuple[int, str, float]]  # (frame_idx, normalized_text, final_confidence)


class TemporalPlateAggregator:
    def __init__(self, rules: TanzaniaRules, confidence: ConfidenceModel) -> None:
        self._rules = rules
        self._confidence = confidence
        self._store: dict[int, list[PlateObservation]] = defaultdict(list)

    def add(self, track_id: int, obs: PlateObservation) -> None:
        if obs.normalized_text or obs.raw_ocr:
            self._store[track_id].append(obs)

    def has(self, track_id: int) -> bool:
        return bool(self._store.get(track_id))

    def observation_count(self, track_id: int) -> int:
        return len(self._store.get(track_id, ()))

    def drop(self, track_id: int) -> None:
        self._store.pop(track_id, None)

    def active_track_ids(self) -> list[int]:
        return list(self._store)

    def aggregate(self, track_id: int) -> TrackAggregate | None:
        obs_list = self._store.get(track_id)
        if not obs_list:
            return None

        by_len: dict[int, float] = defaultdict(float)
        for o in obs_list:
            if o.normalized_text:
                by_len[len(o.normalized_text)] += max(o.final_confidence, 1e-3)
        if not by_len:
            return None

        target_len = max(by_len, key=lambda k: by_len[k])
        cohort = [o for o in obs_list if len(o.normalized_text) == target_len]

        scores: list[dict[str, float]] = [defaultdict(float) for _ in range(target_len)]
        for o in cohort:
            weight = max(o.final_confidence, 1e-3)
            for i in range(target_len):
                ch = o.normalized_text[i]
                posterior = o.positions[i].prob_of(ch) if i < len(o.positions) else 1.0
                scores[i][ch] += weight * max(posterior, 1e-3)
                if i < len(o.positions):
                    for alt_c, alt_p in o.positions[i].alternatives[:3]:
                        if alt_c != ch:
                            scores[i][alt_c] += weight * alt_p * 0.5

        voted = "".join(max(s, key=lambda k: s[k]) for s in scores)
        category = self._rules.classify(voted)

        agreeing = [o for o in cohort if o.normalized_text == voted]
        if agreeing:
            agg_conf = _boosted_confidence([o.final_confidence for o in agreeing])
        else:
            per_pos = [
                scores[i][voted[i]] / max(sum(scores[i].values()), 1e-9) for i in range(target_len)
            ]
            best_single = max(o.final_confidence for o in cohort)
            agg_conf = (sum(per_pos) / len(per_pos)) * best_single

        agg_conf = round(min(agg_conf, _MAX_AGG_CONF), 4)

        corrections: list[str] = []
        for o in agreeing or cohort:
            for c in o.corrections:
                if c not in corrections:
                    corrections.append(c)

        vehicle_type = _mode(o.vehicle_type for o in obs_list)
        first, last = obs_list[0], obs_list[-1]
        return TrackAggregate(
            track_id=track_id,
            plate=voted,
            raw_ocr=_mode(o.raw_ocr for o in cohort) or voted,
            confidence=agg_conf,
            plate_type=category.name,
            review_status=self._confidence.route(agg_conf),
            vehicle_type=vehicle_type,
            corrections=corrections,
            n_frames=len(obs_list),
            first_frame=first.frame_idx,
            last_frame=last.frame_idx,
            first_timestamp=first.timestamp,
            last_timestamp=last.timestamp,
            per_frame=[(o.frame_idx, o.normalized_text, round(o.final_confidence, 3)) for o in obs_list],
        )


def _boosted_confidence(confs: list[float]) -> float:
    """Agreement across frames raises confidence above the best single frame, but
    with diminishing returns — consecutive frames of the same plate are correlated,
    not independent, so a naive noisy-OR would be badly over-confident.

        agg = best + (1 - best) * (1 - exp(-k * extra_support))     (capped)

    where ``extra_support`` is the summed confidence of the *additional* agreeing
    frames. Example: 0.91 & 0.96 -> ~0.97.
    """
    best = max(confs)
    extra_support = sum(confs) - best
    gap_closed = min(0.70, 1.0 - exp(-0.35 * extra_support))
    return best + (1.0 - best) * gap_closed


def _mode(values) -> str:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        if v:
            counts[v] += 1
    return max(counts, key=lambda k: counts[k]) if counts else ""
