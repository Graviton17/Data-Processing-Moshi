"""Stage [4]: chunk-level filtering (pure logic, fully unit-testable).

Rules (all HARD DROP at chunk granularity, per the agreed design):
  * keep only the top-2 speakers by talk time; spans from any other speaker
    are excluded ("> 2 speakers -> drop that chunk");
  * any instant with more than ``max_speakers_per_chunk`` simultaneous speakers
    is excluded;
  * any single-speaker continuous turn longer than ``max_turn_seconds`` is
    excluded ("user speaking > 1 min -> drop that chunk").

The whole file is dropped only if it has < 2 speakers or too little speech
survives.
"""

from __future__ import annotations

from ..config import Config, FiltersConfig
from ..models import PipelineContext, Segment
from .base import Stage

Interval = tuple[float, float]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    out: list[list[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def subtract(start: float, end: float, excluded: list[Interval]) -> list[Interval]:
    """Return the parts of [start, end] not covered by any excluded interval."""
    pieces: list[Interval] = [(start, end)]
    for ex_s, ex_e in excluded:
        nxt: list[Interval] = []
        for s, e in pieces:
            if ex_e <= s or ex_s >= e:
                nxt.append((s, e))
                continue
            if s < ex_s:
                nxt.append((s, ex_s))
            if ex_e < e:
                nxt.append((ex_e, e))
        pieces = nxt
    return [(s, e) for s, e in pieces if e - s > 1e-3]


def talk_times(segments: list[Segment]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in segments:
        out[s.speaker] = out.get(s.speaker, 0.0) + s.duration
    return out


def overcrowded_intervals(segments: list[Segment], max_speakers: int) -> list[Interval]:
    points = sorted({s.start for s in segments} | {s.end for s in segments})
    out: list[Interval] = []
    for a, b in zip(points, points[1:]):
        mid = (a + b) / 2.0
        active = {s.speaker for s in segments if s.start <= mid < s.end}
        if len(active) > max_speakers:
            out.append((a, b))
    return merge_intervals(out)


def long_turn_intervals(
    segments: list[Segment], speaker: str, max_turn: float, merge_gap: float
) -> list[Interval]:
    segs = sorted((s for s in segments if s.speaker == speaker), key=lambda s: s.start)
    if not segs:
        return []
    out: list[Interval] = []
    cur_s, cur_e = segs[0].start, segs[0].end
    for s in segs[1:]:
        if s.start - cur_e <= merge_gap:
            cur_e = max(cur_e, s.end)
        else:
            if cur_e - cur_s > max_turn:
                out.append((cur_s, cur_e))
            cur_s, cur_e = s.start, s.end
    if cur_e - cur_s > max_turn:
        out.append((cur_s, cur_e))
    return out


def compute_filtered_segments(
    segments: list[Segment], cfg: FiltersConfig
) -> tuple[tuple[str, str] | None, list[Interval], list[Segment], str | None]:
    """Core pure function. Returns (main_pair, excluded, retained, drop_reason)."""
    talk = talk_times(segments)
    if len(talk) < 2:
        return None, [], [], "fewer than 2 speakers"

    pair = tuple(sorted(talk, key=lambda k: talk[k], reverse=True)[:2])  # type: ignore[assignment]

    excluded: list[Interval] = []
    for s in segments:
        if s.speaker not in pair:
            excluded.append((s.start, s.end))
    excluded += overcrowded_intervals(segments, cfg.max_speakers_per_chunk)
    for spk in pair:
        excluded += long_turn_intervals(segments, spk, cfg.max_turn_seconds, cfg.turn_merge_gap_sec)
    excluded = merge_intervals(excluded)

    retained: list[Segment] = []
    for s in segments:
        if s.speaker not in pair:
            continue
        for a, b in subtract(s.start, s.end, excluded):
            retained.append(Segment(speaker=s.speaker, start=a, end=b))

    retained_talk = talk_times(retained)
    if any(retained_talk.get(spk, 0.0) < cfg.min_variant_speech_sec for spk in pair):
        return pair, excluded, retained, "insufficient retained speech after filtering"

    return pair, excluded, retained, None


class FilterStage(Stage):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def _run(self, ctx: PipelineContext) -> PipelineContext:
        pair, excluded, retained, reason = compute_filtered_segments(
            ctx.segments, self.config.filters
        )
        ctx.main_pair = pair
        ctx.excluded = excluded
        ctx.segments = retained
        ctx.metadata["excluded_sec"] = round(sum(e - s for s, e in excluded), 2)
        if reason is not None:
            ctx.drop(reason)
        return ctx
