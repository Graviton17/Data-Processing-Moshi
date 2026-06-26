"""Unit tests for the pure chunk-filtering logic (no ML deps required)."""

from dataprep.config import FiltersConfig
from dataprep.models import Segment
from dataprep.stages.filters import (
    compute_filtered_segments,
    long_turn_intervals,
    merge_intervals,
    overcrowded_intervals,
    subtract,
    talk_times,
)


def test_merge_intervals():
    assert merge_intervals([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert merge_intervals([]) == []


def test_subtract():
    assert subtract(0, 10, [(2, 4)]) == [(0, 2), (4, 10)]
    assert subtract(0, 10, [(0, 10)]) == []
    assert subtract(0, 10, []) == [(0, 10)]


def test_talk_times():
    segs = [Segment("A", 0, 5), Segment("B", 5, 8), Segment("A", 8, 10)]
    assert talk_times(segs) == {"A": 7.0, "B": 3.0}


def test_overcrowded_intervals():
    segs = [Segment("A", 0, 10), Segment("B", 5, 15), Segment("C", 6, 7)]
    # only 6-7 has 3 simultaneous speakers
    assert overcrowded_intervals(segs, max_speakers=2) == [(6, 7)]


def test_long_turn_merges_across_small_gaps():
    segs = [Segment("A", 0, 30), Segment("A", 30.2, 70)]  # gap 0.2 < merge_gap
    out = long_turn_intervals(segs, "A", max_turn=60, merge_gap=0.5)
    assert out == [(0, 70)]


def test_long_turn_under_limit_is_kept():
    segs = [Segment("A", 0, 30), Segment("A", 40, 55)]  # gap too big to merge
    assert long_turn_intervals(segs, "A", max_turn=60, merge_gap=0.5) == []


def _cfg(**kw):
    return FiltersConfig(**kw)


def test_drop_when_fewer_than_two_speakers():
    segs = [Segment("A", 0, 30)]
    pair, excluded, retained, reason = compute_filtered_segments(segs, _cfg())
    assert reason == "fewer than 2 speakers"


def test_third_speaker_chunk_is_excluded():
    segs = [
        Segment("A", 0, 20),
        Segment("B", 20, 35),
        Segment("C", 35, 37),  # interloper
        Segment("A", 37, 50),
    ]
    pair, excluded, retained, reason = compute_filtered_segments(segs, _cfg())
    assert reason is None
    assert set(pair) == {"A", "B"}
    assert all(s.speaker in ("A", "B") for s in retained)
    # C's span removed entirely
    assert not any(s.overlaps(35, 37) for s in retained)


def test_long_monologue_chunk_is_dropped():
    segs = [
        Segment("A", 0, 70),    # 70s monologue -> dropped
        Segment("B", 70, 90),
        Segment("A", 90, 100),
    ]
    pair, excluded, retained, reason = compute_filtered_segments(segs, _cfg())
    assert reason is None
    # the 0-70 monologue is gone, the later A 90-100 stays
    assert not any(s.speaker == "A" and s.start < 70 for s in retained)
    assert any(s.speaker == "A" and s.start >= 90 for s in retained)


def test_drop_when_insufficient_retained_speech():
    segs = [Segment("A", 0, 70), Segment("B", 70, 71)]
    pair, excluded, retained, reason = compute_filtered_segments(
        segs, _cfg(min_variant_speech_sec=5.0)
    )
    assert reason == "insufficient retained speech after filtering"
