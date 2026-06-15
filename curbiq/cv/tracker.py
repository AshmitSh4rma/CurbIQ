"""Lightweight IOU tracker with dwell accounting for stationary-vehicle detection."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: tuple
    cls_id: int
    label: str
    first_ts: float
    last_ts: float
    hits: int = 1
    misses: int = 0
    frames_in_zone: int = 0
    seconds_in_zone: float = 0.0
    zone_id: str | None = None
    flagged: bool = False               # already emitted a violation


class IouTracker:
    """Greedy IOU association — fine for fixed-camera scenes with modest counts."""

    def __init__(self, iou_thr: float = 0.3, max_misses: int = 5):
        self.iou_thr = iou_thr
        self.max_misses = max_misses
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections, ts: float, fps: float = 1.0) -> list[Track]:
        dt = 1.0 / fps if fps > 0 else 1.0
        unmatched = set(self.tracks)
        used_dets = set()

        # match each detection to the best unmatched track above the IOU threshold
        for di, det in enumerate(detections):
            best_id, best_iou = None, self.iou_thr
            for tid in unmatched:
                v = iou(self.tracks[tid].box, det.xyxy)
                if v >= best_iou:
                    best_id, best_iou = tid, v
            if best_id is not None:
                t = self.tracks[best_id]
                t.box, t.cls_id, t.label = det.xyxy, det.cls_id, det.label
                t.hits += 1
                t.misses = 0
                t.last_ts = ts
                unmatched.discard(best_id)
                used_dets.add(di)

        # age unmatched tracks; drop the stale ones
        for tid in list(unmatched):
            self.tracks[tid].misses += 1
            if self.tracks[tid].misses > self.max_misses:
                del self.tracks[tid]

        # spawn tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            self.tracks[self._next_id] = Track(self._next_id, det.xyxy, det.cls_id,
                                               det.label, ts, ts)
            self._next_id += 1
        return list(self.tracks.values())

    def mark_in_zone(self, track: Track, zone_id: str | None, fps: float = 1.0):
        """Accumulate dwell time for a track currently inside a no-parking zone."""
        dt = 1.0 / fps if fps > 0 else 1.0
        if zone_id is not None:
            track.frames_in_zone += 1
            track.seconds_in_zone += dt
            track.zone_id = zone_id
        else:
            track.frames_in_zone = 0
            track.seconds_in_zone = 0.0
            track.zone_id = None
