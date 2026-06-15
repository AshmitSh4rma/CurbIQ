"""End-to-end camera ingestion: frame -> detections -> geofence/dwell -> violations.

Emitted ``ViolationEvent``s convert to records in the *raw dataset schema*
(``to_violation_records``), so live camera detections can be appended to the
historical CSV and re-run through the very same ETL + hotspot/congestion/
prioritization stack — closing the loop from reactive patrols to live intelligence.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass

import pandas as pd

from curbiq import config as C
from curbiq.cv.geofence import CameraConfig
from curbiq.cv.tracker import IouTracker


@dataclass
class ViolationEvent:
    ts: str                     # ISO-8601 UTC
    camera_id: str
    vehicle_type: str
    offence_codes: list[int]
    lat: float | None
    lon: float | None
    conf: float
    zone_id: str | None
    location: str
    police_station: str
    junction_name: str | None
    center_code: str | None
    track_id: int | None = None
    dwell_s: float = 0.0


def _event(camera: CameraConfig, det, zone, ts: str, track_id=None, dwell_s=0.0) -> ViolationEvent:
    bx, by = det.bottom_center
    lat, lon = camera.world_of(bx, by)
    return ViolationEvent(
        ts=ts, camera_id=camera.camera_id, vehicle_type=det.label,
        offence_codes=list(zone.offence_codes), lat=lat, lon=lon, conf=det.conf,
        zone_id=zone.zone_id, location=camera.location,
        police_station=camera.police_station, junction_name=camera.junction_name,
        center_code=camera.center_code, track_id=track_id, dwell_s=dwell_s)


def process_image(image, detector, camera: CameraConfig, timestamp: str) -> dict:
    """Single CCTV snapshot: any vehicle standing inside a no-parking zone is a violation."""
    dets = detector.detect(image)
    events = []
    for det in dets:
        bx, by = det.bottom_center
        zone = camera.zone_for(bx, by)
        if zone is not None:
            events.append(_event(camera, det, zone, timestamp))
    return {"detections": dets, "events": events}


def process_video(frames, detector, camera: CameraConfig, fps: float = 1.0,
                  dwell_seconds: float = 60.0, start_time: str | None = None) -> list[ViolationEvent]:
    """Video stream: track vehicles and flag those that DWELL in a no-parking zone.

    ``frames`` yields images (timestamps inferred from ``fps``) or (image, t_seconds) pairs.
    A violation fires once a track's accumulated in-zone dwell exceeds ``dwell_seconds``.
    """
    base = _dt.datetime.fromisoformat(start_time) if start_time else _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
    tracker = IouTracker()
    events: list[ViolationEvent] = []
    for i, frame in enumerate(frames):
        image, t = frame if isinstance(frame, tuple) else (frame, i / fps)
        ts_iso = (base + _dt.timedelta(seconds=t)).isoformat()
        dets = detector.detect(image)
        tracks = tracker.update(dets, t, fps)
        # map current track boxes to zones via their ground point
        det_by_box = {tuple(d.xyxy): d for d in dets}
        for tr in tracks:
            bx = (tr.box[0] + tr.box[2]) / 2.0
            by = tr.box[3]
            zone = camera.zone_for(bx, by)
            tracker.mark_in_zone(tr, zone.zone_id if zone else None, fps)
            if zone is not None and tr.seconds_in_zone >= dwell_seconds and not tr.flagged:
                det = det_by_box.get(tuple(tr.box))
                if det is not None:
                    events.append(_event(camera, det, zone, ts_iso,
                                         track_id=tr.track_id, dwell_s=tr.seconds_in_zone))
                    tr.flagged = True
    return events


# raw-CSV columns the ETL reads (so emitted records round-trip through build_all)
_RAW_COLS = [
    "id", "latitude", "longitude", "location", "vehicle_number", "vehicle_type",
    "description", "violation_type", "offence_code", "created_datetime",
    "closed_datetime", "modified_datetime", "device_id", "created_by_id",
    "center_code", "police_station", "data_sent_to_scita", "junction_name",
    "action_taken_timestamp", "data_sent_to_scita_timestamp",
    "updated_vehicle_number", "updated_vehicle_type", "validation_status",
    "validation_timestamp",
]


def to_violation_records(events: list[ViolationEvent]) -> pd.DataFrame:
    """Convert detected violations to rows in the raw dataset schema."""
    rows = []
    for i, e in enumerate(events):
        labels = [C.OFFENCE_LABELS.get(c, "NO PARKING") for c in e.offence_codes]
        rows.append({
            "id": f"CVDET{i:08d}",
            "latitude": e.lat, "longitude": e.lon, "location": e.location,
            "vehicle_number": None, "vehicle_type": e.vehicle_type,
            "description": None,
            "violation_type": json.dumps(labels),
            "offence_code": json.dumps(e.offence_codes),
            "created_datetime": e.ts, "closed_datetime": None,
            "modified_datetime": e.ts, "device_id": e.camera_id,
            "created_by_id": "CV_PIPELINE", "center_code": e.center_code,
            "police_station": e.police_station, "data_sent_to_scita": "FALSE",
            "junction_name": e.junction_name or "No Junction",
            "action_taken_timestamp": None, "data_sent_to_scita_timestamp": None,
            "updated_vehicle_number": None, "updated_vehicle_type": None,
            "validation_status": "auto_detected", "validation_timestamp": None,
        })
    return pd.DataFrame(rows, columns=_RAW_COLS)
