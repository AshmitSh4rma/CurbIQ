"""Demo: run the camera-ingestion pipeline on a real street image.

Detects vehicles, geofences a no-parking zone, emits violation records in the
dataset schema, and saves an annotated image. Uses the SimulationDetector unless
a real YOLOv8 ONNX model is provided via the CURBIQ_YOLO_ONNX env var
(`pip install onnxruntime`, `yolo export model=yolov8n.pt format=onnx`).

    PYTHONPATH=. .venv/bin/python scripts/cv_demo.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from curbiq.cv import (CameraConfig, Homography, NoParkingZone, load_detector,
                       process_image, process_video, to_violation_records)

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "samples" / "street.jpg"
OUT = ROOT / "data" / "samples" / "street_annotated.jpg"


def build_camera(w: int, h: int) -> CameraConfig:
    # no-parking zone over the kerbside foreground (relative coords -> pixels)
    zone_rel = [(0.00, 0.58), (0.70, 0.58), (0.70, 1.00), (0.00, 1.00)]
    zone = NoParkingZone("NPZ-1", [(x * w, y * h) for x, y in zone_rel], offence_codes=[113])
    # 4-point homography: image corners -> a ~40 m patch near Safina Plaza junction
    homography = Homography(
        src_px=[(0, 0), (w, 0), (w, h), (0, h)],
        dst_lonlat=[(77.6055, 12.9852), (77.6065, 12.9852),
                    (77.6065, 12.9848), (77.6055, 12.9848)])
    return CameraConfig(
        camera_id="CAM-BLR-0007", location="Safina Plaza, Infantry Road, Shivajinagar, Bengaluru",
        police_station="Shivajinagar", junction_name="BTP051 - Safina Plaza Junction",
        center_code="16", zones=[zone], homography=homography)


def main():
    img = np.asarray(Image.open(IMG).convert("RGB"))
    h, w = img.shape[:2]
    camera = build_camera(w, h)
    model = os.environ.get("CURBIQ_YOLO_ONNX") or str(ROOT / "models" / "ssd_mobilenet.onnx")
    detector = load_detector(model_path=model if Path(model).exists() else None)
    print(f"detector: {type(detector).__name__}  | image {w}x{h}")

    res = process_image(img, detector, camera, timestamp="2024-04-08T18:30:00+00:00")
    print(f"\ndetections: {len(res['detections'])}")
    for d in res["detections"]:
        print(f"  {d.label:18} conf={d.conf:.2f}  box={tuple(round(v) for v in d.xyxy)}")
    print(f"\nviolations (in no-parking zone): {len(res['events'])}")
    for e in res["events"]:
        print(f"  {e.vehicle_type:18} @ ({e.lat:.5f},{e.lon:.5f})  zone={e.zone_id}  conf={e.conf:.2f}")

    recs = to_violation_records(res["events"])
    print(f"\nviolation records (dataset schema) — {len(recs)} rows, sample:")
    if len(recs):
        r = recs.iloc[0]
        for k in ("id", "latitude", "longitude", "vehicle_type", "violation_type",
                  "offence_code", "created_datetime", "police_station", "junction_name",
                  "validation_status"):
            print(f"    {k:18}: {r[k]}")

    # video dwell demo: same frame repeated for 90 s at 1 fps -> dwell violation
    frames = [img] * 90
    vevents = process_video(frames, detector, camera, fps=1.0, dwell_seconds=60.0,
                            start_time="2024-04-08T18:30:00+00:00")
    print(f"\nvideo dwell demo (90 frames @1fps, 60s threshold): "
          f"{len(vevents)} dwell violations (e.g. track {vevents[0].track_id} "
          f"after {vevents[0].dwell_s:.0f}s)" if vevents else "\nvideo dwell demo: no dwell violations")

    # annotate + save
    pim = Image.open(IMG).convert("RGB")
    dr = ImageDraw.Draw(pim)
    zx = [(x, y) for x, y in camera.zones[0].polygon_xy]
    dr.polygon(zx, outline=(255, 80, 60), width=4)
    for d in res["detections"]:
        bx, by = d.bottom_center
        viol = camera.zone_for(bx, by) is not None
        col = (255, 59, 48) if viol else (52, 211, 153)
        dr.rectangle(list(d.xyxy), outline=col, width=3)
        dr.text((d.xyxy[0] + 2, d.xyxy[1] + 2), f"{d.label} {d.conf:.2f}", fill=col)
    pim.save(OUT)
    print(f"\nannotated image -> {OUT}")


if __name__ == "__main__":
    main()
