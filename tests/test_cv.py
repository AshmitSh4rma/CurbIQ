"""Tests for the camera-ingestion pipeline (detector math, geofence, tracker, schema)."""
import json

import numpy as np

from curbiq.cv import (CameraConfig, Homography, NoParkingZone, SimulationDetector,
                       process_image, to_violation_records)
from curbiq.cv.detector import decode_yolov8, letterbox, nms
from curbiq.cv.pipeline import _RAW_COLS
from curbiq.cv.tracker import IouTracker, iou


# ---- detector math -------------------------------------------------------
def test_letterbox_shape_and_ratio():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas, r, left, top = letterbox(img, 640)
    assert canvas.shape == (640, 640, 3)
    assert abs(r - 3.2) < 1e-6        # min(640/100, 640/200)
    assert top == (640 - 320) // 2 and left == 0


def test_nms_keeps_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5)
    assert 0 in keep and 2 in keep and 1 not in keep


def test_decode_yolov8_one_car():
    # craft a (1, 84, N) tensor: anchor 0 = strong car (class 2) at letterbox (320,320,40,20)
    out = np.zeros((1, 84, 3), dtype=np.float32)
    out[0, :4, 0] = [320, 320, 40, 20]
    out[0, 4 + 2, 0] = 0.9            # class 2 = car
    dets = decode_yolov8(out, r=1.0, left=0, top=0, conf_thr=0.25, iou_thr=0.5,
                         vehicles_only=True)
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "CAR"
    assert abs(d.xyxy[0] - 300) < 1e-6 and abs(d.xyxy[2] - 340) < 1e-6


# ---- geofence + homography ----------------------------------------------
def test_no_parking_zone_contains():
    z = NoParkingZone("z", [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert z.contains_point(5, 5)
    assert not z.contains_point(15, 5)


def test_homography_maps_corners_and_center():
    H = Homography(src_px=[(0, 0), (10, 0), (10, 10), (0, 10)],
                   dst_lonlat=[(77.60, 12.99), (77.61, 12.99), (77.61, 12.98), (77.60, 12.98)])
    lon, lat = H.to_lonlat(5, 5)            # image centre -> patch centre
    assert abs(lon - 77.605) < 1e-3 and abs(lat - 12.985) < 1e-3


# ---- tracker -------------------------------------------------------------
def test_iou():
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


class _Det:
    def __init__(self, xyxy):
        self.xyxy, self.cls_id, self.label = xyxy, 2, "CAR"


def test_tracker_persists_id_and_dwell():
    tr = IouTracker(iou_thr=0.3)
    t1 = tr.update([_Det((0, 0, 10, 10))], ts=0.0, fps=1.0)
    t2 = tr.update([_Det((0, 0, 10, 10))], ts=1.0, fps=1.0)
    assert len(t2) == 1 and t2[0].track_id == t1[0].track_id and t2[0].hits == 2
    tr.mark_in_zone(t2[0], "z1", fps=1.0)
    tr.mark_in_zone(t2[0], "z1", fps=1.0)
    assert t2[0].frames_in_zone == 2 and abs(t2[0].seconds_in_zone - 2.0) < 1e-9


# ---- end-to-end image + schema ------------------------------------------
def _camera_full(w, h):
    zone = NoParkingZone("NPZ", [(0, 0), (w, 0), (w, h), (0, h)], offence_codes=[113])
    return CameraConfig("CAM1", "Test Rd, Bengaluru", "Shivajinagar",
                        junction_name="J1", center_code="16", lat=12.98, lon=77.6, zones=[zone])


def test_ssd_real_inference_if_available():
    import pytest
    from pathlib import Path
    pytest.importorskip("onnxruntime")
    root = Path(__file__).resolve().parent.parent
    mp, img_p = root / "models" / "ssd_mobilenet.onnx", root / "data" / "samples" / "street.jpg"
    if not mp.exists() or not img_p.exists():
        pytest.skip("SSD model or sample image not present")
    from PIL import Image
    from curbiq.cv import SsdMobileNetDetector, load_detector
    from curbiq.cv.detector import SSD_COCO_VEHICLES
    det = load_detector(str(mp))
    assert isinstance(det, SsdMobileNetDetector)
    dets = det.detect(np.asarray(Image.open(img_p).convert("RGB")))
    assert len(dets) >= 1                                  # bus.jpg has a bus
    assert all(d.conf >= 0.3 for d in dets)
    assert all(d.label in set(SSD_COCO_VEHICLES.values()) for d in dets)


def test_process_image_and_records_schema():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cam = _camera_full(400, 400)
    res = process_image(img, SimulationDetector(), cam, timestamp="2024-04-08T18:30:00+00:00")
    assert len(res["events"]) == len(res["detections"]) >= 3   # all sim vehicles in full-image zone
    recs = to_violation_records(res["events"])
    assert list(recs.columns) == _RAW_COLS
    assert len(recs) == len(res["events"])
    codes = json.loads(recs.iloc[0]["offence_code"])
    labels = json.loads(recs.iloc[0]["violation_type"])
    assert codes == [113] and labels == ["NO PARKING"]
    assert recs.iloc[0]["validation_status"] == "auto_detected"
    assert recs.iloc[0]["latitude"] == 12.98
