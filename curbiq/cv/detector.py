"""Vehicle detection backends + numpy YOLOv8 pre/post-processing.

``OnnxYoloDetector`` runs a real YOLOv8 ``.onnx`` model through onnxruntime
(install ``onnxruntime`` and export weights with ``yolo export model=yolov8n.pt
format=onnx``). ``SimulationDetector`` returns deterministic, plausible boxes so
the rest of the ingestion pipeline runs and is testable without weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# COCO class id -> CurbIQ vehicle_type (vehicle classes only).
COCO_VEHICLE_LABELS: dict[int, str] = {
    1: "BICYCLE", 2: "CAR", 3: "MOTOR CYCLE", 5: "BUS (BMTC/KSRTC)", 7: "LORRY/GOODS VEHICLE",
}


@dataclass
class Detection:
    cls_id: int
    label: str           # CurbIQ vehicle_type
    conf: float
    xyxy: tuple          # (x1, y1, x2, y2) in original-image pixels

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground-contact point (where the vehicle meets the road)."""
        return (self.xyxy[0] + self.xyxy[2]) / 2.0, self.xyxy[3]


# --------------------------------------------------------------------------- #
# numpy helpers (unit-tested independently of any model)
# --------------------------------------------------------------------------- #
def letterbox(img: np.ndarray, new_shape: int = 640, color: int = 114):
    """Resize keeping aspect ratio + pad to a square. Returns (canvas, r, left, top)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy non-max suppression on xyxy boxes. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def decode_yolov8(output: np.ndarray, r: float, left: int, top: int,
                  conf_thr: float, iou_thr: float, vehicles_only: bool) -> list[Detection]:
    """Decode a raw YOLOv8 output tensor (1, 4+nc, N) into Detections."""
    out = output[0].T                                   # (N, 4+nc)
    boxes_xywh, cls_scores = out[:, :4], out[:, 4:]
    conf = cls_scores.max(1)
    cls = cls_scores.argmax(1)
    keep = conf >= conf_thr
    boxes_xywh, conf, cls = boxes_xywh[keep], conf[keep], cls[keep]
    if len(boxes_xywh) == 0:
        return []
    xy, wh = boxes_xywh[:, :2], boxes_xywh[:, 2:]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r       # undo letterbox -> original px
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
    dets: list[Detection] = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for k in nms(xyxy[idx], conf[idx], iou_thr):
            i = idx[k]
            cid = int(cls[i])
            if vehicles_only and cid not in COCO_VEHICLE_LABELS:
                continue
            dets.append(Detection(cid, COCO_VEHICLE_LABELS.get(cid, str(cid)),
                                  float(conf[i]), tuple(map(float, xyxy[i]))))
    return dets


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class OnnxYoloDetector:
    """Real YOLOv8 detection via onnxruntime (CPU). Requires a .onnx model."""

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45,
                 imgsz: int = 640, vehicles_only: bool = True):
        import onnxruntime as ort                       # lazy: optional dependency
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.vehicles_only = vehicles_only

    def detect(self, image: np.ndarray) -> list[Detection]:
        canvas, r, left, top = letterbox(image, self.imgsz)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input_name: x})[0]
        return decode_yolov8(out, r, left, top, self.conf, self.iou, self.vehicles_only)


class SimulationDetector:
    """Deterministic stand-in detector (no weights). Boxes scale with image size."""

    # (label, coco_id, relative xyxy, confidence)
    _PROTOS = [
        ("BUS (BMTC/KSRTC)", 5, (0.10, 0.16, 0.62, 0.93), 0.94),
        ("CAR", 2, (0.66, 0.55, 0.83, 0.71), 0.82),
        ("CAR", 2, (0.01, 0.55, 0.13, 0.67), 0.75),
        ("MOTOR CYCLE", 3, (0.49, 0.66, 0.58, 0.82), 0.63),
    ]

    def __init__(self, seed: int = 42, jitter: float = 0.004):
        self.seed, self.jitter = seed, jitter

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        rng = np.random.default_rng(self.seed)
        dets = []
        for label, cid, rel, conf in self._PROTOS:
            box = np.array(rel) + rng.normal(0, self.jitter, 4)
            xyxy = np.clip(box, 0, 1) * np.array([w, h, w, h])
            dets.append(Detection(cid, label, conf, tuple(map(float, xyxy))))
        return dets


# COCO 90-class label ids (TF/SSD convention) -> CurbIQ vehicle_type.
SSD_COCO_VEHICLES: dict[int, str] = {
    2: "BICYCLE", 3: "CAR", 4: "MOTOR CYCLE", 6: "BUS (BMTC/KSRTC)", 8: "LORRY/GOODS VEHICLE",
}


class SsdMobileNetDetector:
    """Real COCO detection via ONNX SSD-MobileNet v1 (onnxruntime, CPU).

    Input is raw uint8 NHWC (the model embeds its own preprocessing); outputs are
    normalized [ymin,xmin,ymax,xmax] boxes + COCO class ids + scores.
    """

    def __init__(self, model_path: str, conf: float = 0.30, vehicles_only: bool = True):
        import onnxruntime as ort                       # lazy: optional dependency
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.conf, self.vehicles_only = conf, vehicles_only

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        out = self.sess.run(None, {self.input_name: image[None].astype(np.uint8)})
        named = dict(zip([o.name for o in self.sess.get_outputs()], out))

        def pick(key):
            return next(named[n] for n in named if key in n)

        boxes = pick("boxes")[0]
        classes = pick("classes")[0].astype(int)
        scores = pick("scores")[0]
        dets = []
        for b, c, s in zip(boxes, classes, scores):
            if s < self.conf or (self.vehicles_only and c not in SSD_COCO_VEHICLES):
                continue
            ymin, xmin, ymax, xmax = b
            dets.append(Detection(int(c), SSD_COCO_VEHICLES.get(int(c), str(c)), float(s),
                                  (float(xmin * w), float(ymin * h), float(xmax * w), float(ymax * h))))
        return dets


def load_detector(model_path: str | None = None, backend: str = "auto", conf: float = 0.30, **kw):
    """Return a real ONNX detector if a model + onnxruntime are available, else simulation.

    backend: 'auto' (by filename), 'ssd' (SSD-MobileNet), or 'yolov8'.
    """
    if model_path and Path(model_path).exists():
        b = backend
        if b == "auto":
            b = "ssd" if "ssd" in Path(model_path).name.lower() else "yolov8"
        try:
            if b == "ssd":
                return SsdMobileNetDetector(model_path, conf=conf)
            return OnnxYoloDetector(model_path, conf=conf, **kw)
        except Exception as e:           # missing onnxruntime / bad model -> fall back
            print(f"[cv] ONNX detector unavailable ({e}); using SimulationDetector")
    return SimulationDetector()
