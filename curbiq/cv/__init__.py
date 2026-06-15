"""Live camera/CCTV ingestion: detect vehicles, geofence no-parking zones,
track dwell time, and emit violation records in the dataset schema so live
detections flow straight into the same hotspot / congestion / priority analytics.

The detector backend is pluggable:
  * ``OnnxYoloDetector`` — real YOLOv8 inference via onnxruntime (production);
  * ``SimulationDetector`` — deterministic stand-in so the full pipeline is
    runnable/testable without GPU, torch, or model weights.
"""
from curbiq.cv.detector import (Detection, OnnxYoloDetector, SimulationDetector,
                                SsdMobileNetDetector, load_detector)
from curbiq.cv.geofence import CameraConfig, Homography, NoParkingZone
from curbiq.cv.pipeline import (ViolationEvent, process_image, process_video,
                                to_violation_records)
from curbiq.cv.tracker import IouTracker, Track

__all__ = [
    "Detection", "OnnxYoloDetector", "SsdMobileNetDetector", "SimulationDetector", "load_detector",
    "NoParkingZone", "Homography", "CameraConfig",
    "IouTracker", "Track",
    "ViolationEvent", "process_image", "process_video", "to_violation_records",
]
