#!/usr/bin/env bash
# Fetch the SSD-MobileNet v1 COCO ONNX model (~28 MB) used for real
# camera-ingestion inference. (Any YOLOv8 .onnx works too — point
# CURBIQ_YOLO_ONNX at it instead.)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/models/ssd_mobilenet.onnx"
URL="https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
mkdir -p "$ROOT/models"
if [[ -f "$OUT" ]]; then
  echo "model already present: $OUT"
  exit 0
fi
echo "downloading SSD-MobileNet COCO ONNX (~28 MB) ..."
curl -fSL "$URL" -o "$OUT" && echo "saved -> $OUT"
echo "(real inference also needs onnxruntime:  .venv/bin/pip install onnxruntime)"
