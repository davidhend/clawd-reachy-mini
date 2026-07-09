"""Camera + face-tracking service for Reachy Mini — no SDK, no WebRTC.

Owns the camera continuously (live MJPG stream via GStreamer) and provides:
  - on-demand JPEG stills, served instantly from the latest frame
  - face tracking: a YuNet detector + ~8Hz control loop that steers the head
    via the daemon's /api/move/set_target so Reachy looks at the nearest face

This works because the daemon's media devices are released (the voice service
holds them released for mic access); this service releases them itself if
needed. The tracker defers to the daemon's move queue: while a recorded move /
goto is running (emotions, gestures, wake/sleep animations), it pauses and
re-syncs to the actual head pose afterwards.

Endpoints:
  GET  /capture?width=&height=   -> image/jpeg (native 1920x1080 by default)
  POST /track/start?target=face  -> enable tracking; target is "face" or a COCO
                                    label ("person", "cat", "dog", "cup", ...)
  POST /track/stop               -> disable tracking (head stays where it is)
  GET  /track/status             -> tracking state + last target info
  GET  /detect                   -> one-shot object detection (NanoDet, 80 COCO
                                    classes) on the latest frame, as JSON
  GET  /health

Environment:
  REACHY_CAMERA_PORT    default 8089
  REACHY_VIDEO_DEVICE   default /dev/video0
  REACHY_DAEMON_URL     default http://127.0.0.1:8000
  REACHY_TRACK_AUTOSTART default "true"
  REACHY_ROLL_TRIM      radians added to commanded roll (default 0.0; this
                        unit is level around -0.12, see project notes)
  REACHY_YUNET_MODEL    override path to the YuNet .onnx
  REACHY_NANODET_MODEL  override path to the NanoDet .onnx
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Response

logger = logging.getLogger(__name__)

VIDEO_DEVICE = os.environ.get("REACHY_VIDEO_DEVICE", "/dev/video0")
DAEMON_URL = os.environ.get("REACHY_DAEMON_URL", "http://127.0.0.1:8000").rstrip("/")
STREAM_WIDTH, STREAM_HEIGHT = 1920, 1080
# Focal length in pixels at stream resolution, from the daemon's camera
# calibration (K fx=2001.8 at 3840 wide -> ~1001 at 1920). Converts pixel
# offsets to angles; ~88 deg horizontal FOV.
FOCAL_PX = 1001.0

DETECT_WIDTH = 640  # detection runs on a downscale; ~10ms/frame on the i5
TRACK_HZ = 8.0
DEADBAND_RAD = math.radians(1.5)
STEP_GAIN = 0.35  # fraction of the angular error applied per tick
MAX_STEP_RAD = math.radians(4.0)  # per-tick slew limit (~32 deg/s at 8Hz)
MAX_YAW = math.radians(45.0)
MAX_PITCH = math.radians(25.0)
FACE_LOST_S = 3.0  # keep last heading this long before easing back to center
CENTER_STEP_RAD = math.radians(1.5)  # easing speed back to center
MOVE_CHECK_PERIOD_S = 0.5  # how often to poll /api/move/running

ROLL_TRIM = float(os.environ.get("REACHY_ROLL_TRIM", "0.0"))

DEFAULT_MODEL = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_PATH = Path(os.environ.get("REACHY_YUNET_MODEL", str(DEFAULT_MODEL)))
DEFAULT_NANODET = Path(__file__).parent / "models" / "object_detection_nanodet_2022nov.onnx"
NANODET_PATH = Path(os.environ.get("REACHY_NANODET_MODEL", str(DEFAULT_NANODET)))
NANODET_INPUT = 416  # model's fixed input size

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class NanoDetDetector:
    """COCO object detection via cv2.dnn (adapted from opencv_zoo, Apache-2.0).

    Thread-safe: a lock serializes inference so the tracker loop and the
    /detect endpoint can share one instance.
    """

    def __init__(self, model_path: Path, prob_threshold: float = 0.35, iou_threshold: float = 0.6):
        import cv2
        import numpy as np

        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.strides = (8, 16, 32, 64)
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet(str(model_path))
        self._lock = threading.Lock()

        self.anchors_mlvl = []
        for stride in self.strides:
            feat = NANODET_INPUT // stride
            shift = np.arange(0, feat) * stride
            xv, yv = np.meshgrid(shift, shift)
            cx = xv.flatten() + 0.5 * (stride - 1)
            cy = yv.flatten() + 0.5 * (stride - 1)
            self.anchors_mlvl.append(np.column_stack((cx, cy)))

    def detect(self, img_bgr) -> list[dict]:
        """Detect objects in a stream-resolution BGR image.

        Returns [{"label", "confidence", "cx", "cy", "w", "h"}] in original
        image pixels.
        """
        import cv2
        import numpy as np

        h, w = img_bgr.shape[:2]
        inp = cv2.resize(img_bgr, (NANODET_INPUT, NANODET_INPUT)).astype(np.float32)
        blob = cv2.dnn.blobFromImage((inp - self.mean) / self.std)
        with self._lock:
            self.net.setInput(blob)
            outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        bboxes_mlvl, scores_mlvl = [], []
        for stride, cls_score, bbox_pred, anchors in zip(
            self.strides, outs[::2], outs[1::2], self.anchors_mlvl
        ):
            if cls_score.ndim == 3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim == 3:
                bbox_pred = bbox_pred.squeeze(axis=0)
            x_exp = np.exp(bbox_pred.reshape(-1, self.reg_max + 1))
            dist = np.dot(x_exp / x_exp.sum(axis=1, keepdims=True), self.project).reshape(-1, 4)
            dist *= stride
            x1 = np.clip(anchors[:, 0] - dist[:, 0], 0, NANODET_INPUT)
            y1 = np.clip(anchors[:, 1] - dist[:, 1], 0, NANODET_INPUT)
            x2 = np.clip(anchors[:, 0] + dist[:, 2], 0, NANODET_INPUT)
            y2 = np.clip(anchors[:, 1] + dist[:, 3], 0, NANODET_INPUT)
            bboxes_mlvl.append(np.column_stack([x1, y1, x2, y2]))
            scores_mlvl.append(cls_score)

        bboxes = np.concatenate(bboxes_mlvl, axis=0)
        scores = np.concatenate(scores_mlvl, axis=0)
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        boxes_wh = bboxes.copy()
        boxes_wh[:, 2:4] -= boxes_wh[:, 0:2]
        indices = cv2.dnn.NMSBoxes(
            boxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold
        )
        results = []
        sx, sy = w / NANODET_INPUT, h / NANODET_INPUT
        for i in np.array(indices).flatten():
            x1, y1, x2, y2 = bboxes[i]
            results.append({
                "label": COCO_LABELS[int(class_ids[i])],
                "confidence": round(float(confidences[i]), 2),
                "cx": (x1 + x2) / 2 * sx,
                "cy": (y1 + y2) / 2 * sy,
                "w": (x2 - x1) * sx,
                "h": (y2 - y1) * sy,
            })
        return results


def compute_step(err_rad: float) -> float:
    """Per-tick correction for one axis: proportional, slew-limited, deadbanded."""
    if abs(err_rad) < DEADBAND_RAD:
        return 0.0
    return max(-MAX_STEP_RAD, min(MAX_STEP_RAD, STEP_GAIN * err_rad))


def pixel_to_angles(cx: float, cy: float) -> tuple[float, float]:
    """Angular error of a pixel from frame center.

    Returns (yaw_err, pitch_err) in radians, signed so that ADDING them to the
    current head target turns toward the pixel (daemon convention: positive
    yaw = left, positive pitch = down).
    """
    dx = cx - STREAM_WIDTH / 2.0
    dy = cy - STREAM_HEIGHT / 2.0
    yaw_err = -math.atan2(dx, FOCAL_PX)  # pixel right of center -> turn right (negative yaw)
    pitch_err = math.atan2(dy, FOCAL_PX)  # pixel below center -> pitch down (positive)
    return yaw_err, pitch_err


class CameraStream:
    """Continuously pulls MJPG frames, keeping only the latest."""

    def __init__(self, device: str):
        self.device = device
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="camera-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._seq

    def wait_for_frame(self, timeout_s: float = 5.0) -> bytes | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            jpeg, _ = self.latest()
            if jpeg:
                return jpeg
            time.sleep(0.05)
        return None

    def _run(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)

        desc = (
            f"v4l2src device={self.device} "
            f"! image/jpeg,width={STREAM_WIDTH},height={STREAM_HEIGHT} "
            f"! appsink name=sink drop=true max-buffers=1 sync=false"
        )
        while not self._stop.is_set():
            pipeline = None
            try:
                pipeline = Gst.parse_launch(desc)
                sink = pipeline.get_by_name("sink")
                if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("pipeline refused to start (device busy?)")
                logger.info(f"🎥 Camera stream up ({self.device}, {STREAM_WIDTH}x{STREAM_HEIGHT} MJPG)")
                while not self._stop.is_set():
                    sample = sink.emit("try-pull-sample", Gst.SECOND)
                    if sample is None:
                        if sink.get_property("eos"):
                            raise RuntimeError("unexpected EOS from live source")
                        continue
                    buf = sample.get_buffer()
                    ok, mapinfo = buf.map(Gst.MapFlags.READ)
                    if ok:
                        data = bytes(mapinfo.data)
                        buf.unmap(mapinfo)
                        with self._lock:
                            self._jpeg = data
                            self._seq += 1
            except Exception as e:
                logger.error(f"Camera stream died: {e} — retrying in 3s")
                self._stop.wait(3.0)
            finally:
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)


class HeadTracker:
    """Detects the chosen target (face or COCO object) and steers the head to it."""

    def __init__(self, stream: CameraStream, autostart: bool):
        self.stream = stream
        self.enabled = threading.Event()
        if autostart:
            self.enabled.set()
        self.mode = "face"  # "face" or "object"
        self.target_label: str | None = None  # COCO label when mode == "object"
        self.last_face_ts: float = 0.0
        self.last_face_center: tuple[float, float] | None = None
        self._yaw = 0.0
        self._pitch = 0.0
        self._sending = False
        self._needs_resync = True
        self._stop = threading.Event()
        self._detector = None
        self._nanodet: NanoDetDetector | None = None

    def set_target(self, target: str) -> None:
        """Switch what to track: "face" or a COCO label like "person"/"cat"."""
        target = target.strip().lower()
        if target == "face":
            self.mode = "face"
            self.target_label = None
        elif target in COCO_LABELS:
            self.mode = "object"
            self.target_label = target
        else:
            raise ValueError(f"unknown target {target!r} — use 'face' or a COCO label")
        self.last_face_ts = 0.0

    def get_nanodet(self) -> NanoDetDetector:
        if self._nanodet is None:
            if not NANODET_PATH.exists():
                raise RuntimeError(f"NanoDet model not found: {NANODET_PATH}")
            self._nanodet = NanoDetDetector(NANODET_PATH)
        return self._nanodet

    def decode_latest(self):
        """Latest stream frame as a BGR array, or None."""
        import cv2
        import numpy as np

        jpeg, _ = self.stream.latest()
        if jpeg is None:
            return None
        return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    def start(self) -> None:
        threading.Thread(target=self._run, name="face-tracker", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        now = time.monotonic()
        visible = (now - self.last_face_ts) < 1.5
        return {
            "tracking": self.enabled.is_set(),
            "target": "face" if self.mode == "face" else self.target_label,
            "target_visible": visible,
            "last_seen_ago_s": round(now - self.last_face_ts, 1) if self.last_face_ts else None,
            "yaw_deg": round(math.degrees(self._yaw), 1),
            "pitch_deg": round(math.degrees(self._pitch), 1),
        }

    def _load_detector(self):
        if self._detector is None:
            import cv2

            if not MODEL_PATH.exists():
                raise RuntimeError(f"YuNet model not found: {MODEL_PATH}")
            scale_h = int(STREAM_HEIGHT * DETECT_WIDTH / STREAM_WIDTH)
            self._detector = cv2.FaceDetectorYN.create(
                str(MODEL_PATH), "", (DETECT_WIDTH, scale_h), score_threshold=0.6
            )
        return self._detector

    def _detect_largest(self, jpeg: bytes) -> tuple[float, float] | None:
        """Center of the largest matching target in stream pixels, or None."""
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

        if self.mode == "object":
            hits = [
                d for d in self.get_nanodet().detect(img) if d["label"] == self.target_label
            ]
            if not hits:
                return None
            best = max(hits, key=lambda d: d["w"] * d["h"])
            return best["cx"], best["cy"]

        scale = DETECT_WIDTH / img.shape[1]
        small = cv2.resize(img, (DETECT_WIDTH, int(img.shape[0] * scale)))
        detector = self._load_detector()
        _, faces = detector.detect(small)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[2] * f[3])
        cx = (best[0] + best[2] / 2.0) / scale
        cy = (best[1] + best[3] / 2.0) / scale
        return cx, cy

    def _resync_pose(self, client: httpx.Client) -> None:
        """Adopt the actual head pose so we never yank from a stale target."""
        try:
            r = client.get("/api/state/present_head_pose")
            r.raise_for_status()
            pose = r.json()
            self._yaw = float(pose.get("yaw", 0.0))
            self._pitch = float(pose.get("pitch", 0.0))
            self._needs_resync = False
            logger.debug(f"Resynced to present pose yaw={self._yaw:.2f} pitch={self._pitch:.2f}")
        except Exception as e:
            logger.debug(f"present_head_pose failed: {e}")

    def _moves_running(self, client: httpx.Client) -> bool:
        try:
            r = client.get("/api/move/running")
            data = r.json()
            items = data if isinstance(data, list) else [data]
            return any(items)
        except Exception:
            return False

    def _send_target(self, client: httpx.Client) -> None:
        body = {
            "target_head_pose": {
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": ROLL_TRIM,
                "pitch": self._pitch,
                "yaw": self._yaw,
            }
        }
        try:
            client.post("/api/move/set_target", json=body)
            self._sending = True
        except Exception as e:
            logger.debug(f"set_target failed: {e}")

    def _run(self) -> None:
        period = 1.0 / TRACK_HZ
        last_seq = -1
        last_move_check = 0.0
        moves_busy = False
        client = httpx.Client(base_url=DAEMON_URL, timeout=3.0)

        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if not self.enabled.is_set():
                    self._sending = False
                    self._needs_resync = True
                    self._stop.wait(0.2)
                    continue

                # Defer to the daemon's move queue (emotions, gestures, sleep).
                if t0 - last_move_check > MOVE_CHECK_PERIOD_S:
                    was_busy = moves_busy
                    moves_busy = self._moves_running(client)
                    last_move_check = t0
                    if was_busy and not moves_busy:
                        self._needs_resync = True
                if moves_busy:
                    self._sending = False
                    self._stop.wait(period)
                    continue

                if self._needs_resync:
                    self._resync_pose(client)

                jpeg, seq = self.stream.latest()
                if jpeg is None or seq == last_seq:
                    self._stop.wait(0.02)
                    continue
                last_seq = seq

                center = self._detect_largest(jpeg)
                now = time.monotonic()
                if center is not None:
                    self.last_face_ts = now
                    self.last_face_center = center
                    yaw_err, pitch_err = pixel_to_angles(*center)
                    self._yaw = max(-MAX_YAW, min(MAX_YAW, self._yaw + compute_step(yaw_err)))
                    self._pitch = max(-MAX_PITCH, min(MAX_PITCH, self._pitch + compute_step(pitch_err)))
                    self._send_target(client)
                elif self._sending and (now - self.last_face_ts) > FACE_LOST_S:
                    # Ease back to center, then go quiet so we don't fight
                    # other controllers while idle.
                    if abs(self._yaw) < math.radians(1.0) and abs(self._pitch) < math.radians(1.0):
                        self._sending = False
                    else:
                        self._yaw -= max(-CENTER_STEP_RAD, min(CENTER_STEP_RAD, self._yaw))
                        self._pitch -= max(-CENTER_STEP_RAD, min(CENTER_STEP_RAD, self._pitch))
                        self._send_target(client)
            except Exception:
                logger.exception("Tracker tick failed")
                self._stop.wait(1.0)

            elapsed = time.monotonic() - t0
            if elapsed < period:
                self._stop.wait(period - elapsed)


app = FastAPI(title="reachy-camera")
_stream = CameraStream(VIDEO_DEVICE)
_tracker = HeadTracker(
    _stream, autostart=os.environ.get("REACHY_TRACK_AUTOSTART", "true").lower() in ("1", "true", "yes")
)


async def _ensure_media_released() -> None:
    try:
        async with httpx.AsyncClient(base_url=DAEMON_URL, timeout=10.0) as client:
            r = await client.get("/api/media/status")
            r.raise_for_status()
            if not r.json().get("released"):
                logger.info("Daemon holds media — releasing for camera access")
                (await client.post("/api/media/release")).raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"Daemon unreachable while checking media status: {e}")


@app.on_event("startup")
async def _startup() -> None:
    await _ensure_media_released()
    _stream.start()
    _tracker.start()


@app.get("/health")
async def health() -> dict:
    jpeg, seq = _stream.latest()
    return {"status": "ok", "device": VIDEO_DEVICE, "frames": seq, "streaming": jpeg is not None}


@app.get("/capture")
async def capture(width: int = STREAM_WIDTH, height: int = STREAM_HEIGHT) -> Response:
    jpeg = _stream.wait_for_frame(timeout_s=5.0)
    if jpeg is None:
        raise HTTPException(status_code=503, detail="no frame from camera stream")
    if (width, height) != (STREAM_WIDTH, STREAM_HEIGHT):
        try:
            import cv2
            import numpy as np

            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (max(160, min(width, 3840)), max(120, min(height, 2592))))
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpeg = enc.tobytes()
        except Exception as e:
            logger.warning(f"Resize failed, returning native frame: {e}")
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/track/start")
async def track_start(target: str = "face") -> dict:
    try:
        _tracker.set_target(target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{e}; COCO labels: {', '.join(COCO_LABELS)}")
    _tracker.enabled.set()
    logger.info(f"🎯 Tracking ENABLED (target: {target})")
    return _tracker.status()


@app.post("/track/stop")
async def track_stop() -> dict:
    _tracker.enabled.clear()
    logger.info("😴 Tracking DISABLED")
    return _tracker.status()


@app.get("/track/status")
async def track_status() -> dict:
    return _tracker.status()


@app.get("/detect")
async def detect() -> dict:
    """One-shot COCO object detection on the latest frame."""
    img = await asyncio.to_thread(_tracker.decode_latest)
    if img is None:
        raise HTTPException(status_code=503, detail="no frame from camera stream")
    try:
        detections = await asyncio.to_thread(_tracker.get_nanodet().detect, img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "detections": [
            {**d, "cx": round(d["cx"]), "cy": round(d["cy"]),
             "w": round(d["w"]), "h": round(d["h"])}
            for d in detections
        ],
        "frame_size": [STREAM_WIDTH, STREAM_HEIGHT],
    }


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    port = int(os.environ.get("REACHY_CAMERA_PORT", "8089"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
