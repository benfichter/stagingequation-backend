from __future__ import annotations

import base64
import os
import threading
import time
from typing import Any

import httpx
from fastapi import HTTPException, status

_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_DEVICE = None


def is_moge_enabled() -> bool:
    return os.getenv("MOGE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def _resolve_device():
    import torch

    preferred = os.getenv("MOGE_DEVICE", "cuda").lower()
    if preferred == "cuda" and not torch.cuda.is_available():
        preferred = "cpu"
    return torch.device(preferred)


def _load_model(device: torch.device):
    import torch

    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise RuntimeError("MoGe is not installed. Install the 'moge' package.") from exc

    model_id = os.getenv("MOGE_MODEL_ID", "Ruicheng/moge-2-vitl-normal")
    model = MoGeModel.from_pretrained(model_id).to(device)
    model.eval()
    return model


def _get_model():
    global _MODEL
    global _MODEL_DEVICE

    if _MODEL is not None and _MODEL_DEVICE is not None:
        return _MODEL, _MODEL_DEVICE

    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_DEVICE is not None:
            return _MODEL, _MODEL_DEVICE
        device = _resolve_device()
        _MODEL = _load_model(device)
        _MODEL_DEVICE = device
        return _MODEL, _MODEL_DEVICE


def _decode_image(image_bytes: bytes) -> np.ndarray:
    import cv2
    import numpy as np

    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid image data.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _estimate_height(
    points: np.ndarray, mask: np.ndarray, normal: np.ndarray | None
) -> tuple[float, float, float] | None:
    import numpy as np

    valid_mask = mask > 0
    valid_points = points[valid_mask]
    if valid_points.size == 0:
        return None

    y_coords = valid_points[:, 1]
    floor_y = float(np.percentile(y_coords, 95))
    ceiling_y = float(np.percentile(y_coords, 5))

    if normal is not None:
        normal_y = normal[:, :, 1]
        floor_mask = (normal_y < -0.7) & valid_mask
        ceiling_mask = (normal_y > 0.7) & valid_mask
        if floor_mask.any():
            floor_y = float(np.mean(points[floor_mask][:, 1]))
        if ceiling_mask.any():
            ceiling_y = float(np.mean(points[ceiling_mask][:, 1]))

    height = abs(floor_y - ceiling_y)
    return height, floor_y, ceiling_y


def _floor_points(points: np.ndarray, mask: np.ndarray, normal: np.ndarray | None) -> np.ndarray | None:
    import numpy as np

    valid_mask = mask > 0
    valid_points = points[valid_mask]
    if valid_points.size == 0:
        return None

    if normal is not None:
        normal_y = normal[:, :, 1]
        floor_mask = (normal_y < -0.7) & valid_mask
        if floor_mask.any():
            return points[floor_mask]

    y_coords = valid_points[:, 1]
    threshold = np.percentile(y_coords, 80)
    floor_points = valid_points[valid_points[:, 1] > threshold]
    return floor_points if floor_points.size else valid_points


def _infer_remote(image_bytes: bytes, calibration_height_m: float | None) -> dict[str, Any]:
    remote = os.getenv("MOGE_REMOTE_URL")
    if not remote:
        raise RuntimeError("MOGE_REMOTE_URL is not configured")

    headers = {}
    api_key = os.getenv("MOGE_REMOTE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {}
    if calibration_height_m:
        data["calibration_height_m"] = str(calibration_height_m)

    timeout = float(os.getenv("MOGE_REMOTE_TIMEOUT", "120"))
    url = remote.rstrip("/") + "/infer"
    response = httpx.post(
        url,
        data=data,
        files={"file": ("room.jpg", image_bytes, "image/jpeg")},
        headers=headers,
        timeout=timeout,
    )

    if response.status_code == status.HTTP_401_UNAUTHORIZED:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="MoGe API key invalid")

    if response.status_code >= 400:
        raise RuntimeError(f"MoGe service error: {response.status_code} {response.text}")

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("MoGe service returned an unexpected response.")
    return payload


def _runpod_config() -> tuple[str | None, str | None]:
    return os.getenv("MOGE_RUNPOD_ENDPOINT_ID"), os.getenv("MOGE_RUNPOD_API_KEY")


def _runpod_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _runpod_timeout() -> float:
    return float(os.getenv("MOGE_RUNPOD_TIMEOUT", "180"))


def _runpod_poll_interval() -> float:
    return float(os.getenv("MOGE_RUNPOD_POLL_INTERVAL", "2"))


def _runpod_run(payload: dict[str, Any]) -> str:
    endpoint_id, api_key = _runpod_config()
    if not endpoint_id or not api_key:
        raise RuntimeError("MOGE_RUNPOD_ENDPOINT_ID or MOGE_RUNPOD_API_KEY is not configured")

    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    response = httpx.post(url, json=payload, headers=_runpod_headers(api_key), timeout=_runpod_timeout())
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod run error: {response.status_code} {response.text}")

    data = response.json()
    job_id = data.get("id")
    if not job_id:
        raise RuntimeError("RunPod did not return a job id.")
    return job_id


def _runpod_poll(job_id: str) -> dict[str, Any]:
    endpoint_id, api_key = _runpod_config()
    if not endpoint_id or not api_key:
        raise RuntimeError("MOGE_RUNPOD_ENDPOINT_ID or MOGE_RUNPOD_API_KEY is not configured")

    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    deadline = time.monotonic() + _runpod_timeout()
    interval = _runpod_poll_interval()

    while time.monotonic() < deadline:
        response = httpx.get(url, headers=_runpod_headers(api_key), timeout=_runpod_timeout())
        if response.status_code >= 400:
            raise RuntimeError(f"RunPod status error: {response.status_code} {response.text}")

        data = response.json()
        status_value = str(data.get("status") or "").upper()
        if status_value in {"COMPLETED", "SUCCESS", "SUCCEEDED"}:
            output = data.get("output")
            if isinstance(output, dict):
                return output
            raise RuntimeError("RunPod returned an unexpected output payload.")
        if status_value in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RuntimeError(f"RunPod job failed: {data.get('error') or data}")

        time.sleep(interval)

    raise RuntimeError("RunPod job timed out.")


def _infer_runpod(image_bytes: bytes, calibration_height_m: float | None) -> dict[str, Any]:
    payload = {
        "input": {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        }
    }
    if calibration_height_m:
        payload["input"]["calibration_height_m"] = calibration_height_m

    job_id = _runpod_run(payload)
    return _runpod_poll(job_id)


def request_moge_warmup() -> dict[str, Any]:
    endpoint_id, api_key = _runpod_config()
    if not endpoint_id or not api_key:
        return {"status": "not_configured"}

    payload = {"input": {"warm": True}}
    job_id = _runpod_run(payload)
    return {"status": "warming", "job_id": job_id}


def infer_room_dimensions(
    image_bytes: bytes, calibration_height_m: float | None = None
) -> dict[str, Any] | None:
    if not is_moge_enabled():
        return None

    if os.getenv("MOGE_RUNPOD_ENDPOINT_ID"):
        return _infer_runpod(image_bytes, calibration_height_m)

    if os.getenv("MOGE_REMOTE_URL"):
        return _infer_remote(image_bytes, calibration_height_m)

    import numpy as np
    import torch

    model, device = _get_model()
    image = _decode_image(image_bytes)
    tensor = torch.from_numpy(image).to(device).float() / 255.0
    tensor = tensor.permute(2, 0, 1)

    with torch.no_grad():
        output = model.infer(tensor)

    points = output["points"].detach().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy()
    normal = output.get("normal")
    normal_map = normal.detach().cpu().numpy() if normal is not None else None

    height_data = _estimate_height(points, mask, normal_map)
    if height_data is None:
        return None
    height, _, _ = height_data

    floor_points = _floor_points(points, mask, normal_map)
    if floor_points is None:
        return None

    x_coords = floor_points[:, 0]
    z_coords = floor_points[:, 2]
    width = float(np.max(x_coords) - np.min(x_coords))
    depth = float(np.max(z_coords) - np.min(z_coords))
    area = width * depth

    calibration_factor = None
    if calibration_height_m and height > 0:
        calibration_factor = float(calibration_height_m / height)
        width *= calibration_factor
        depth *= calibration_factor
        height *= calibration_factor
        area *= calibration_factor ** 2

    return {
        "width": width,
        "depth": depth,
        "height": height,
        "area": area,
        "unit": "m",
        "calibration_factor": calibration_factor,
    }
