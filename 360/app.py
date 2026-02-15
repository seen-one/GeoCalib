from __future__ import annotations

import io
import logging
import random
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import py360convert
import torch
from flask import Flask, jsonify, render_template, request, send_file
from geocalib import GeoCalib
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)

# Tell Flask it is behind a proxy.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = GeoCalib().to(DEVICE)
MODEL_LOCK = Lock()
PANORAMAX_IDS_PATH = Path(__file__).resolve().parent / "random_panoramax_bike_ids.txt"
MAX_SAMPLE_COUNT = 144


class ImageCache:
    def __init__(self, max_size=50):
        self.cache = OrderedDict()
        self.max_size = max_size

    def set(self, key, data, content_type="image/jpeg"):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = {"data": data, "content_type": content_type}
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None


image_cache = ImageCache(max_size=300)
panoramax_ids = []


def error_response(message, status_code=400):
    return jsonify({"status": "error", "error": message}), status_code


def load_panoramax_ids():
    global panoramax_ids
    if not panoramax_ids:
        if PANORAMAX_IDS_PATH.exists():
            with PANORAMAX_IDS_PATH.open("r", encoding="utf-8") as f:
                panoramax_ids = [line.strip() for line in f if line.strip()]
        else:
            logger.warning("Panoramax ID file not found at %s", PANORAMAX_IDS_PATH)
    return panoramax_ids


def download_image_bytes(url):
    req = Request(url, headers={"User-Agent": "GeoCalib-360/1.0"})
    with urlopen(req, timeout=30) as response:
        data = response.read()
        if not data:
            raise ValueError("Downloaded response was empty.")
        return data


def numpy_rgb_to_tensor(image_rgb):
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape [H, W, 3].")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    image_rgb = np.ascontiguousarray(image_rgb)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0)
    return tensor.to(DEVICE)


def estimate_roll_deg(sample_bgr):
    sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
    img = numpy_rgb_to_tensor(sample_rgb)

    # Guard model inference for concurrent Flask requests.
    with MODEL_LOCK:
        with torch.no_grad():
            result = MODEL.calibrate(img, camera_model="pinhole")

    roll_rad = result["gravity"].rp[..., 0].detach().cpu().reshape(-1)[0]
    return float(torch.rad2deg(roll_rad).item())


@app.route("/")
@app.route("/predict_360")
def predict_360_page():
    return render_template("index.html")


@app.route("/viewer_360")
def viewer_360():
    return render_template("viewer_360.html")


@app.route("/api/random_id")
def get_random_id():
    ids = load_panoramax_ids()
    if ids:
        return jsonify({"id": random.choice(ids)})
    return error_response("No IDs found", 404)


@app.route("/mem_image/<image_id>")
def get_mem_image(image_id):
    item = image_cache.get(image_id)
    if item:
        return send_file(io.BytesIO(item["data"]), mimetype=item["content_type"])
    return "Image not found", 404


@app.route("/api/predict_360", methods=["POST"])
def api_predict_360():
    if "file" not in request.files and "url" not in request.form:
        return error_response("No file or URL part")

    image_bytes = None
    file = request.files.get("file")
    url = (request.form.get("url") or "").strip()

    if file and file.filename:
        image_bytes = file.read()
    elif url:
        try:
            image_bytes = download_image_bytes(url)
        except (ValueError, HTTPError, URLError, TimeoutError) as e:
            logger.error("Error downloading image from URL: %s", e)
            return error_response(f"Failed to download image: {e}")
        except Exception as e:
            logger.error("Unexpected URL download error: %s", e)
            return error_response(f"Failed to download image: {e}")

    if not image_bytes:
        return error_response("No selected file or valid URL")

    try:
        fov_deg = float(request.form.get("fov", 60))
        inlier_threshold_deg = float(request.form.get("inlier_threshold_deg", 2.0))
        sample_count = int(request.form.get("sample_count", 36))
    except (TypeError, ValueError):
        return error_response("Invalid numeric request parameter")

    if sample_count < 4:
        return error_response("sample_count should be >= 4")
    if sample_count > MAX_SAMPLE_COUNT:
        return error_response(f"sample_count should be <= {MAX_SAMPLE_COUNT}")
    if not (1.0 <= fov_deg <= 179.0):
        return error_response("fov should be between 1 and 179 degrees")
    if not (0.01 <= inlier_threshold_deg <= 90.0):
        return error_response("inlier_threshold_deg should be between 0.01 and 90")

    main_image_id = str(uuid.uuid4())
    image_cache.set(main_image_id, image_bytes)

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return error_response("Failed to decode image")

        sample_yaws = np.linspace(0, 360, sample_count, endpoint=False, dtype=np.float32)

        def extract_sample(idx, yaw_deg):
            sample_img = py360convert.e2p(
                img_bgr,
                fov_deg=fov_deg,
                u_deg=float(yaw_deg),
                v_deg=0.0,
                in_rot_deg=0.0,
                out_hw=(400, 400),
                mode="bilinear",
            )
            return idx, sample_img

        max_workers = min(8, sample_count)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(extract_sample, i, y) for i, y in enumerate(sample_yaws)]
            sample_images = [None] * sample_count
            for future in futures:
                idx, sample_img = future.result()
                sample_images[idx] = sample_img

        predicted_rolls = np.array(
            [estimate_roll_deg(sample_img) for sample_img in sample_images], dtype=np.float32
        )
        yaw_rads = np.deg2rad(sample_yaws)
        roll_rads = np.deg2rad(predicted_rolls)

        def normalize_rad(rad):
            return (rad + np.pi) % (2 * np.pi) - np.pi

        def angular_diff_deg(a_deg, b_deg):
            return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)

        def model_roll_deg(alpha_rad, beta_rad, yaw_rad_values):
            return np.rad2deg(np.arctan(np.tan(beta_rad) * np.cos(yaw_rad_values - alpha_rad)))

        def evaluate_hypothesis(alpha_rad, beta_rad):
            modeled = model_roll_deg(alpha_rad, beta_rad, yaw_rads)
            errors = np.array(
                [angular_diff_deg(float(p), float(m)) for p, m in zip(predicted_rolls, modeled)],
                dtype=np.float32,
            )
            inliers = errors <= inlier_threshold_deg
            inlier_count = int(np.sum(inliers))
            mae = float(np.mean(errors[inliers])) if inlier_count > 0 else float("inf")
            rmse = (
                float(np.sqrt(np.mean(np.square(errors[inliers]))))
                if inlier_count > 0
                else float("inf")
            )
            return {
                "alpha_rad": float(normalize_rad(alpha_rad)),
                "beta_rad": float(beta_rad),
                "modeled_rolls": modeled,
                "errors": errors,
                "inliers": inliers,
                "inlier_count": inlier_count,
                "mae": mae,
                "rmse": rmse,
            }

        best = None
        tiny = 1e-6
        tan_rolls = np.tan(roll_rads)
        for i in range(sample_count):
            for j in range(i + 1, sample_count):
                ai = tan_rolls[i]
                aj = tan_rolls[j]
                theta_i = yaw_rads[i]
                theta_j = yaw_rads[j]

                p = ai * np.cos(theta_j) - aj * np.cos(theta_i)
                q = ai * np.sin(theta_j) - aj * np.sin(theta_i)
                if abs(p) < tiny and abs(q) < tiny:
                    continue

                alpha_candidate_1 = np.arctan2(-p, q)
                for alpha_candidate in (alpha_candidate_1, alpha_candidate_1 + np.pi):
                    cos_term = np.cos(theta_i - alpha_candidate)
                    if abs(cos_term) < tiny:
                        continue
                    tan_beta = ai / cos_term
                    beta_candidate = np.arctan(tan_beta)

                    hypothesis = evaluate_hypothesis(alpha_candidate, beta_candidate)
                    if best is None:
                        best = hypothesis
                        continue
                    if hypothesis["inlier_count"] > best["inlier_count"]:
                        best = hypothesis
                    elif (
                        hypothesis["inlier_count"] == best["inlier_count"]
                        and hypothesis["mae"] < best["mae"]
                    ):
                        best = hypothesis

        if best is None:
            return error_response("Unable to estimate a stable hypothesis", 500)

        inlier_indices = np.where(best["inliers"])[0]
        if len(inlier_indices) >= 2:

            def objective(alpha_rad, beta_rad):
                modeled = model_roll_deg(alpha_rad, beta_rad, yaw_rads[inlier_indices])
                errors = np.array(
                    [
                        angular_diff_deg(float(predicted_rolls[idx]), float(modeled[k]))
                        for k, idx in enumerate(inlier_indices)
                    ],
                    dtype=np.float32,
                )
                return float(np.mean(np.square(errors)))

            alpha_center = best["alpha_rad"]
            beta_center = best["beta_rad"]

            alpha_grid_1 = np.deg2rad(np.arange(-6.0, 6.0 + 0.001, 0.5))
            beta_grid_1 = np.deg2rad(np.arange(-6.0, 6.0 + 0.001, 0.5))
            best_obj = objective(alpha_center, beta_center)

            for da in alpha_grid_1:
                for db in beta_grid_1:
                    alpha_try = normalize_rad(alpha_center + da)
                    beta_try = np.clip(beta_center + db, np.deg2rad(-89.0), np.deg2rad(89.0))
                    obj = objective(alpha_try, beta_try)
                    if obj < best_obj:
                        best_obj = obj
                        alpha_center = alpha_try
                        beta_center = beta_try

            alpha_grid_2 = np.deg2rad(np.arange(-1.0, 1.0 + 0.001, 0.1))
            beta_grid_2 = np.deg2rad(np.arange(-1.0, 1.0 + 0.001, 0.1))
            for da in alpha_grid_2:
                for db in beta_grid_2:
                    alpha_try = normalize_rad(alpha_center + da)
                    beta_try = np.clip(beta_center + db, np.deg2rad(-89.0), np.deg2rad(89.0))
                    obj = objective(alpha_try, beta_try)
                    if obj < best_obj:
                        best_obj = obj
                        alpha_center = alpha_try
                        beta_center = beta_try

            best = evaluate_hypothesis(alpha_center, beta_center)

        alpha_deg = float(np.rad2deg(best["alpha_rad"]))
        beta_deg = float(np.rad2deg(best["beta_rad"]))

        tan_beta = np.tan(best["beta_rad"])
        roll_deg = float(np.rad2deg(np.arctan(tan_beta * np.cos(best["alpha_rad"]))))
        pitch_deg = float(np.rad2deg(-np.arctan(tan_beta * np.sin(best["alpha_rad"]))))

        sample_entries = []
        modeled_rolls = best["modeled_rolls"]
        for idx in range(sample_count):
            success, buffer = cv2.imencode(".jpg", sample_images[idx])
            if not success:
                return error_response("Failed to encode sampled image", 500)
            sample_image_id = str(uuid.uuid4())
            image_cache.set(sample_image_id, buffer.tobytes())
            sample_entries.append(
                {
                    "index": idx,
                    "yaw_deg": float(sample_yaws[idx]),
                    "predicted_roll_deg": float(predicted_rolls[idx]),
                    "modeled_roll_deg": float(modeled_rolls[idx]),
                    "error_deg": float(best["errors"][idx]),
                    "inlier": bool(best["inliers"][idx]),
                    "image_id": sample_image_id,
                }
            )

        return jsonify(
            {
                "status": "success",
                "main_id": main_image_id,
                "sample_count": sample_count,
                "fov_deg": fov_deg,
                "inlier_threshold_deg": inlier_threshold_deg,
                "alpha_deg": alpha_deg,
                "beta_deg": beta_deg,
                "roll": roll_deg,
                "pitch": pitch_deg,
                "inlier_count": int(best["inlier_count"]),
                "inlier_ratio": float(best["inlier_count"] / sample_count),
                "mae_inlier_deg": float(best["mae"]),
                "rmse_inlier_deg": float(best["rmse"]),
                "samples": sample_entries,
            }
        )
    except Exception as e:
        logger.error("Error processing 360 image: %s", e)
        logger.error(traceback.format_exc())
        return error_response(str(e), 500)


@app.after_request
def add_header(response):
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
    response.headers["Cache-Control"] = "public, max-age=0"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
