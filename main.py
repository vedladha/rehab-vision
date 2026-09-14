"""Run one recorded exercise video through pose, analysis, and reporting."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone

import config as cfg


def main():
    from dotenv import load_dotenv
    from openai import OpenAI

    from src import video
    from src.analysis import analyze
    from src.pose import (
        build_request,
        cache_key,
        interpolate_short_gaps,
        request,
        select_track,
        unwrap,
    )
    from src.render import render
    from src.report import plot, write_reports

    load_dotenv(cfg.PROJECT_DIR / ".env", override=True)

    if cfg.EXERCISE not in cfg.PRESCRIPTION:
        raise ValueError(f"Unknown exercise: {cfg.EXERCISE}")
    if cfg.EXERCISE == "single_leg_rdl" and cfg.CAMERA_VIEW != "side":
        raise ValueError(
            "Single-leg RDL requires a side-view recording. Set CAMERA_VIEW = 'side' "
            "and select a side-view video; front and oblique views are not supported."
        )
    if cfg.EXERCISE == "single_leg_rdl" and cfg.ANALYZED_LEG not in ("left", "right"):
        raise ValueError("Set ANALYZED_LEG to the supporting left or right leg.")
    if not cfg.INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Put one recording at {cfg.INPUT_VIDEO}")

    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    output_dir = cfg.OUTPUT_DIR / stamp
    output_dir.mkdir(parents=True)

    started = time.perf_counter()
    metrics = {}
    source_meta = video.probe(cfg.INPUT_VIDEO)

    converted = cfg.CACHE_DIR / f"{cfg.INPUT_VIDEO.stem}.converted.mp4"
    stage_started = time.perf_counter()
    if not converted.exists():
        print("Converting video...", flush=True)
        video.convert(
            cfg.INPUT_VIDEO,
            converted,
            cfg.INFERENCE_HEIGHT,
            cfg.CONVERT_CRF,
        )
    else:
        print("Using cached converted video.", flush=True)
    metrics["conversion_s"] = time.perf_counter() - stage_started

    info = video.inspect(converted)
    video_b64, request_options = build_request(
        converted,
        info["fps"],
        info["frames"],
        cfg,
    )
    pose_key = cache_key(converted, cfg.MODEL, request_options)
    pose_cache = cfg.CACHE_DIR / f"poses.{pose_key}.json"

    usage = None
    if cfg.REUSE_POSES and pose_cache.exists():
        print("Using cached pose response.", flush=True)
        cached = json.loads(pose_cache.read_text())
        payload = cached["payload"]
        gateway_authentication = cached.get("authentication", "unknown_cached")
        api_seconds = 0.0
    else:
        print("Uploading video and estimating poses...", flush=True)
        api_key = os.getenv("VLMRUN_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VLMRUN_API_KEY is missing. Copy .env.example to .env "
                "and add it locally."
            )

        gateway_authentication = "api_key"
        client = OpenAI(
            api_key=api_key,
            base_url=cfg.GATEWAY_BASE_URL,
            timeout=cfg.REQUEST_TIMEOUT,
        )
        payload, usage, api_seconds = request(
            client,
            video_b64,
            request_options,
            cfg,
        )
        pose_cache.parent.mkdir(parents=True, exist_ok=True)
        pose_cache.write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model": cfg.MODEL,
                    "authentication": gateway_authentication,
                    "payload": payload,
                    "usage": usage,
                    "request": request_options,
                }
            )
        )

    metrics["api_round_trip_s"] = api_seconds
    raw_pose_path = output_dir / "raw_pose_response.json"
    raw_pose_path.write_text(json.dumps(payload, indent=2))

    print("Analyzing movement...", flush=True)
    stage_started = time.perf_counter()
    poses, tracking_quality = select_track(unwrap(payload), info["frames"])
    track = interpolate_short_gaps(
        poses,
        info["fps"],
        cfg.INTERPOLATION_MAX_SECONDS,
    )
    if cfg.EXERCISE == "single_leg_rdl":
        import numpy as np

        identity_valid = np.zeros(info["frames"], bool)
        for item in tracking_quality:
            identity_valid[item["frame"]] = item["person_count"] == 1
        previous_id = None
        for index, pose in enumerate(poses):
            if pose is not None:
                current_id = pose["track_id"]
                if previous_id is not None and current_id != previous_id:
                    identity_valid[index] = False
                previous_id = current_id
        track["identity_valid"] = identity_valid
    analysis = analyze(
        track,
        info["fps"],
        cfg.EXERCISE,
        cfg.CAMERA_VIEW,
        cfg.ANALYZED_LEG,
        cfg,
        info["width"],
        info["height"],
    )
    metrics["analysis_s"] = time.perf_counter() - stage_started

    print("Rendering dashboard video...", flush=True)
    stage_started = time.perf_counter()
    temporary_video = output_dir / "annotated.temp.mp4"
    render_meta = render(
        converted,
        track,
        analysis,
        temporary_video,
        cfg.PRESCRIPTION[cfg.EXERCISE],
    )
    metrics["rendering_s"] = time.perf_counter() - stage_started

    print("Encoding final MP4...", flush=True)
    stage_started = time.perf_counter()
    final_video = output_dir / "annotated.mp4"
    video.encode(temporary_video, final_video, cfg.OUTPUT_CRF)
    temporary_video.unlink()
    metrics["encoding_s"] = time.perf_counter() - stage_started
    metrics["total_s"] = time.perf_counter() - started

    source_meta["sha256"] = hashlib.sha256(
        cfg.INPUT_VIDEO.read_bytes()
    ).hexdigest()
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_meta,
        "input_name": cfg.INPUT_VIDEO.name,
        "exercise": cfg.EXERCISE,
        "camera_view": cfg.CAMERA_VIEW,
        "analyzed_leg": None if cfg.EXERCISE == "squat" else cfg.ANALYZED_LEG,
        "model": cfg.MODEL,
        "gateway_base_url": cfg.GATEWAY_BASE_URL,
        "gateway_authentication": gateway_authentication,
        "request": request_options,
        "reference_commit": "5e4b92195d16889c21cd5de55c9f14337d5d8542",
        "usage": usage,
        "api_cost": None,
        "processing": metrics,
        "tracking_quality": tracking_quality,
        "analysis_details": analysis.details,
        "render": render_meta,
        "provisional_detection_settings": {
            "smooth_seconds": cfg.SMOOTH_SECONDS,
            "min_phase_seconds": cfg.MIN_PHASE_SECONDS,
            "min_rep_seconds": cfg.MIN_REP_SECONDS,
            "amplitude_fraction": cfg.AMPLITUDE_FRACTION,
            "return_fraction": cfg.RETURN_FRACTION,
            "interpolation_max_seconds": cfg.INTERPOLATION_MAX_SECONDS,
        },
    }

    prescription = cfg.PRESCRIPTION[cfg.EXERCISE]
    write_reports(output_dir, analysis, prescription, run)
    plot(output_dir, analysis)
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
