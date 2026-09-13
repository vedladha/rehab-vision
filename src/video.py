"""Prepare input videos and encode rendered results with FFmpeg."""

import json
import subprocess
from pathlib import Path

import cv2


def run(cmd):
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stderr[-3000:])
    return process


def probe(path):
    process = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,"
                "duration:stream_side_data=rotation"
            ),
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(process.stdout)


def convert(source, output, height, crf):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map_metadata",
            "0",
            "-vf",
            f"scale=-2:'min({height},ih)'",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output),
        ]
    )


def inspect(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 30),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    info["duration"] = info["frames"] / info["fps"]
    return info


def encode(source, output, crf):
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
