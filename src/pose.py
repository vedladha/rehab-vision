"""VLM Run request, cache, and conservative track preparation."""
from __future__ import annotations
import base64, hashlib, json, time
from pathlib import Path
import numpy as np

CONTENT_OBJECT = "vid.pose.kpts"
SUPPORTED_CONTENT_OBJECTS = {CONTENT_OBJECT, "vitpose_plus.pose.frames"}

def build_request(video, fps, n_frames, cfg):
    extra = {"method": "pose", "video_fps": fps if cfg.EVERY_FRAME else cfg.VIDEO_FPS,
             "precision": cfg.PRECISION}
    if cfg.EVERY_FRAME or cfg.VIDEO_MAX_FRAMES is not None:
        extra["video_max_frames"] = n_frames if cfg.EVERY_FRAME else cfg.VIDEO_MAX_FRAMES
    return base64.b64encode(Path(video).read_bytes()).decode(), extra

def request(client, video_b64, extra, cfg):
    start = time.perf_counter()
    response = client.chat.completions.create(model=cfg.MODEL, messages=[{"role":"user",
        "content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,"+video_b64}}]}],
        response_format={"type":"json_object"}, extra_body=extra)
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage, time.perf_counter() - start

def unwrap(payload):
    content = payload["data"][0]["content"]
    if content.get("object") not in SUPPORTED_CONTENT_OBJECTS:
        raise RuntimeError(f"Expected {CONTENT_OBJECT}, got {content.get('object')!r}")
    if content["object"] == "vitpose_plus.pose.frames":
        return content["items"]

    frames = {}
    for metadata in content.get("frames", []):
        index = _frame_index(metadata.get("frame_index"))
        frames[index] = {"index": index, "persons": []}

    # New responses list people separately; group them by their actual frame.
    for person in content["items"]:
        index = _frame_index(person.get("frame"))
        frame = frames.setdefault(index, {"index": index, "persons": []})
        frame["persons"].append(person)

    return [frames[index] for index in sorted(frames)]


def _frame_index(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid pose frame index: {value!r}")
    return value

def cache_key(video, model, extra):
    path = Path(video); stat = path.stat()
    raw = json.dumps({"name":path.name,"size":stat.st_size,"mtime":stat.st_mtime_ns,
                      "model":model,"request":extra}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def person_to_pose(person):
    xy = np.asarray(person.get("kpts_xy", []), float)
    if xy.shape != (17, 2): xy = np.full((17, 2), np.nan)
    valid = np.isfinite(xy).all(axis=1) & np.any(xy != 0, axis=1)
    scores = person.get("kpts_score", person.get("kpts_scores"))
    confidence = np.asarray(scores, float) if scores is not None and len(scores) == 17 else None
    return {"xy":xy, "valid":valid, "confidence":confidence,
            "track_id":person.get("track_id"), "bbox":person.get("bbox_xywh")}

def select_track(frames, n_frames):
    result = [None] * n_frames; quality = []
    previous_id = None
    for item in frames:
        i = _frame_index(item.get("index"))
        if i >= n_frames:
            raise ValueError(f"Pose frame {i} exceeds video length {n_frames}")
        people = item.get("persons") or []
        status = "missing"
        if people:
            chosen = next((p for p in people if p.get("track_id") == previous_id), None)
            if chosen is None:
                chosen = max(people, key=lambda p: p.get("bbox_xywh", [0,0,0,0])[2] * p.get("bbox_xywh", [0,0,0,0])[3])
            pose = person_to_pose(chosen); result[i] = pose; previous_id = pose["track_id"]
            status = "multiple_people" if len(people) > 1 else "observed"
        quality.append({"frame":i,"status":status,"person_count":len(people)})
    return result, quality

def interpolate_short_gaps(poses, fps, max_seconds):
    n = len(poses); xy = np.full((n,17,2), np.nan); observed = np.zeros((n,17), bool)
    confidence = np.full((n,17), np.nan)
    for i, pose in enumerate(poses):
        if pose is not None:
            xy[i,pose["valid"]] = pose["xy"][pose["valid"]]; observed[i] = pose["valid"]
            if pose["confidence"] is not None: confidence[i] = pose["confidence"]
    valid = observed.copy(); interpolated = np.zeros_like(valid); limit = max(0, int(fps*max_seconds))
    for j in range(17):
        ids = np.flatnonzero(observed[:,j])
        for a,b in zip(ids[:-1], ids[1:]):
            if 1 < b-a <= limit+1:
                for i in range(a+1,b):
                    xy[i,j] = xy[a,j] + (xy[b,j]-xy[a,j]) * (i-a)/(b-a)
                    valid[i,j] = interpolated[i,j] = True
    return {"xy":xy,"valid":valid,"observed":observed,"interpolated":interpolated,
            "confidence":confidence}
