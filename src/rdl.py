"""Observable supporting-leg movement, with separate rules for each view."""

import numpy as np

from .geometry import angle_degrees
from .skeleton import INDEX


UNITS = {
    "knee_flexion_deg": "degrees; 0 means projected straight knee",
    "trunk_lean_deg": "degrees from image vertical",
    "hip_ankle_offset_pct_height": "% image height; positive screen right",
    "knee_inward_offset_pct_height": "% image height; positive toward other hip",
    "pelvic_tilt_deg": "degrees; positive means anatomical right hip lower",
    "lateral_trunk_lean_deg": "degrees; positive screen right",
    "ankle_x_pct_height": "% image height from left edge",
    "ankle_y_pct_height": "% image height from top edge",
}


def measurements(track, view, leg, width, height, fps, cfg):
    if leg not in ("left", "right"):
        raise ValueError("For RDL, ANALYZED_LEG must be the supporting left or right leg.")
    if view not in ("side", "front", "oblique"):
        raise ValueError("CAMERA_VIEW must be side, front, or oblique.")
    if not width or not height or fps <= 0:
        raise ValueError("RDL measurements require video dimensions and frame rate.")

    xy = np.asarray(track["xy"], float)
    pixels = xy * [width, height]
    valid = track["observed"].copy() & track["valid"]
    valid &= np.isfinite(xy).all(axis=2)
    valid &= ((xy >= 0) & (xy <= 1)).all(axis=2)
    scores = track["confidence"]
    valid &= ~np.isfinite(scores) | (scores >= cfg.RDL_MIN_KEYPOINT_CONFIDENCE)
    identity_valid = track.get("identity_valid", np.ones(len(xy), bool))
    valid[~identity_valid] = False
    hip, knee, ankle, shoulder = [INDEX[f"{leg}_{joint}"] for joint in
                                  ("hip", "knee", "ankle", "shoulder")]
    other = "left" if leg == "right" else "right"
    other_hip = INDEX[f"{other}_hip"]
    leg_ids = [hip, knee, ankle]
    other_ids = [INDEX[f"{other}_{joint}"] for joint in ("hip", "knee", "ankle")]
    reasons = ["observed"] * len(xy)
    swapped = False

    for i in range(len(xy)):
        if i and valid[i - 1, leg_ids + other_ids].all() and valid[i, leg_ids + other_ids].all():
            previous = pixels[i - 1, leg_ids + other_ids]
            current = pixels[i, leg_ids + other_ids]
            same = np.linalg.norm(current - previous, axis=1).mean()
            crossed = np.linalg.norm(current - previous[[3, 4, 5, 0, 1, 2]], axis=1).mean()
            if same / height > cfg.RDL_SWAP_MIN_DISPLACEMENT and crossed < same * 0.5:
                swapped = True
        if swapped:
            valid[i] = False
            reasons[i] = "possible_left_right_swap"
        elif not identity_valid[i]:
            reasons[i] = "person_identity_unavailable"
        elif not valid[i, leg_ids + [shoulder]].all():
            reasons[i] = "required_joints_unavailable"
        elif valid[i, [knee, ankle] + other_ids[1:]].all():
            separations = np.linalg.norm(pixels[i, [knee, ankle]] - pixels[i, other_ids[1:]], axis=1)
            if np.all(separations / height < cfg.RDL_OVERLAP_DISTANCE):
                valid[i, leg_ids] = False
                reasons[i] = "legs_overlap"

    signals = {name: np.full(len(xy), np.nan) for name in UNITS}
    for i, points in enumerate(pixels):
        if valid[i, ankle]:
            signals["ankle_x_pct_height"][i], signals["ankle_y_pct_height"][i] = points[ankle] / height * 100
        if view == "side":
            if valid[i, leg_ids].all():
                signals["knee_flexion_deg"][i] = 180 - angle_degrees(*xy[i, leg_ids], width, height)
            if valid[i, [shoulder, hip]].all():
                delta = points[shoulder] - points[hip]
                signals["trunk_lean_deg"][i] = np.degrees(np.arctan2(abs(delta[0]), -delta[1]))
            if valid[i, [hip, ankle]].all():
                signals["hip_ankle_offset_pct_height"][i] = (points[hip, 0] - points[ankle, 0]) / height * 100
        elif view == "front":
            if valid[i, leg_ids + [other_hip]].all():
                span = points[ankle, 1] - points[hip, 1]
                direction = np.sign(points[other_hip, 0] - points[hip, 0])
                if abs(span) > height * 0.02 and direction:
                    fraction = (points[knee, 1] - points[hip, 1]) / span
                    line_x = points[hip, 0] + fraction * (points[ankle, 0] - points[hip, 0])
                    signals["knee_inward_offset_pct_height"][i] = direction * (points[knee, 0] - line_x) / height * 100
            if valid[i, [11, 12]].all():
                delta = points[12] - points[11]
                if abs(delta[0]) > height * 0.02:
                    signals["pelvic_tilt_deg"][i] = np.degrees(np.arctan2(delta[1], abs(delta[0])))
            if valid[i, [5, 6, 11, 12]].all():
                delta = points[[5, 6]].mean(axis=0) - points[[11, 12]].mean(axis=0)
                signals["lateral_trunk_lean_deg"][i] = np.degrees(np.arctan2(delta[0], -delta[1]))

    eligible = valid[:, leg_ids + [shoulder]].all(axis=1)
    details = {
        "supporting_leg": leg,
        "leg_source": "user configuration; support not inferred",
        "camera_view": view,
        "units": UNITS,
        "frame_status": reasons,
        "rep_count_available": view == "side",
        "ankle_trail_seconds": cfg.RDL_ANKLE_TRAIL_SECONDS,
        "thresholds": {name: getattr(cfg, name) for name in dir(cfg) if name.startswith("RDL_")},
    }
    return signals, eligible, details


def detect_cycles(lean, eligible, fps, cfg):
    """Require an observed upright start and return; break at tracking gaps."""
    phase = ["unavailable"] * len(lean)
    cycles = []
    ready = False
    start = None
    peak = None
    initial_partial = False
    for i, value in enumerate(lean):
        if not eligible[i] or not np.isfinite(value):
            if start is not None:
                cycles.append((start, i - 1, peak, "incomplete_tracking"))
            ready, start, peak = False, None, None
            continue
        if start is None:
            phase[i] = "ready" if value <= cfg.RDL_RETURN_LEAN_DEG else "unavailable"
            if value <= cfg.RDL_RETURN_LEAN_DEG:
                ready = True
            elif value >= cfg.RDL_ENTER_LEAN_DEG:
                start, peak = i, i
                initial_partial = not ready
                phase[i] = "lowering" if ready else "partial movement"
            continue
        if value > lean[peak]:
            peak = i
        phase[i] = "return" if value < lean[peak] - cfg.RDL_PAUSE_TOLERANCE_DEG else "lowering"
        if value <= cfg.RDL_RETURN_LEAN_DEG:
            enough = (i - start) / fps >= cfg.MIN_REP_SECONDS
            enough &= (peak - start) / fps >= cfg.MIN_PHASE_SECONDS
            enough &= (i - peak) / fps >= cfg.MIN_PHASE_SECONDS
            enough &= lean[peak] >= cfg.RDL_PEAK_LEAN_DEG
            status = "complete" if enough else "incomplete_excursion_or_duration"
            if initial_partial:
                status = "incomplete_start_unobserved"
            cycles.append((start, i, peak, status))
            ready, start, peak = True, None, None
            phase[i] = "ready"
    if start is not None:
        cycles.append((start, len(lean) - 1, peak, "incomplete_clip_boundary"))
    return phase, cycles


def summarize(values, start, end, fps, minimum_fraction):
    segment = np.asarray(values[start:end + 1])
    ids = np.flatnonzero(np.isfinite(segment))
    result = {"min": None, "max": None, "range": None, "min_s": None,
              "max_s": None, "observed_fraction": len(ids) / len(segment)}
    if len(ids) < 2 or len(ids) / len(segment) < minimum_fraction:
        return result
    low, high = ids[np.argmin(segment[ids])], ids[np.argmax(segment[ids])]
    result.update(min=float(segment[low]), max=float(segment[high]),
                  range=float(segment[high] - segment[low]),
                  min_s=(start + int(low)) / fps, max_s=(start + int(high)) / fps)
    return result


def analyze_rdl(track, fps, view, leg, cfg, width, height):
    from .analysis import Analysis, Event

    signals, eligible, details = measurements(track, view, leg, width, height, fps, cfg)
    lean = signals["trunk_lean_deg"]
    phases, cycles = detect_cycles(lean, eligible, fps, cfg)
    events = []
    previous_complete = None
    for start, end, peak, status in cycles:
        bounds = [start, peak, end]
        measurements_by_name = {
            name: summarize(values, start, end, fps, cfg.MIN_VALID_FRACTION)
            for name, values in signals.items()
        }
        metrics = {"duration_s": (end - start) / fps,
                   "lowering_s": None, "return_s": None, "bottom_pause_s": None,
                   "time_between_previous_s": None,
                   "observations": measurements_by_name}
        event_phases = {}
        if status == "complete":
            a = b = peak
            while a > start and abs(lean[a - 1] - lean[peak]) <= cfg.RDL_PAUSE_TOLERANCE_DEG:
                a -= 1
            while b < end and abs(lean[b + 1] - lean[peak]) <= cfg.RDL_PAUSE_TOLERANCE_DEG:
                b += 1
            paused = (b - a) / fps >= cfg.MIN_PHASE_SECONDS
            if not paused:
                a = b = peak
            metrics.update(lowering_s=(a - start) / fps, return_s=(end - b) / fps,
                           bottom_pause_s=(b - a) / fps)
            metrics["time_between_previous_s"] = (start - previous_complete) / fps if previous_complete is not None else None
            previous_complete = end
            event_phases = {
                "lowering": {"start_s": start / fps, "end_s": a / fps},
                "bottom_pause": {"start_s": a / fps, "end_s": b / fps},
                "return": {"start_s": b / fps, "end_s": end / fps},
            }
            for i in range(start, end):
                phases[i] = "lowering" if i < a else "bottom pause" if paused and i <= b else "return"
            bounds.extend([a, b])
        else:
            previous_complete = None
        evidence = [i / fps for i in sorted(set(bounds))]
        for observation in measurements_by_name.values():
            evidence.extend(t for t in (observation["min_s"], observation["max_s"]) if t is not None)
        events.append(Event(len(events) + 1, status, start / fps, end / fps,
                            event_phases, metrics, sorted(set(evidence)), leg))

    details["consistency"] = {}
    for name in signals:
        ranges = [e.metrics["observations"][name]["range"] for e in events if e.status == "complete"]
        ranges = [value for value in ranges if value is not None]
        details["consistency"][name] = {
            "median_rep_range": float(np.median(ranges)) if ranges else None,
            "range_std": float(np.std(ranges)) if len(ranges) > 1 else None,
            "rep_count": len(ranges),
        }
    return Analysis("single_leg_rdl", fps, phases, events,
                    {name: values.tolist() for name, values in signals.items()},
                    {name: bool(np.isfinite(values).any()) for name, values in signals.items()},
                    {"confidence_available": bool(np.isfinite(track["confidence"]).any())}, details)
