"""Turn pose coordinates into exercise phases, events, and measurements."""

from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry import angle_degrees, trunk_angle
from .skeleton import INDEX


@dataclass
class Event:
    """One observed repetition or step-balance sequence."""

    number: int
    status: str
    start_s: float
    end_s: float
    phases: dict
    metrics: dict
    evidence_timestamps_s: list
    leg: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Analysis:
    """The complete analysis used by reports and the video dashboard."""

    exercise: str
    fps: float
    phase: list
    events: list
    signals: dict
    availability: dict
    tracking: dict
    details: dict = field(default_factory=dict)

    def count_at(self, frame):
        """Count only events completed by the displayed video frame."""
        current_time = frame / self.fps
        return sum(
            event.status == "complete" and event.end_s <= current_time
            for event in self.events
        )


def _smooth(values, window):
    """Reduce single-frame pose jitter with a moving median."""
    values = np.asarray(values, dtype=float)
    smoothed = values.copy()
    half_window = max(1, window // 2)

    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        neighborhood = values[start:end]

        if np.isfinite(neighborhood).any():
            smoothed[index] = np.nanmedian(neighborhood)
        else:
            smoothed[index] = np.nan

    return smoothed


def _required_joints_are_valid(track, joints):
    """Return one Boolean value per frame for a group of required joints."""
    return np.all(track["valid"][:, joints], axis=1)


def _phased_cycles(
    signal,
    valid,
    fps,
    cfg,
    low_is_peak=True,
    min_amplitude=0.0,
):
    """Find start, farthest point, and return for repeated movements.

    Separate enter and return levels add hysteresis. This prevents small pose
    jitter near one threshold from creating several repetitions.
    """
    smoothing_window = max(3, round(cfg.SMOOTH_SECONDS * fps) | 1)
    smoothed = _smooth(signal, smoothing_window)
    phases = ["unavailable"] * len(smoothed)

    if valid.any():
        baseline_percentile = 85 if low_is_peak else 15
        extreme_percentile = 10 if low_is_peak else 90
        baseline = np.nanpercentile(smoothed[valid], baseline_percentile)
        extreme = np.nanpercentile(smoothed[valid], extreme_percentile)
    else:
        baseline = np.nan
        extreme = np.nan

    amplitude = abs(baseline - extreme)
    enter_level = baseline + (extreme - baseline) * cfg.AMPLITUDE_FRACTION
    peak_level = baseline + (extreme - baseline) * 0.70
    return_level = baseline + (extreme - baseline) * cfg.RETURN_FRACTION

    def passes(value, threshold):
        return value <= threshold if low_is_peak else value >= threshold

    def returns(value, threshold):
        return value >= threshold if low_is_peak else value <= threshold

    state = "ready"
    start_frame = None
    peak_frame = None
    invalid_frames = 0
    raw_events = []

    for frame, value in enumerate(smoothed):
        if not valid[frame] or not np.isfinite(value):
            invalid_frames += 1
            phases[frame] = "unavailable"

            allowed_gap = round(cfg.INTERPOLATION_MAX_SECONDS * fps)
            if state != "ready" and invalid_frames > allowed_gap:
                last_valid_frame = max(start_frame, frame - invalid_frames)
                raw_events.append(
                    (
                        start_frame,
                        last_valid_frame,
                        peak_frame,
                        "incomplete_tracking",
                    )
                )
                state = "ready"
                start_frame = None
                peak_frame = None
            continue

        invalid_frames = 0

        if state == "ready":
            phases[frame] = "ready"
            if amplitude >= min_amplitude and passes(value, enter_level):
                state = "outward"
                start_frame = frame

        elif state == "outward":
            phases[frame] = "descent" if low_is_peak else "outward"

            if passes(value, peak_level):
                state = "return"
                peak_frame = frame
            elif returns(value, return_level):
                # Movement reversed too early to qualify as an event.
                state = "ready"
                start_frame = None

        else:
            phases[frame] = "ascent" if low_is_peak else "return"

            if passes(value, smoothed[peak_frame]):
                peak_frame = frame

            if returns(value, return_level):
                duration = (frame - start_frame) / fps
                status = (
                    "complete"
                    if duration >= cfg.MIN_REP_SECONDS
                    else "rejected_short"
                )
                raw_events.append((start_frame, frame, peak_frame, status))
                state = "ready"
                start_frame = None
                peak_frame = None

    if state != "ready":
        raw_events.append(
            (
                start_frame,
                len(smoothed) - 1,
                peak_frame,
                "incomplete_clip_boundary",
            )
        )

    return smoothed, phases, raw_events


def _event_metrics(
    start_frame,
    end_frame,
    peak_frame,
    track,
    fps,
    exercise,
    width,
    height,
):
    """Create phase timestamps and basic timing metrics for one event."""
    if peak_frame is None:
        peak_frame = (start_frame + end_frame) // 2

    def seconds(frame):
        return frame / fps

    phases = {}
    metrics = {
        "duration_s": round((end_frame - start_frame) / fps, 3),
        "time_between_previous_s": None,
    }

    if exercise == "squat":
        phases = {
            "descent": {
                "start_s": seconds(start_frame),
                "end_s": seconds(peak_frame),
            },
            "bottom_pause": {
                "start_s": seconds(peak_frame),
                "end_s": seconds(peak_frame),
            },
            "ascent": {
                "start_s": seconds(peak_frame),
                "end_s": seconds(end_frame),
            },
        }
        metrics.update(
            {
                "descent_s": round((peak_frame - start_frame) / fps, 3),
                "bottom_pause_s": None,
                "ascent_s": round((end_frame - peak_frame) / fps, 3),
            }
        )

        if width and height:
            knee_angles = _squat_knee_angles(
                track,
                start_frame,
                end_frame,
                width,
                height,
            )
            metrics["minimum_projected_knee_angle_deg"] = (
                round(float(min(knee_angles)), 1) if knee_angles else None
            )

    elif exercise == "skater_squat":
        phases = {
            "outward": {
                "start_s": seconds(start_frame),
                "end_s": seconds(peak_frame),
            },
            "return": {
                "start_s": seconds(peak_frame),
                "end_s": seconds(end_frame),
            },
        }
        metrics.update(
            {
                "outward_s": round((peak_frame - start_frame) / fps, 3),
                "outward_pause_s": None,
                "return_s": round((end_frame - peak_frame) / fps, 3),
            }
        )

    elif exercise == "single_leg_rdl":
        phases = {
            "lowering": {
                "start_s": seconds(start_frame),
                "end_s": seconds(peak_frame),
            },
            "bottom_pause": {
                "start_s": seconds(peak_frame),
                "end_s": seconds(peak_frame),
            },
            "return": {
                "start_s": seconds(peak_frame),
                "end_s": seconds(end_frame),
            },
        }
        metrics.update(
            {
                "lowering_s": round((peak_frame - start_frame) / fps, 3),
                "bottom_pause_s": None,
                "return_s": round((end_frame - peak_frame) / fps, 3),
            }
        )

    return phases, metrics, peak_frame


def _squat_knee_angles(track, start_frame, end_frame, width, height):
    """Collect available left and right projected knee angles."""
    angles = []

    for frame in range(start_frame, end_frame + 1):
        frame_angles = []
        for side in ("left", "right"):
            joint_ids = [
                INDEX[f"{side}_hip"],
                INDEX[f"{side}_knee"],
                INDEX[f"{side}_ankle"],
            ]
            if track["valid"][frame, joint_ids].all():
                frame_angles.append(
                    angle_degrees(
                        *track["xy"][frame, joint_ids],
                        width,
                        height,
                    )
                )

        if frame_angles:
            angles.append(min(frame_angles))

    return angles


def _tracking_summary(track):
    """Summarize how much of the clip came directly from pose observations."""
    observed_frames = track["observed"].any(axis=1)
    return {
        "observed_frame_fraction": round(float(observed_frames.mean()), 3),
        "interpolated_joint_samples": int(track["interpolated"].sum()),
        "confidence_available": bool(np.isfinite(track["confidence"]).any()),
    }


def _exercise_signal(track, exercise, view, leg, width, height):
    """Choose the movement signal and required joints for an exercise."""
    coordinates = track["xy"]
    hip_height = (coordinates[:, 11, 1] + coordinates[:, 12, 1]) / 2

    if exercise == "squat":
        return {
            "signal": hip_height,
            "required_joints": [11, 12, 13, 14, 15, 16],
            "low_is_peak": False,
            "minimum_amplitude": 0.04,
            "availability": {
                "projected_knee_flexion": view == "side",
                "lateral_knee_movement": view == "front",
                "trunk_angle": view == "side",
            },
        }

    if exercise == "skater_squat":
        supporting_leg = "left" if leg == "right" else "right"
        moving_ankle = INDEX[f"{leg}_ankle"]
        moving_hip = INDEX[f"{leg}_hip"]

        return {
            "signal": np.abs(
                coordinates[:, moving_ankle, 0]
                - coordinates[:, moving_hip, 0]
            ),
            "required_joints": [
                moving_ankle,
                moving_hip,
                INDEX[f"{supporting_leg}_knee"],
            ],
            "low_is_peak": False,
            "minimum_amplitude": 0.04,
            "availability": {
                "backward_component": view in ("side", "oblique"),
                "outward_component": view in ("front", "oblique"),
                "supporting_knee_inward": view == "front",
            },
        }

    if exercise == "single_leg_rdl":
        trunk_joints = [5, 6, 11, 12]
        signal = np.full(len(coordinates), np.nan)

        for frame in range(len(coordinates)):
            if track["valid"][frame, trunk_joints].all():
                signal[frame] = trunk_angle(coordinates[frame], width, height)

        return {
            "signal": signal,
            "required_joints": trunk_joints
            + [INDEX[f"{leg}_knee"], INDEX[f"{leg}_ankle"]],
            "low_is_peak": False,
            "minimum_amplitude": 4.0,
            "availability": {
                "trunk_hip_hinge": view in ("side", "oblique"),
                "spine_straight": False,
            },
        }

    raise ValueError(f"Unsupported exercise: {exercise}")


def analyze(
    track,
    fps,
    exercise,
    view,
    leg,
    cfg,
    width=None,
    height=None,
):
    """Analyze squat, skater squat, RDL, or step-balance movement."""
    if exercise == "single_leg_rdl":
        from .rdl import analyze_rdl

        return analyze_rdl(track, fps, view, leg, cfg, width, height)

    if exercise == "side_step_balance":
        return _analyze_steps(track, fps, view, cfg)

    setup = _exercise_signal(track, exercise, view, leg, width, height)
    valid_frames = _required_joints_are_valid(
        track,
        setup["required_joints"],
    )
    smoothed, phases, raw_events = _phased_cycles(
        setup["signal"],
        valid_frames,
        fps,
        cfg,
        setup["low_is_peak"],
        setup["minimum_amplitude"],
    )

    # The generic detector names motion "outward" and "return". Replace those
    # names with terms that make sense for the displayed exercise.
    if exercise == "squat":
        phases = [
            "descent" if phase == "outward" else
            "ascent" if phase == "return" else
            phase
            for phase in phases
        ]
    elif exercise == "single_leg_rdl":
        phases = [
            "lowering" if phase == "outward" else phase
            for phase in phases
        ]

    events = []
    for start, end, peak, status in raw_events:
        event_phases, metrics, peak = _event_metrics(
            start,
            end,
            peak,
            track,
            fps,
            exercise,
            width,
            height,
        )
        _add_event_measurements(
            metrics,
            smoothed,
            track,
            start,
            end,
            exercise,
            view,
            leg,
            width,
            height,
        )

        events.append(
            Event(
                number=len(events) + 1,
                status=status,
                start_s=start / fps,
                end_s=end / fps,
                phases=event_phases,
                metrics=metrics,
                evidence_timestamps_s=[
                    round(start / fps, 3),
                    round(peak / fps, 3),
                    round(end / fps, 3),
                ],
                leg=None if exercise == "squat" else leg,
            )
        )

    _add_between_event_times(events)

    return Analysis(
        exercise=exercise,
        fps=fps,
        phase=phases,
        events=events,
        signals={"movement": smoothed.tolist()},
        availability=setup["availability"],
        tracking=_tracking_summary(track),
    )


def _add_event_measurements(
    metrics,
    smoothed,
    track,
    start,
    end,
    exercise,
    view,
    leg,
    width,
    height,
):
    """Add measurements that need the full span of an event."""
    event_signal = np.asarray(smoothed[start : end + 1], dtype=float)
    if np.isfinite(event_signal).any():
        excursion = np.nanmax(event_signal) - np.nanmin(event_signal)
        metrics["projected_movement_excursion"] = round(float(excursion), 4)
    else:
        metrics["projected_movement_excursion"] = None

    if width and height:
        trunk_angles = []
        trunk_joints = [5, 6, 11, 12]
        for frame in range(start, end + 1):
            if track["valid"][frame, trunk_joints].all():
                trunk_angles.append(
                    trunk_angle(track["xy"][frame], width, height)
                )

        front_view_rdl = view == "front" and exercise == "single_leg_rdl"
        metrics["projected_trunk_angle_range_deg"] = (
            round(float(max(trunk_angles) - min(trunk_angles)), 1)
            if trunk_angles and not front_view_rdl
            else None
        )

    if exercise == "skater_squat":
        supporting_leg = "left" if leg == "right" else "right"
        knee = INDEX[f"{supporting_leg}_knee"]
        hip = INDEX[f"{supporting_leg}_hip"]
        lateral_position = (
            track["xy"][start : end + 1, knee, 0]
            - track["xy"][start : end + 1, hip, 0]
        )
        metrics["supporting_knee_lateral_displacement"] = (
            round(float(np.nanmax(lateral_position) - np.nanmin(lateral_position)), 4)
            if view == "front"
            else None
        )

    if exercise == "single_leg_rdl":
        joint_ids = [
            INDEX[f"{leg}_hip"],
            INDEX[f"{leg}_knee"],
            INDEX[f"{leg}_ankle"],
        ]
        knee_angles = []
        for frame in range(start, end + 1):
            if track["valid"][frame, joint_ids].all():
                knee_angles.append(
                    angle_degrees(
                        *track["xy"][frame, joint_ids],
                        width,
                        height,
                    )
                )

        metrics["supporting_knee_angle_change_deg"] = (
            round(float(max(knee_angles) - min(knee_angles)), 1)
            if knee_angles
            else None
        )


def _add_between_event_times(events):
    """Measure from one event's completion to the next event's start."""
    for previous, current in zip(events, events[1:]):
        current.metrics["time_between_previous_s"] = round(
            current.start_s - previous.end_s,
            3,
        )


def _analyze_steps(track, fps, view, cfg):
    """Find groups of three lateral steps followed by a balance phase."""
    required_joints = [11, 12, 15, 16]
    valid_frames = _required_joints_are_valid(track, required_joints)

    ankle_x = track["xy"][:, [15, 16], 0]
    lateral_center = np.nanmean(ankle_x, axis=1)
    smoothing_window = max(3, round(cfg.SMOOTH_SECONDS * fps) | 1)
    lateral_center = _smooth(lateral_center, smoothing_window)
    lateral_velocity = np.r_[0, np.diff(lateral_center)] * fps

    if valid_frames.any():
        typical_motion = np.nanpercentile(
            np.abs(lateral_velocity[valid_frames]),
            70,
        )
        velocity_threshold = max(typical_motion * 0.45, 0.015)
    else:
        velocity_threshold = np.inf

    moving = np.abs(lateral_velocity) > velocity_threshold
    phases = ["stepping" if is_moving else "ready" for is_moving in moving]
    step_frames = _step_onsets(moving, fps)
    events = _group_steps(track, step_frames, lateral_center, fps, view)
    _add_between_event_times(events)

    return Analysis(
        exercise="side_step_balance",
        fps=fps,
        phase=phases,
        events=events,
        signals={
            "lateral_center": lateral_center.tolist(),
            "lateral_velocity": lateral_velocity.tolist(),
        },
        availability={
            "balance_single_leg": view == "front",
            "lateral_trunk_movement": view == "front",
        },
        tracking=_tracking_summary(track),
    )


def _step_onsets(moving, fps):
    """Record the first frame of each separated burst of lateral motion."""
    step_frames = []
    inside_motion = False
    minimum_separation = round(0.5 * fps)

    for frame, is_moving in enumerate(moving):
        if is_moving and not inside_motion:
            separated = (
                not step_frames
                or frame - step_frames[-1] >= minimum_separation
            )
            if separated:
                step_frames.append(frame)
            inside_motion = True
        elif not is_moving:
            inside_motion = False

    return step_frames


def _group_steps(track, step_frames, lateral_center, fps, view):
    """Turn each group of three detected steps into one sequence."""
    events = []

    for group_start in range(0, len(step_frames), 3):
        steps = step_frames[group_start : group_start + 3]
        complete = len(steps) == 3
        start_frame = steps[0]

        next_group = group_start + 3
        if complete and next_group < len(step_frames):
            end_frame = step_frames[next_group] - 1
        else:
            end_frame = len(lateral_center) - 1

        balance_start = steps[-1]
        status = "complete" if complete else "incomplete_clip_boundary"

        hip_height = np.nanmean(
            track["xy"][start_frame : end_frame + 1, [11, 12], 1],
            axis=1,
        )
        shoulder_x = np.nanmean(
            track["xy"][start_frame : end_frame + 1, [5, 6], 0],
            axis=1,
        )
        hip_x = np.nanmean(
            track["xy"][start_frame : end_frame + 1, [11, 12], 0],
            axis=1,
        )
        lateral_trunk = shoulder_x - hip_x

        events.append(
            Event(
                number=len(events) + 1,
                status=status,
                start_s=start_frame / fps,
                end_s=end_frame / fps,
                phases={
                    "steps": {
                        "timestamps_s": [round(frame / fps, 3) for frame in steps]
                    },
                    "balance": {
                        "start_s": balance_start / fps,
                        "end_s": end_frame / fps,
                    },
                },
                metrics={
                    "steps_observed": len(steps),
                    "balance_duration_s": (
                        round((end_frame - balance_start) / fps, 3)
                        if complete
                        else None
                    ),
                    "time_between_previous_s": None,
                    "projected_hip_height_range": round(
                        float(np.nanmax(hip_height) - np.nanmin(hip_height)),
                        4,
                    ),
                    "visible_lateral_trunk_range": (
                        round(
                            float(
                                np.nanmax(lateral_trunk)
                                - np.nanmin(lateral_trunk)
                            ),
                            4,
                        )
                        if view == "front"
                        else None
                    ),
                },
                evidence_timestamps_s=[
                    round(frame / fps, 3) for frame in steps
                ],
            )
        )

    return events
