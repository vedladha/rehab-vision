import cv2
import numpy as np

from .skeleton import draw_person


BG = (20, 22, 26)
WHITE = (245, 245, 245)
MUTED = (145, 150, 158)
BLUE = (255, 160, 40)
GREEN = (102, 207, 81)
ORANGE = (40, 155, 255)


def put(image, text, position, scale, color=WHITE, weight=2):
    cv2.putText(
        image,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        weight,
        cv2.LINE_AA,
    )


def centered(image, text, x, y, scale, color=WHITE):
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    put(image, text, (int(x - size[0] / 2), y), scale, color, 1)


def completed_durations(analysis, frame):
    current_time = frame / analysis.fps
    events = [
        event for event in analysis.events
        if event.status == "complete" and event.end_s <= current_time
    ]
    durations = []
    for number, event in enumerate(events, 1):
        duration = event.metrics.get("duration_s")
        if analysis.exercise == "side_step_balance":
            duration = event.end_s - event.start_s
        if duration is None or not np.isfinite(duration) or duration <= 0:
            duration = None
        durations.append((number, duration))
    return durations


def duration_chart(image, durations, unit, compact=False):
    left, top, right, bottom = (110, 665, 554, 815) if compact else (110, 460, 554, 790)
    known = [(number, value) for number, value in durations if value is not None]
    values = [value for _, value in known]
    padding = max((max(values) - min(values)) * 0.35, 0.15) if values else 0
    lower = max(0, min(values) - padding) if values else 0
    upper = max(values) + padding if values else 2
    x_limit = max(6, len(durations))

    put(image, f"{unit.title()} duration", (40, 612 if compact else 386), 0.7 if compact else 0.85)
    put(image, "Completed so far", (40, 641 if compact else 418), 0.48, MUTED, 1)

    label_height = 120 if compact else 190
    axis_label = np.full((32, label_height, 3), BG, np.uint8)
    centered(axis_label, "Duration (s)", label_height / 2, 22, 0.48 if compact else 0.53, MUTED)
    axis_label = cv2.rotate(axis_label, cv2.ROTATE_90_COUNTERCLOCKWISE)
    label_y = 680 if compact else 530
    image[label_y:label_y + label_height, 22:54] = axis_label

    for value in np.linspace(lower, upper, 5):
        y = int(bottom - (value - lower) / (upper - lower) * (bottom - top))
        cv2.line(image, (left, y), (right, y), (43, 46, 51), 1)
        centered(image, f"{value:.2f}", 79, y + 5, 0.43, MUTED)

    def x_position(number):
        return int(left + (number - 0.5) / x_limit * (right - left))

    tick_step = max(1, int(np.ceil(x_limit / 12)))
    for number in range(1, x_limit + 1):
        if (number - 1) % tick_step == 0 or number == x_limit:
            centered(image, str(number), x_position(number), bottom + 29, 0.48, MUTED)
    centered(image, f"{unit.title()} number", (left + right) / 2, bottom + 66, 0.53, MUTED)

    if not known:
        centered(image, f"Waiting for a completed {unit}", (left + right) / 2, (top + bottom) // 2, 0.49, MUTED)
        return

    fastest = min(values)
    slowest = max(values)
    previous = None
    for number, value in durations:
        if value is None:
            previous = None
            continue
        point = (x_position(number), int(bottom - (value - lower) / (upper - lower) * (bottom - top)))
        if previous is not None:
            cv2.line(image, previous, point, BLUE, 2, cv2.LINE_AA)
        color = BLUE
        if len(known) > 1 and fastest < slowest:
            color = GREEN if value == fastest else ORANGE if value == slowest else BLUE
        cv2.circle(image, point, 7, color, -1, cv2.LINE_AA)
        if x_limit <= 10:
            centered(image, f"{value:.2f}", point[0], point[1] - 18, 0.44)
        previous = point


def duration_summary(image, durations, unit):
    known = [(number, value) for number, value in durations if value is not None]
    if len(known) < 2:
        put(image, "Fastest / slowest: unavailable", (40, 931), 0.59, MUTED)
        put(image, f"Needs two completed {unit}s with timing", (40, 971), 0.48, MUTED, 1)
        return

    fastest = min(known, key=lambda item: item[1])
    slowest = max(known, key=lambda item: item[1])
    difference = (slowest[1] - fastest[1]) / fastest[1] * 100
    put(image, f"Fastest:    {unit} {fastest[0]}  /  {fastest[1]:.2f}s", (40, 922), 0.64, GREEN)
    put(image, f"Slowest:   {unit} {slowest[0]}  /  {slowest[1]:.2f}s", (40, 967), 0.64, ORANGE)
    put(image, f"{difference:.1f}% longer than fastest", (40, 1018), 0.59, MUTED)


def panel(analysis, frame, width, height, prescription=None, tracking_status=None):
    image = np.full((1080, 608, 3), BG, np.uint8)
    durations = completed_durations(analysis, frame)
    unit = "sequence" if analysis.exercise == "side_step_balance" else "rep"
    phase = analysis.phase[min(frame, len(analysis.phase) - 1)] if analysis.phase else "unavailable"

    put(image, analysis.exercise.replace("_", " ").title(), (40, 62), 0.95)
    put(image, str(analysis.count_at(frame)), (40, 180), 3.2, WHITE, 5)
    put(image, f"completed {unit}s", (43, 218), 0.57, MUTED)
    put(image, "CURRENT PHASE", (40, 280), 0.48, MUTED)
    put(image, phase.replace("_", " ").capitalize(), (40, 321), 0.85, BLUE)

    duration_chart(image, durations, unit)
    duration_summary(image, durations, unit)

    if analysis.exercise == "single_leg_rdl":
        image = rdl_panel(analysis, frame, durations)

    scale = min(width / 608, height / 1080)
    resized = cv2.resize(image, (round(608 * scale), round(1080 * scale)), interpolation=cv2.INTER_AREA)
    result = np.full((height, width, 3), BG, np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    result[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return result


def rdl_panel(analysis, frame, durations):
    image = np.full((1080, 608, 3), BG, np.uint8)
    details = analysis.details
    side = details["camera_view"] == "side"
    count = str(analysis.count_at(frame)) if details["rep_count_available"] else "--"
    put(image, "Single Leg RDL", (40, 52), 0.95)
    put(image, f"Supporting leg: {details['supporting_leg'].title()} (configured)", (40, 89), 0.56, BLUE)
    put(image, count, (40, 192), 2.7, WHITE, 4)
    put(image, "completed reps" if side else "count unavailable in this view", (40, 223), 0.5, MUTED)
    put(image, "CURRENT PHASE", (280, 145), 0.44, MUTED)
    phase = analysis.phase[frame].capitalize()
    put(image, phase, (280, 182), 0.62, BLUE)

    names = (["knee_flexion_deg", "trunk_lean_deg", "hip_ankle_offset_pct_height"]
             if side else ["knee_inward_offset_pct_height", "pelvic_tilt_deg", "lateral_trunk_lean_deg"])
    labels = (["Knee bend", "Trunk lean", "Hip / ankle offset"]
              if side else ["Knee inward offset", "Pelvic tilt", "Lateral trunk lean"])
    for y, name, label in zip((265, 299, 333), names, labels):
        value = analysis.signals[name][frame]
        units = "deg" if name.endswith("deg") else "% height"
        text = f"{value:.1f} {units}" if np.isfinite(value) else "unavailable"
        put(image, label, (40, y), 0.52, MUTED)
        put(image, text, (340, y), 0.52)

    name = names[0]
    values = np.asarray(analysis.signals[name], float)
    start = max(0, frame - round(10 * analysis.fps))
    visible = values[start:frame + 1]
    finite = visible[np.isfinite(visible)]
    unit = "deg" if side else "% image height"
    put(image, f"{labels[0]} ({unit})", (40, 385), 0.56)
    left, right, top, bottom = 90, 555, 413, 557
    cv2.rectangle(image, (left, top), (right, bottom), (50, 53, 58), 1)
    if len(finite):
        low, high = float(finite.min()) - 1, float(finite.max()) + 1
        put(image, f"{high:.1f}", (40, top + 8), 0.4, MUTED, 1)
        put(image, f"{low:.1f}", (40, bottom), 0.4, MUTED, 1)
        previous = None
        for i, value in enumerate(visible):
            if not np.isfinite(value):
                previous = None
                continue
            point = (int(left + i / max(1, len(visible) - 1) * (right - left)),
                     int(bottom - (value - low) / (high - low) * (bottom - top)))
            if previous is not None:
                cv2.line(image, previous, point, BLUE, 2, cv2.LINE_AA)
            previous = point
    else:
        centered(image, "Measurement unavailable", 320, 491, 0.5, MUTED)
    put(image, f"{start / analysis.fps:.1f}s", (90, 582), 0.42, MUTED, 1)
    put(image, f"{frame / analysis.fps:.1f}s", (505, 582), 0.42, MUTED, 1)

    if details["rep_count_available"]:
        duration_chart(image, durations, "rep", compact=True)
        duration_summary(image, durations, "rep")
    else:
        put(image, "Rep timing unavailable in this view", (40, 659), 0.57, MUTED)
        put(image, "Use a side recording for hinge repetitions.", (40, 700), 0.5, MUTED, 1)
    status = details["frame_status"][frame]
    if status != "observed":
        put(image, status.replace("_", " "), (40, 1060), 0.44, ORANGE, 1)
    return image


def draw_rdl_ankle_trail(image, analysis, frame):
    x = analysis.signals["ankle_x_pct_height"]
    y = analysis.signals["ankle_y_pct_height"]
    history = round(analysis.details["ankle_trail_seconds"] * analysis.fps)
    color = BLUE if analysis.details["supporting_leg"] == "left" else ORANGE
    previous = None
    for i in range(max(0, frame - history), frame + 1):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            previous = None
            continue
        point = (round(x[i] / 100 * image.shape[0]), round(y[i] / 100 * image.shape[0]))
        if previous is not None:
            cv2.line(image, previous, point, color, 2, cv2.LINE_AA)
        previous = point
    if previous is not None and np.isfinite(x[frame]) and np.isfinite(y[frame]):
        cv2.circle(image, previous, 9, color, 2, cv2.LINE_AA)


def render(video, track, analysis, output, prescription):
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    frame_index = 0

    while True:
        frame_available, frame = capture.read()
        if not frame_available:
            break

        if frame_index < len(track["xy"]):
            if analysis.exercise == "single_leg_rdl":
                draw_rdl_ankle_trail(frame, analysis, frame_index)
            pose = {
                "xy": track["xy"][frame_index],
                "valid": track["valid"][frame_index],
            }
            observed = bool(track["observed"][frame_index].any())
            draw_person(frame, pose, observed)

            if track["interpolated"][frame_index].any():
                put(
                    frame,
                    "interpolated - not a new observation",
                    (20, 35),
                    0.55,
                    (100, 180, 255),
                )

        if track["observed"][frame_index].any():
            tracking_status = "observed"
        elif track["interpolated"][frame_index].any():
            tracking_status = "interpolated"
        else:
            tracking_status = "unavailable"

        dashboard = panel(
            analysis,
            frame_index,
            width,
            height,
            prescription,
            tracking_status,
        )
        writer.write(np.hstack([frame, dashboard]))
        frame_index += 1

    capture.release()
    writer.release()

    return {
        "frames_written": frame_index,
        "output_size": [width * 2, height],
    }
