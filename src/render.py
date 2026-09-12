import cv2
import numpy as np

from .skeleton import draw_person


BG = (20, 22, 26)
WHITE = (245, 245, 245)
MUTED = (145, 150, 158)
BLUE = (255, 160, 40)
GREEN = (102, 207, 81)


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


def panel(
    analysis,
    frame,
    width,
    height,
    prescription,
    tracking_status,
):
    image = np.full((height, width, 3), BG, np.uint8)
    current_time = frame / analysis.fps
    count = analysis.count_at(frame)
    phase = (
        analysis.phase[min(frame, len(analysis.phase) - 1)]
        if analysis.phase
        else "unavailable"
    )

    put(
        image,
        analysis.exercise.replace("_", " ").title(),
        (40, 60),
        1.0,
    )
    put(image, str(count), (40, 155), 2.7, WHITE, 5)
    put(image, "observed complete", (45, 190), 0.55, MUTED)

    target = prescription.get(
        "reps_per_set",
        prescription.get(
            "sequences_per_set",
            prescription.get("reps_per_leg"),
        ),
    )
    put(
        image,
        f"Prescription: {prescription['sets']} x {target}",
        (40, 230),
        0.62,
        MUTED,
    )

    put(image, "CURRENT FRAME", (40, 290), 0.48, MUTED)
    put(image, f"Phase  {phase}", (40, 330), 0.74, BLUE)
    put(image, f"Time   {current_time:6.2f}s", (40, 370), 0.64)

    tracking_color = GREEN if tracking_status == "observed" else MUTED
    put(
        image,
        f"Tracking  {tracking_status}",
        (40, 410),
        0.54,
        tracking_color,
    )

    put(image, "OBSERVED EVENTS", (40, 470), 0.48, MUTED)
    visible_events = [
        event
        for event in analysis.events
        if event.end_s <= current_time
    ][-2:]

    event_y = 510
    for event in visible_events:
        duration = event.metrics.get(
            "duration_s",
            event.metrics.get("balance_duration_s"),
        )
        label = f"{event.number:02d}  {event.status.replace('_', ' ')}"
        if duration is not None:
            label += f"  {duration:.2f}s"

        event_color = WHITE if event.status == "complete" else (90, 170, 255)
        put(image, label, (40, event_y), 0.48, event_color)
        event_y += 31

    signal_name = next(iter(analysis.signals))
    values = np.asarray(analysis.signals[signal_name], dtype=float)
    chart = (40, height - 150, width - 40, height - 78)
    cv2.rectangle(
        image,
        (chart[0], chart[1]),
        (chart[2], chart[3]),
        (45, 48, 55),
        1,
    )

    finite_values = values[np.isfinite(values)]
    if len(finite_values) and frame > 1:
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
        span = max(maximum - minimum, 1e-6)
        visible_count = min(frame + 1, len(values))
        points = []

        for index, value in enumerate(values[:visible_count]):
            if not np.isfinite(value):
                continue

            x = int(
                chart[0]
                + (chart[2] - chart[0])
                * index
                / max(1, len(values) - 1)
            )
            y = int(
                chart[3]
                - (chart[3] - chart[1])
                * (value - minimum)
                / span
            )
            points.append((x, y))

        if len(points) > 1:
            cv2.polylines(
                image,
                [np.asarray(points, np.int32)],
                False,
                BLUE,
                2,
                cv2.LINE_AA,
            )

    put(
        image,
        "movement signal (through current frame)",
        (40, height - 158),
        0.40,
        MUTED,
        1,
    )
    put(
        image,
        "Whole-session values appear only after observed",
        (40, height - 64),
        0.42,
        MUTED,
        1,
    )
    put(
        image,
        "events. Measurements are 2D pose estimates.",
        (40, height - 38),
        0.42,
        MUTED,
        1,
    )
    return image


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
