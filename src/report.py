import csv
import json
from pathlib import Path

import numpy as np


def clean(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def write_reports(output, analysis, prescription, run):
    output = Path(output)
    events = [clean(event.to_dict()) for event in analysis.events]

    events_json = output / "events.json"
    events_json.write_text(json.dumps(events, indent=2))

    fields = [
        "number",
        "status",
        "leg",
        "start_s",
        "end_s",
        "phases",
        "metrics",
        "evidence_timestamps_s",
    ]
    events_csv = output / "events.csv"

    with events_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for event in events:
            row = {
                **event,
                "phases": json.dumps(event["phases"]),
                "metrics": json.dumps(event["metrics"]),
                "evidence_timestamps_s": json.dumps(
                    event["evidence_timestamps_s"]
                ),
            }
            writer.writerow(row)

    complete = sum(event["status"] == "complete" for event in events)
    incomplete = len(events) - complete

    summary = (
        "# Session summary\n\n"
        f"Exercise: {analysis.exercise.replace('_', ' ')}\n\n"
        f"Observed complete events: {complete}\n\n"
        f"Observed incomplete/rejected events: {incomplete}\n\n"
        f"Prescription (context only): {json.dumps(prescription)}\n\n"
        "Set boundaries were not inferred. Measurements are single-camera "
        "2D pose estimates. They do not establish 3D motion, weight "
        "distribution, joint forces, band tension, muscle activation, "
        "clinical safety, or diagnosis.\n"
    )

    summary_path = output / "summary.md"
    summary_path.write_text(summary)

    run_path = output / "run.json"
    run_path.write_text(json.dumps(clean(run), indent=2))


def plot(output, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signal_name, values = next(iter(analysis.signals.items()))
    values = np.asarray(values, dtype=float)
    times = np.arange(len(values)) / analysis.fps

    figure, axes = plt.subplots(figsize=(12, 4))
    axes.plot(times, values, color="#1f6fea")

    for event in analysis.events:
        color = "#2f9e44" if event.status == "complete" else "#e8590c"
        axes.axvspan(
            event.start_s,
            event.end_s,
            alpha=0.12,
            color=color,
        )

    title = analysis.exercise.replace("_", " ").title()
    axes.set(
        title=f"{title} movement signal",
        xlabel="time (s)",
        ylabel=signal_name.replace("_", " "),
    )
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(Path(output) / "movement.png", dpi=150)
    plt.close(figure)
