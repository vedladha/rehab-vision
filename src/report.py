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
    if analysis.exercise == "single_leg_rdl":
        details = analysis.details
        summary += (
            f"\nSupporting leg: {details['supporting_leg']} (configured, not inferred).\n"
            f"\nView: {details['camera_view']}. Rep timing is available only in side view.\n"
            "\nMeasurements use observed joints only. Missing, interpolated, overlapping, "
            "or suspect identity intervals can make measurements unavailable.\n"
            "\nHip travel is a screen-horizontal offset, not proof of backward motion. "
            "Ankle movement includes possible camera movement and tracking noise. "
            "Pelvic tilt does not measure pelvic rotation.\n"
            "\nRDL detection is provisional and has not been evaluated on labeled RDL recordings.\n"
        )
        if not details["rep_count_available"]:
            summary = summary.replace(f"Observed complete events: {complete}", "Observed complete events: unavailable for this view")
            summary = summary.replace(f"Observed incomplete/rejected events: {incomplete}", "Observed incomplete/rejected events: unavailable for this view")
        write_rdl_observations(output, analysis)
    summary_path.write_text(summary)

    run_path = output / "run.json"
    run_path.write_text(json.dumps(clean(run), indent=2))


def write_rdl_observations(output, analysis):
    names = list(analysis.signals)
    with (output / "observations.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "time_s", "supporting_leg", "status"] + names)
        writer.writeheader()
        for i, status in enumerate(analysis.details["frame_status"]):
            row = {"frame": i, "time_s": i / analysis.fps,
                   "supporting_leg": analysis.details["supporting_leg"], "status": status}
            row.update({name: clean(analysis.signals[name][i]) for name in names})
            writer.writerow(row)
    (output / "measurements.json").write_text(json.dumps(clean({
        "units": analysis.details["units"],
        "consistency": analysis.details["consistency"],
        "scope": "whole recording; completed reps only",
        "events_file": "events.json",
        "observations_file": "observations.csv",
    }), indent=2))


def plot(output, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if analysis.exercise == "single_leg_rdl":
        names = [name for name in analysis.signals if analysis.availability[name]]
        figure, axes = plt.subplots(max(1, len(names)), 1, figsize=(11, max(3, 2.3 * len(names))), squeeze=False)
        for ax, name in zip(axes[:, 0], names):
            values = analysis.signals[name]
            ax.plot(np.arange(len(values)) / analysis.fps, values, color="#1f6fea")
            ax.set_title(name.replace("_", " "))
            ax.set_xlabel("Video time (s)")
            ax.set_ylabel("deg" if name.endswith("deg") else "% image height")
            ax.grid(alpha=0.2)
        if not names:
            axes[0, 0].text(0.5, 0.5, "Measurements unavailable", ha="center")
        figure.tight_layout()
        figure.savefig(Path(output) / "movement.png", dpi=150)
        plt.close(figure)
        return

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
