# Rehab Vision

Recorded-video movement analysis for four exercises in a home program. It reports observable timing, counts, and 2D consistency. It is not a clinical safety assessment, diagnosis, or exercise prescription.


## Example output

![Side-view squat video with pose landmarks, repetition count, movement phase, and dashboard](docs/images/squat-side-dashboard.jpg)


## Run it

1. **Get a VLM Run API key** at [app.vlm.run/sign-in](https://app.vlm.run/sign-in).

2. **Create a local `.env` file** at the repository root:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and add the key after `VLMRUN_API_KEY=`. Do not commit this file. The pose pipeline uploads the converted video to VLM Run for processing.

3. **Create and activate the Conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate rehab-vision
   ```

4. **Add one exercise clip** under `data/input/`.

5. **Configure the recording** in [`config.py`](config.py) by setting `INPUT_VIDEO`, `EXERCISE`, `CAMERA_VIEW`, and `ANALYZED_LEG`.

6. **Run the analysis** from the repository root:

   ```bash
   python main.py
   ```

Results are saved under `data/output/`, including an annotated video, event data, a plot, and a session summary. Converted videos and pose results are cached to avoid unnecessary API requests.

Use a `side` view for squat and RDL angles, a `front` view for lateral movement, or an `oblique` view for skater squats. Measurements and detection settings are provisional 2D estimates, not clinical guidance.

## Validation

```bash
pytest -q
python scripts/make_demo.py
```

## Recording guidance

### Single-leg RDL

Set `EXERCISE = "single_leg_rdl"` and `ANALYZED_LEG = "left"` or `"right"` for the **supporting leg**, using your anatomical side, not the side of the screen. Keep that leg unchanged throughout the clip. Support is configured, not inferred from the image.

Use a steady **side view** showing the supporting shoulder, hip, knee, and ankle for rep timing, projected knee bend, trunk lean, and hip travel. Include an upright start and finish. Use a separate **front view** for supporting-knee inward displacement, pelvic tilt, and lateral trunk lean. Front and oblique views do not count RDL reps; they show unavailable timing. Oblique view currently supports the ankle trace only.

The video keeps the duration chart and adds current measurements, a movement trace, and a two-second supporting-ankle trail. The trail follows the image, so camera movement and pose noise affect it. Its presentation is inspired by the [running reference](https://github.com/jeremyipark/vision-demos/tree/main/running).

RDL outputs include `observations.csv` (one timestamped row per frame), `measurements.json` (units and whole-recording consistency), and per-rep extrema with timestamps in `events.json`/`events.csv`. Distances are percentages of image height, not centimeters. Hip offset is positive toward screen right; it does not establish backward motion. Knee offset is relative to the projected hip-to-ankle line, positive toward the opposite hip. Pelvic tilt is positive when the anatomical right hip appears lower; it does not measure pelvis rotation.

Only directly observed, in-frame joints contribute. Interpolated samples are excluded. Gaps interrupt a rep; suspected left/right swaps suppress subsequent measurements until the clip ends. These checks cannot detect every labeling error. Missing confidence scores stay unavailable. A detected bottom pause means the trunk signal stays within the configured angle tolerance around its maximum for at least `MIN_PHASE_SECONDS`; zero means no pause met that detection rule. All `RDL_*` thresholds are provisional detection settings, not rehabilitation targets.

RDL measurement accuracy remains unverified until labeled RDL recordings are evaluated. Compare clips only with consistent camera placement. These metrics do not establish foot pressure, muscle activation, ACL loading, or clinical safety.

Record one exercise per clip with the phone steady, your full body visible, good lighting, and no other people in frame. Include a few seconds before and after the movement.
