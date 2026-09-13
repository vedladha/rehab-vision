from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_VIDEO = DATA_DIR / "input" / "squat_front.mov"

# Choose: squat, side_step_balance, skater_squat, single_leg_rdl
EXERCISE = "squat"
# Choose front, side, or oblique. 
CAMERA_VIEW = "front"
# Required for skater_squat and single_leg_rdl: left or right.
# For skater squat this is the moving leg; for RDL it is the supporting leg.
# Ignored for squat, which counts whole-body repetitions.
ANALYZED_LEG = "right"

MODEL = "usyd-community/vitpose-plus-large"
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0
EVERY_FRAME = True
VIDEO_FPS = 2.0
VIDEO_MAX_FRAMES = None
PRECISION = 5
REUSE_POSES = True
MAX_PERSONS = 1

INFERENCE_HEIGHT = 1080
EXPORT_HEIGHT = 720
CONVERT_CRF = 23
OUTPUT_CRF = 20
INTERPOLATION_MAX_SECONDS = 0.15
MIN_VALID_FRACTION = 0.75

SMOOTH_SECONDS = 0.17
MIN_PHASE_SECONDS = 0.20
MIN_REP_SECONDS = 0.65
AMPLITUDE_FRACTION = 0.18
RETURN_FRACTION = 0.35
BALANCE_MIN_SECONDS = 0.35

OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"

PRESCRIPTION = {
    "squat": {"sets": 3, "reps_per_set": 12},
    "side_step_balance": {"sets": 3, "sequences_per_set": 5},
    "skater_squat": {"sets": 3, "reps_per_leg": 10},
    "single_leg_rdl": {"sets": 3, "reps_per_leg": 10},
}
