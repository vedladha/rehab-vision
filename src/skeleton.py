"""COCO-17 keypoints and drawing helpers. Modified from vision-demos."""
import cv2

NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
INDEX = {name: i for i, name in enumerate(NAMES)}
EDGES = [(15,13,"L"),(13,11,"L"),(16,14,"R"),(14,12,"R"),(11,12,"T"),
         (5,11,"L"),(6,12,"R"),(5,6,"T"),(5,7,"L"),(7,9,"L"),
         (6,8,"R"),(8,10,"R")]
COLORS = {"L": (255,200,0), "R": (0,140,255), "T": (80,220,80)}

def draw_person(image, pose, observed=True):
    h, w = image.shape[:2]
    pts = pose["xy"]
    valid = pose["valid"]
    for a, b, side in EDGES:
        if valid[a] and valid[b]:
            p = tuple((pts[a] * [w, h]).round().astype(int))
            q = tuple((pts[b] * [w, h]).round().astype(int))
            color = COLORS[side] if observed else tuple(c // 2 for c in COLORS[side])
            cv2.line(image, p, q, color, 3, cv2.LINE_AA)
    for i in range(5, 17):
        if valid[i]:
            p = tuple((pts[i] * [w, h]).round().astype(int))
            cv2.circle(image, p, 4, (245,245,245), -1, cv2.LINE_AA)
