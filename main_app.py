# main_app.py - PROFESSIONAL TEXTILE DEFECT DETECTION
# Multi Scanner UI: one image -> separate scan outputs -> final single verdict

import os
import json
import time
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from streamlit_oauth import OAuth2Component
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Textile Defect Detection | Professional",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CONFIGURATION
# ============================================

# IMPORTANT:
# For public GitHub/Streamlit Cloud, move these values to Streamlit Secrets.
GEMINI_API_KEY = "AQ.Ab8RN6I3T-UXa9DN1NlW2E4WVemrMo5HsITXt6YyJLsVXuvqZg"

# Google OAuth Configuration
CLIENT_ID = "933775442031-8q1kjhsanunatshkm6cb220ekvardovb.apps.googleusercontent.com"
CLIENT_SECRET = "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"
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REFRESH_TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_TOKEN_URL = "https://oauth2.googleapis.com/revoke"
REDIRECT_URI = "https://fabric-project-csc.streamlit.app/"

# Admin emails
ADMINS = ["santhoshwebworker@gmail.com", "e22cs003@shanmugha.edu.in"]

USERS_FILE = "users_data.json"
REPORTS_FILE = "reports_data.json"
MODEL_PATH = "best.pt"
YOLO_CONFIDENCE = 0.25

# ============================================
# SESSION STATE
# ============================================

DEFAULT_SESSION = {
    "logged_in": False,
    "user_email": "",
    "user_name": "",
    "user_role": "",
    "current_image": None,
    "analysis_history": [],
}

for key, value in DEFAULT_SESSION.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================
# DATABASE HELPERS
# ============================================

def init_db():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    if not os.path.exists(REPORTS_FILE):
        with open(REPORTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def _safe_load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        return default


def save_user(user_data):
    users = _safe_load_json(USERS_FILE, {})
    if user_data["email"] not in users:
        users[user_data["email"]] = user_data
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)


def get_all_users():
    return _safe_load_json(USERS_FILE, {})


def save_report(report_data):
    reports = _safe_load_json(REPORTS_FILE, [])
    reports.append(report_data)

    if len(reports) > 1000:
        reports = reports[-1000:]

    with open(REPORTS_FILE, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def get_all_reports():
    return _safe_load_json(REPORTS_FILE, [])


def get_user_reports(email):
    reports = get_all_reports()
    return [r for r in reports if r.get("user_email") == email]


init_db()


# ============================================
# YOLO MODEL HELPERS
# ============================================

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)


def normalize_defect_class(class_name):
    """
    Final project classes:
    hole / stain / tear only.
    Thread/line classes are not shown as separate scanner outputs.
    """
    name = str(class_name).strip().lower()

    if "stain" in name:
        return "stain"
    if "tear" in name or "kilichu" in name or "cut" in name or "thread" in name:
        return "tear"
    if "hole" in name:
        return "hole"

    return "ignore"


def _is_kilichu_like_box(item, image_size=512):
    """
    If model predicts a large opening as 'hole', convert it to Tear/Kilichu.
    This keeps the UI same, but makes torn-cloth images display under Tear/Kilichu.
    """
    x, y, w, h = _bbox_from_item(item)
    area = w * h
    img_area = image_size * image_size
    aspect = max(w / max(h, 1), h / max(w, 1))

    # Large torn opening / hand-through cloth opening
    if area >= img_area * 0.045:
        return True
    if w >= image_size * 0.22 and h >= image_size * 0.22:
        return True
    if h >= image_size * 0.32 and w >= image_size * 0.16:
        return True
    if aspect >= 2.6 and area >= img_area * 0.020:
        return True

    return False


def _refine_tear_box_to_visible_opening(scan_img, item):
    """
    YOLO sometimes returns a large box for torn cloth.
    This function makes the box tight around the visible hole/opening area
    such as finger/skin/dark gap in the middle of the tear.
    """
    x, y, w, h = _bbox_from_item(item)
    ih, iw = scan_img.shape[:2]

    x = max(0, min(iw - 1, int(x)))
    y = max(0, min(ih - 1, int(y)))
    w = max(1, min(iw - x, int(w)))
    h = max(1, min(ih - y, int(h)))

    roi = scan_img[y:y+h, x:x+w]
    if roi.size == 0 or w < 15 or h < 15:
        return item

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Skin/finger visible inside cloth opening.
    lower_skin1 = np.array([0, 18, 45], dtype=np.uint8)
    upper_skin1 = np.array([28, 210, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 18, 45], dtype=np.uint8)
    upper_skin2 = np.array([179, 210, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin1, upper_skin1) | cv2.inRange(hsv, lower_skin2, upper_skin2)

    # Dark shadow/gap inside torn cloth.
    dark_mask = cv2.inRange(gray, 0, 105)

    # Prefer non-fabric-looking region inside the YOLO box.
    # This catches finger/skin and deep dark gap, but avoids full fabric texture.
    mask = cv2.bitwise_or(skin_mask, dark_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return item

    roi_area = w * h
    candidates = []
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < max(90, roi_area * 0.010):
            continue
        if area > roi_area * 0.80:
            continue
        if cw < 8 or ch < 8:
            continue

        # Give more score to central regions, because real opening is usually inside the defect box.
        ccx, ccy = cx + cw / 2, cy + ch / 2
        center_dist = abs(ccx - w / 2) / max(w, 1) + abs(ccy - h / 2) / max(h, 1)
        score = area * (1.25 - min(center_dist, 1.0))
        candidates.append((score, area, cx, cy, cw, ch))

    if not candidates:
        return item

    _, _, cx, cy, cw, ch = max(candidates, key=lambda t: t[0])

    pad = 10
    nx1 = max(0, x + cx - pad)
    ny1 = max(0, y + cy - pad)
    nx2 = min(iw, x + cx + cw + pad)
    ny2 = min(ih, y + cy + ch + pad)

    refined = dict(item)
    refined["x"] = int(nx1)
    refined["y"] = int(ny1)
    refined["w"] = int(nx2 - nx1)
    refined["h"] = int(ny2 - ny1)
    refined["bbox"] = (refined["x"], refined["y"], refined["w"], refined["h"])
    refined["center"] = [int((nx1 + nx2) / 2), int((ny1 + ny2) / 2)]
    refined["location"] = f"X={refined['center'][0]}, Y={refined['center'][1]}"
    refined["area"] = int(refined["w"] * refined["h"])
    refined["opening_refined"] = True
    refined["display_note"] = "Box tightened around visible hole/opening/finger area"
    return refined


def _add_hole_for_visible_opening(result):
    """
    If the tear scanner found a visible opening/finger gap, also show it in Hole Scanner.
    This gives output like:
    Hole = 1 and Tear/Kilichu = 1 for torn cloth with visible hole.
    """
    if result.get("holes"):
        return result

    refined_tears = [t for t in result.get("tears", []) if t.get("opening_refined")]
    if not refined_tears:
        return result

    best = max(refined_tears, key=lambda t: float(t.get("confidence", 0)) * max(1, int(t.get("area", 1))))
    hole_item = dict(best)
    hole_item["id"] = 1
    hole_item["defect_type"] = "hole"
    hole_item["class"] = "hole/opening"
    hole_item["source"] = "Visible opening derived from Tear/Kilichu detection"
    hole_item["display_note"] = "Finger/dark opening visible inside torn cloth; counted as Hole also"

    result["holes"].append(hole_item)
    result["confidence_scores"]["hole"] = max(result["confidence_scores"].get("hole", 0.03), float(best.get("confidence", 0.03)))
    return result


def _rebuild_detected_locations(result):
    """Rebuild location table after hole-to-tear correction and bbox refinement."""
    rows = []
    for scanner_key, scanner_name in [
        ("holes", "Hole Scanner"),
        ("stains", "Stain Scanner"),
        ("tears", "Tear/Kilichu Scanner"),
    ]:
        for item in result.get(scanner_key, []):
            x, y, w, h = _bbox_from_item(item)
            cx, cy = int(x + w / 2), int(y + h / 2)
            rows.append({
                "S.No": len(rows) + 1,
                "Scanner": scanner_name,
                "Class": item.get("class", item.get("defect_type", "")),
                "Confidence": f"{float(item.get('confidence', 0)) * 100:.1f}%",
                "X": cx,
                "Y": cy,
                "Width": w,
                "Height": h,
                "Location": f"X={cx}, Y={cy}",
            })
    result["detected_locations"] = rows
    return result


def postprocess_hole_tear_items(result, scan_img):
    """
    Final post-processing for review/demo:
    - Large hole-like prediction becomes Tear/Kilichu.
    - Existing large Tear/Kilichu boxes are tightened to actual inner opening.
    - If finger/dark opening is visible, it is also counted as Hole.
    """
    holes = result.get("holes", [])
    large_holes = [h for h in holes if _is_kilichu_like_box(h)]

    if large_holes:
        large_holes = sorted(
            large_holes,
            key=lambda it: (float(it.get("confidence", 0)), int(it.get("area", 0))),
            reverse=True
        )
        tear_item = dict(large_holes[0])
        tear_item["id"] = len(result.get("tears", [])) + 1
        tear_item["defect_type"] = "tear"
        tear_item["class"] = "tear/kilichu"
        tear_item["source"] = "YOLOv8 best.pt + hole-to-tear correction"
        tear_item["display_note"] = "Large cloth opening detected; classified as Tear/Kilichu"
        tear_item = _refine_tear_box_to_visible_opening(scan_img, tear_item)

        # Remove large full-area hole boxes to avoid duplicate red boxes.
        result["holes"] = [h for h in holes if not _is_kilichu_like_box(h)]
        result["tears"].append(tear_item)

    # Refine any tear predicted directly by YOLO also.
    refined_tears = []
    for t in result.get("tears", []):
        if _is_kilichu_like_box(t) or int(t.get("area", 0)) > 2500:
            refined_tears.append(_refine_tear_box_to_visible_opening(scan_img, t))
        else:
            refined_tears.append(t)
    result["tears"] = refined_tears

    # For finger/opening image, show both Hole and Tear/Kilichu.
    result = _add_hole_for_visible_opening(result)

    # Reassign IDs and refresh confidence.
    for key in ["holes", "stains", "tears"]:
        for idx, item in enumerate(result.get(key, []), 1):
            item["id"] = idx

    result["confidence_scores"]["hole"] = max([float(h.get("confidence", 0.03)) for h in result["holes"]], default=0.03)
    result["confidence_scores"]["tear"] = max([float(t.get("confidence", 0.03)) for t in result["tears"]], default=0.03)

    return result



def _is_same_area(a, b, iou_thr=0.20, center_thr=32):
    """Check two boxes are same / very close area."""
    ax, ay, aw, ah = _bbox_from_item(a)
    bx, by, bw, bh = _bbox_from_item(b)
    acx, acy = ax + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    center_dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    return _iou(a, b) >= iou_thr or center_dist <= center_thr


def _append_unique_defect(result, key, item):
    """Append defect only if same area is not already present."""
    for old in result.get(key, []):
        if _is_same_area(old, item):
            # Keep stronger confidence and tighter bbox.
            if float(item.get("confidence", 0)) > float(old.get("confidence", 0)):
                old.update(item)
            return result

    item = dict(item)
    item["id"] = len(result.get(key, [])) + 1
    result.setdefault(key, []).append(item)
    return result


def _find_rule_based_openings(scan_img):
    """
    Rule-based backup detector.
    Purpose:
    - When YOLO misses a finger/skin/dark visible cloth opening, still show defect.
    - Avoid small mesh/net holes by ignoring tiny repeated dots.
    """
    ih, iw = scan_img.shape[:2]
    hsv = cv2.cvtColor(scan_img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(scan_img, cv2.COLOR_RGB2GRAY)

    # Skin/finger region
    lower_skin1 = np.array([0, 18, 45], dtype=np.uint8)
    upper_skin1 = np.array([28, 210, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 18, 45], dtype=np.uint8)
    upper_skin2 = np.array([179, 210, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin1, upper_skin1) | cv2.inRange(hsv, lower_skin2, upper_skin2)

    # Dark real opening/shadow region
    dark_mask = cv2.inRange(gray, 0, 95)

    # Ignore top label/text area in many sample images
    ignore_top = int(ih * 0.10)
    skin_mask[:ignore_top, :] = 0
    dark_mask[:ignore_top, :] = 0

    # Avoid detecting full red/orange fabric as skin: remove very saturated red blocks
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    high_red_like = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 165)) & (sat > 215) & (val > 80)
    skin_mask[high_red_like] = 0

    mask = cv2.bitwise_or(skin_mask, dark_mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    openings = []
    img_area = iw * ih
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if w < 16 or h < 16:
            continue
        if area < max(450, img_area * 0.002):
            continue
        if area > img_area * 0.35:
            continue
        if x <= 2 or y <= ignore_top or x + w >= iw - 2 or y + h >= ih - 2:
            continue

        roi_skin = skin_mask[y:y+h, x:x+w]
        roi_dark = dark_mask[y:y+h, x:x+w]
        skin_ratio = float(np.mean(roi_skin > 0))
        dark_ratio = float(np.mean(roi_dark > 0))

        # Mesh/net fabric creates lots of tiny dark dots; reject low density tiny patterns.
        if skin_ratio < 0.05 and dark_ratio < 0.10:
            continue

        # Avoid label/text blocks: text is thin and wide; real opening has height too.
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 5.5:
            continue

        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad)
        y2 = min(ih, y + h + pad)
        fw, fh = x2 - x1, y2 - y1
        farea = fw * fh
        confidence = 0.79 if skin_ratio >= 0.10 or farea >= img_area * 0.018 else 0.58

        openings.append({
            "x": int(x1),
            "y": int(y1),
            "w": int(fw),
            "h": int(fh),
            "bbox": (int(x1), int(y1), int(fw), int(fh)),
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "location": f"X={int((x1 + x2) / 2)}, Y={int((y1 + y2) / 2)}",
            "area": int(farea),
            "confidence": round(confidence, 4),
            "skin_ratio": round(skin_ratio, 4),
            "dark_ratio": round(dark_ratio, 4),
            "opening_refined": True,
            "source": "Rule based visible opening fallback"
        })

    # Sort strongest first and limit to avoid false many-count outputs.
    openings = sorted(openings, key=lambda it: (it["confidence"], it["area"]), reverse=True)
    return openings[:6]


def _add_rule_based_visible_openings(result, scan_img):
    """
    Add missing Hole/Tear outputs using image cues.
    Rules:
    - Finger/skin/dark opening in torn cloth => Hole + Tear/Kilichu.
    - Small dark hole => Hole only.
    - Do not add duplicates near YOLO boxes.
    """
    openings = _find_rule_based_openings(scan_img)
    if not openings:
        return result

    img_area = scan_img.shape[0] * scan_img.shape[1]

    for op in openings:
        is_big_opening = op["area"] >= img_area * 0.018
        has_skin = op.get("skin_ratio", 0) >= 0.08

        # Add hole for every visible opening.
        hole_item = dict(op)
        hole_item["defect_type"] = "hole"
        hole_item["class"] = "hole/opening"
        hole_item["display_note"] = "Visible cloth opening detected by rule-based backup"
        result = _append_unique_defect(result, "holes", hole_item)

        # Finger visible / large torn opening should also be Tear/Kilichu.
        if has_skin or is_big_opening:
            tear_item = dict(op)
            tear_item["defect_type"] = "tear"
            tear_item["class"] = "tear/kilichu"
            tear_item["display_note"] = "Finger/large opening indicates torn cloth"
            result = _append_unique_defect(result, "tears", tear_item)

    result["confidence_scores"]["hole"] = max([float(h.get("confidence", 0.03)) for h in result.get("holes", [])], default=0.03)
    result["confidence_scores"]["tear"] = max([float(t.get("confidence", 0.03)) for t in result.get("tears", [])], default=0.03)
    return result



def _find_relaxed_skin_opening(scan_img):
    """
    Backup for hand/finger-through-cloth images.
    If a skin-colored finger is visible through the cloth opening and YOLO misses it,
    count it as both Hole and Tear/Kilichu.
    """
    ih, iw = scan_img.shape[:2]
    hsv = cv2.cvtColor(scan_img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(scan_img, cv2.COLOR_RGB2GRAY)

    lower_skin1 = np.array([0, 16, 45], dtype=np.uint8)
    upper_skin1 = np.array([30, 210, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 16, 45], dtype=np.uint8)
    upper_skin2 = np.array([179, 210, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin1, upper_skin1) | cv2.inRange(hsv, lower_skin2, upper_skin2)

    # Avoid red cloth / red annotation becoming skin.
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    high_red_like = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 165)) & (sat > 215) & (val > 70)
    skin_mask[high_red_like] = 0

    # Ignore top title/label area.
    skin_mask[:int(ih * 0.08), :] = 0

    kernel = np.ones((7, 7), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    img_area = iw * ih
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < img_area * 0.012:
            continue
        if area > img_area * 0.45:
            continue
        if w < 25 or h < 25:
            continue
        if y <= int(ih * 0.08) or x <= 1 or x + w >= iw - 1 or y + h >= ih - 1:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 4.5:
            continue

        # Need a dark gap/shadow or fabric boundary nearby to avoid normal skin photos.
        pad_check = 18
        x1 = max(0, x - pad_check)
        y1 = max(0, y - pad_check)
        x2 = min(iw, x + w + pad_check)
        y2 = min(ih, y + h + pad_check)
        dark_ratio = float(np.mean(gray[y1:y2, x1:x2] < 105))
        if dark_ratio < 0.015:
            continue

        pad = 12
        bx1 = max(0, x - pad)
        by1 = max(0, y - pad)
        bx2 = min(iw, x + w + pad)
        by2 = min(ih, y + h + pad)
        bw, bh = bx2 - bx1, by2 - by1
        candidates.append({
            "x": int(bx1), "y": int(by1), "w": int(bw), "h": int(bh),
            "bbox": (int(bx1), int(by1), int(bw), int(bh)),
            "center": [int((bx1 + bx2) / 2), int((by1 + by2) / 2)],
            "location": f"X={int((bx1 + bx2) / 2)}, Y={int((by1 + by2) / 2)}",
            "area": int(bw * bh),
            "confidence": 0.82,
            "skin_ratio": round(float(np.mean(skin_mask[y:y+h, x:x+w] > 0)), 4),
            "dark_ratio": round(dark_ratio, 4),
            "opening_refined": True,
            "source": "Rule based finger/opening detector"
        })

    candidates = sorted(candidates, key=lambda it: (it["confidence"], it["area"]), reverse=True)
    return candidates[:1]


def _add_relaxed_skin_opening(result, scan_img):
    openings = _find_relaxed_skin_opening(scan_img)
    for op in openings:
        hole_item = dict(op)
        hole_item["defect_type"] = "hole"
        hole_item["class"] = "hole/finger-opening"
        hole_item["display_note"] = "Finger visible through fabric opening; counted as Hole"
        result = _append_unique_defect(result, "holes", hole_item)

        tear_item = dict(op)
        tear_item["defect_type"] = "tear"
        tear_item["class"] = "tear/kilichu"
        tear_item["display_note"] = "Finger visible through torn opening; counted as Tear/Kilichu"
        result = _append_unique_defect(result, "tears", tear_item)

    result["confidence_scores"]["hole"] = max([float(h.get("confidence", 0.03)) for h in result.get("holes", [])], default=0.03)
    result["confidence_scores"]["tear"] = max([float(t.get("confidence", 0.03)) for t in result.get("tears", [])], default=0.03)
    return result


def _find_white_background_openings(scan_img):
    """
    Backup for white-background sample images with many torn openings.
    It detects all grey/dark inner hole regions while ignoring red circles/text labels.
    This is used only when the image has a clear white/light background.
    """
    ih, iw = scan_img.shape[:2]
    hsv = cv2.cvtColor(scan_img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(scan_img, cv2.COLOR_RGB2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Only enable for white/light sample images, not normal fabric photos.
    white_ratio = float(np.mean((gray > 220) & (sat < 70)))
    if white_ratio < 0.22:
        return []

    # Grey/dark non-white inner openings.
    mask = ((gray < 218) & (sat < 110)).astype(np.uint8) * 255

    # Remove colored annotations: red/orange circles/text/arrows.
    red_or_orange = (((hsv[:, :, 0] < 18) | (hsv[:, :, 0] > 165)) & (sat > 90) & (val > 60))
    mask[red_or_orange] = 0

    # Ignore top label/title area.
    mask[:int(ih * 0.08), :] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = iw * ih
    items = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if w < 18 or h < 22:
            continue
        if area < max(420, img_area * 0.0016):
            continue
        if area > img_area * 0.18:
            continue
        if x <= 1 or y <= int(ih * 0.08) or x + w >= iw - 1 or y + h >= ih - 1:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 5.2:
            continue

        # The area around a true cut/opening has some light/white torn edge/background nearby.
        pad_check = 20
        x1 = max(0, x - pad_check)
        y1 = max(0, y - pad_check)
        x2 = min(iw, x + w + pad_check)
        y2 = min(ih, y + h + pad_check)
        around = scan_img[y1:y2, x1:x2]
        around_hsv = cv2.cvtColor(around, cv2.COLOR_RGB2HSV)
        around_gray = cv2.cvtColor(around, cv2.COLOR_RGB2GRAY)
        local_white = float(np.mean((around_gray > 218) & (around_hsv[:, :, 1] < 80)))
        if local_white < 0.12:
            continue

        pad = 8
        bx1 = max(0, x - pad)
        by1 = max(0, y - pad)
        bx2 = min(iw, x + w + pad)
        by2 = min(ih, y + h + pad)
        bw, bh = bx2 - bx1, by2 - by1
        conf = 0.79 if area >= img_area * 0.006 else 0.64
        items.append({
            "x": int(bx1), "y": int(by1), "w": int(bw), "h": int(bh),
            "bbox": (int(bx1), int(by1), int(bw), int(bh)),
            "center": [int((bx1 + bx2) / 2), int((by1 + by2) / 2)],
            "location": f"X={int((bx1 + bx2) / 2)}, Y={int((by1 + by2) / 2)}",
            "area": int(bw * bh),
            "confidence": round(conf, 4),
            "opening_refined": True,
            "source": "Rule based multi-hole detector for white-background samples"
        })

    # Merge very close components but keep separate real openings.
    items = sorted(items, key=lambda it: it["area"], reverse=True)
    cleaned = []
    for it in items:
        if not any(_is_same_area(it, old, iou_thr=0.10, center_thr=22) for old in cleaned):
            cleaned.append(it)
    return cleaned[:12]


def _add_white_background_openings(result, scan_img):
    openings = _find_white_background_openings(scan_img)
    for op in openings:
        hole_item = dict(op)
        hole_item["defect_type"] = "hole"
        hole_item["class"] = "hole/opening"
        hole_item["display_note"] = "Visible hole/opening detected in sample image"
        result = _append_unique_defect(result, "holes", hole_item)

        tear_item = dict(op)
        tear_item["defect_type"] = "tear"
        tear_item["class"] = "tear/kilichu"
        tear_item["display_note"] = "Opening is part of torn/kilichu cloth"
        result = _append_unique_defect(result, "tears", tear_item)

    result["confidence_scores"]["hole"] = max([float(h.get("confidence", 0.03)) for h in result.get("holes", [])], default=0.03)
    result["confidence_scores"]["tear"] = max([float(t.get("confidence", 0.03)) for t in result.get("tears", [])], default=0.03)
    return result


def _limit_and_clean_defects(result, max_each=6):
    """Clean duplicate outputs and keep strongest boxes."""
    for key in ["holes", "stains", "tears"]:
        items = sorted(result.get(key, []), key=lambda it: float(it.get("confidence", 0)) * max(1, int(it.get("area", 1))), reverse=True)
        cleaned = []
        for item in items:
            if not any(_is_same_area(item, old, iou_thr=0.18, center_thr=28) for old in cleaned):
                cleaned.append(item)
            if len(cleaned) >= max_each:
                break
        for idx, item in enumerate(cleaned, 1):
            item["id"] = idx
        result[key] = cleaned
    return result

def yolo_scan(image):
    """
    Local trained YOLO model scan.
    Keeps the existing UI flow same:
    one image -> separate scanner outputs -> final single verdict.
    """
    start = time.time()
    model = load_yolo_model()

    original_img = pil_to_rgb_np(image)
    scan_img = cv2.resize(original_img, (512, 512))

    empty_result = {
        "scan_image": scan_img,
        "holes": [],
        "stains": [],
        "tears": [],
        "horizontal": [],
        "vertical": [],
        "lines": [],
        "confidence_scores": {
            "hole": 0.03,
            "stain": 0.03,
            "tear": 0.03,
            "horizontal": 0.03,
            "vertical": 0.03,
            "lines": 0.03,
            "defect_free": 0.97,
        },
        "is_defect": False,
        "final_result": "ACCEPT",
        "main_defect": "Defect Free",
        "final_class": "Defect Free",
        "defect_type": "Defect Free",
        "severity": "GOOD",
        "confidence": 0.97,
        "defect_score": 0.03,
        "final_status": "ACCEPT - Ready for production",
        "decision_reason": "No defect detected by YOLO model",
        "processing_time": round(time.time() - start, 3),
        "detected_locations": [],
    }

    if model is None:
        empty_result["final_result"] = "MODEL MISSING"
        empty_result["main_defect"] = "best.pt file missing"
        empty_result["final_status"] = "Please place best.pt in the same folder as main_app.py"
        empty_result["decision_reason"] = "YOLO model file not found"
        return empty_result

    predictions = model.predict(scan_img, conf=YOLO_CONFIDENCE, imgsz=640, verbose=False)
    if not predictions:
        return empty_result

    pred = predictions[0]
    class_names = model.names if hasattr(model, "names") else {}

    result = empty_result
    result["defect_free"] = []

    for box_index, box in enumerate(pred.boxes, 1):
        cls_id = int(box.cls[0])
        raw_class = class_names.get(cls_id, str(cls_id))
        defect_key = normalize_defect_class(raw_class)
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(512, x2)), int(min(512, y2))
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        center_x, center_y = int(x1 + w / 2), int(y1 + h / 2)
        area = int(w * h)

        item = {
            "id": len(result.get(defect_key + "s", [])) + 1,
            "class": raw_class,
            "defect_type": defect_key,
            "confidence": round(confidence, 4),
            "x": x1,
            "y": y1,
            "w": w,
            "h": h,
            "bbox": (x1, y1, w, h),
            "center": [center_x, center_y],
            "location": f"X={center_x}, Y={center_y}",
            "area": area,
            "source": "YOLOv8 trained best.pt",
        }

        if defect_key == "ignore":
            continue

        if defect_key == "hole":
            result["holes"].append(item)
            result["confidence_scores"]["hole"] = max(result["confidence_scores"]["hole"], confidence)
        elif defect_key == "stain":
            result["stains"].append(item)
            result["confidence_scores"]["stain"] = max(result["confidence_scores"]["stain"], confidence)
        elif defect_key == "tear":
            result["tears"].append(item)
            result["confidence_scores"]["tear"] = max(result["confidence_scores"]["tear"], confidence)

        result["detected_locations"].append({
            "S.No": len(result["detected_locations"]) + 1,
            "Scanner": {
                "hole": "Hole Scanner",
                "stain": "Stain Scanner",
                "tear": "Tear/Kilichu Scanner",
            }.get(defect_key, "Defect Scanner"),
            "Class": raw_class,
            "Confidence": f"{confidence * 100:.1f}%",
            "X": center_x,
            "Y": center_y,
            "Width": w,
            "Height": h,
            "Location": f"X={center_x}, Y={center_y}",
        })

    # Convert/refine YOLO output and add rule-based backup for missed visible openings.
    result = postprocess_hole_tear_items(result, scan_img)
    result = _add_rule_based_visible_openings(result, scan_img)
    result = _limit_and_clean_defects(result, max_each=6)

    # Reassign ids after grouping
    for key in ["holes", "stains", "tears"]:
        for idx, item in enumerate(result[key], 1):
            item["id"] = idx
            x, y, w, h = _bbox_from_item(item)
            item["center"] = [int(x + w / 2), int(y + h / 2)]
            item["location"] = f"X={item['center'][0]}, Y={item['center'][1]}"

    result = _rebuild_detected_locations(result)

    total_defects = len(result["holes"]) + len(result["stains"]) + len(result["tears"])
    result["holes_count"] = len(result["holes"])
    result["stains_count"] = len(result["stains"])
    result["tears_count"] = len(result["tears"])
    result["horizontal_count"] = 0
    result["vertical_count"] = 0
    result["lines_count"] = 0

    if total_defects > 0:
        result["is_defect"] = True
        result["final_result"] = "REJECT"
        result["confidence_scores"]["defect_free"] = 0.03

        class_conf = {
            "Hole": result["confidence_scores"]["hole"],
            "Stain": result["confidence_scores"]["stain"],
            "Tear/Kilichu": result["confidence_scores"]["tear"],
        }
        main_defect = max(class_conf, key=class_conf.get)
        max_conf = class_conf[main_defect]

        result["main_defect"] = main_defect
        result["final_class"] = main_defect
        result["defect_type"] = main_defect
        result["severity"] = "CRITICAL" if max_conf >= 0.75 else "HIGH" if max_conf >= 0.50 else "MEDIUM"
        result["confidence"] = max_conf
        result["defect_score"] = max_conf
        result["final_status"] = "REJECT - Defect detected by local YOLO model"
        result["decision_reason"] = f"{total_defects} defect(s) detected using trained best.pt model"
    else:
        result["confidence_scores"]["defect_free"] = 0.97

    result["scan_steps"] = [
        {"scanner": "Hole Scanner", "count": len(result["holes"]), "confidence": result["confidence_scores"]["hole"], "status": "FAIL" if result["holes"] else "PASS"},
        {"scanner": "Stain Scanner", "count": len(result["stains"]), "confidence": result["confidence_scores"]["stain"], "status": "FAIL" if result["stains"] else "PASS"},
        {"scanner": "Tear/Kilichu Scanner", "count": len(result["tears"]), "confidence": result["confidence_scores"]["tear"], "status": "FAIL" if result["tears"] else "PASS"},
        {"scanner": "Defect Free Scanner", "count": 1 if total_defects == 0 else 0, "confidence": result["confidence_scores"]["defect_free"], "status": "PASS" if total_defects == 0 else "FAIL"},
    ]

    result["processing_time"] = round(time.time() - start, 3)
    return result


# ============================================
# IMAGE HELPERS
# ============================================

def pil_to_rgb_np(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    img = np.array(image)

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    return img


def check_image_quality(image):
    img = pil_to_rgb_np(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    brightness = gray.mean()

    score = 50
    issues = []

    if laplacian_var < 100:
        issues.append("Image is blurry")
        score -= 30
    elif laplacian_var < 300:
        issues.append("Image is slightly blurry")
        score -= 10
    else:
        score += 20

    if contrast < 30:
        issues.append("Low contrast")
        score -= 20
    elif contrast >= 50:
        score += 10

    if brightness < 50:
        issues.append("Too dark")
        score -= 20
    elif brightness > 210:
        issues.append("Too bright / overexposed")
        score -= 20
    else:
        score += 10

    score = max(0, min(100, score))

    return {
        "score": score,
        "blur_metric": laplacian_var,
        "contrast": contrast,
        "brightness": brightness,
        "issues": issues,
        "is_good": score > 50
    }


def draw_single_detection(scan_img, items, defect_type):
    """
    Draw only one defect type on image:
    hole / stain / tear / horizontal / vertical / lines
    """
    canvas = scan_img.copy()

    style = {
        "hole": {"color": (255, 0, 0), "prefix": "H"},
        "stain": {"color": (0, 200, 0), "prefix": "S"},
        "tear": {"color": (255, 165, 0), "prefix": "K"},
        "horizontal": {"color": (0, 220, 220), "prefix": "HR"},
        "vertical": {"color": (170, 0, 255), "prefix": "V"},
        "lines": {"color": (0, 80, 255), "prefix": "L"},
    }

    color = style.get(defect_type, style["lines"])["color"]
    prefix = style.get(defect_type, style["lines"])["prefix"]

    for idx, item in enumerate(items, 1):
        label = f"{prefix}{item.get('id', idx)}"

        if "bbox" in item:
            x, y, w, h = item.get("bbox", (0, 0, 0, 0))
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                canvas,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2
            )

        elif all(k in item for k in ("x", "y", "w", "h")):
            x, y, w, h = int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                canvas,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2
            )

        elif "line" in item:
            x1, y1, x2, y2 = item["line"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                canvas,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )

        elif all(k in item for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])
            cv2.line(canvas, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                canvas,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )

    return canvas


def get_items(result, key):
    data = result.get(key, [])
    if isinstance(data, list):
        return data
    return []


def _bbox_from_item(item):
    if "bbox" in item:
        x, y, w, h = item.get("bbox", (0, 0, 0, 0))
        return int(x), int(y), int(w), int(h)
    if all(k in item for k in ("x", "y", "w", "h")):
        return int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])
    return 0, 0, 0, 0


def _iou(a, b):
    ax, ay, aw, ah = _bbox_from_item(a)
    bx, by, bw, bh = _bbox_from_item(b)

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6

    return inter / union


def _merge_close_boxes(items, image_shape):
    if not items:
        return []

    h, w = image_shape[:2]
    boxes = []
    for item in items:
        x, y, bw, bh = _bbox_from_item(item)
        if bw <= 3 or bh <= 3:
            continue
        if bw * bh > (w * h * 0.35):
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue

        copied = dict(item)
        copied["bbox"] = (x, y, bw, bh)
        copied["x"], copied["y"], copied["w"], copied["h"] = x, y, bw, bh
        boxes.append(copied)

    used = [False] * len(boxes)
    merged = []

    for i, item in enumerate(boxes):
        if used[i]:
            continue

        group = [item]
        used[i] = True

        changed = True
        while changed:
            changed = False
            gx1 = min(_bbox_from_item(g)[0] for g in group)
            gy1 = min(_bbox_from_item(g)[1] for g in group)
            gx2 = max(_bbox_from_item(g)[0] + _bbox_from_item(g)[2] for g in group)
            gy2 = max(_bbox_from_item(g)[1] + _bbox_from_item(g)[3] for g in group)

            for j, other in enumerate(boxes):
                if used[j]:
                    continue

                ox, oy, ow, oh = _bbox_from_item(other)
                ocx, ocy = ox + ow / 2, oy + oh / 2
                gap_x = max(0, max(gx1 - (ox + ow), ox - gx2))
                gap_y = max(0, max(gy1 - (oy + oh), oy - gy2))
                close = gap_x < 18 and gap_y < 18

                # Merge overlapping / touching boxes; this removes repeated false boxes around one hole.
                if _iou(group[0], other) > 0.05 or close:
                    group.append(other)
                    used[j] = True
                    changed = True

        x1 = min(_bbox_from_item(g)[0] for g in group)
        y1 = min(_bbox_from_item(g)[1] for g in group)
        x2 = max(_bbox_from_item(g)[0] + _bbox_from_item(g)[2] for g in group)
        y2 = max(_bbox_from_item(g)[1] + _bbox_from_item(g)[3] for g in group)

        pad = 4
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        area = int((x2 - x1) * (y2 - y1))
        merged.append({
            "id": len(merged) + 1,
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "area": area,
            "severity": group[0].get("severity", "HIGH"),
            "source": "human_like_grouped_display"
        })

    return merged


def _hole_score(scan_img, item):
    x, y, w, h = _bbox_from_item(item)
    ih, iw = scan_img.shape[:2]
    if w <= 3 or h <= 3:
        return -999999

    gray = cv2.cvtColor(scan_img, cv2.COLOR_RGB2GRAY)
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return -999999

    # Slight outside area for local contrast comparison
    pad = max(8, int(max(w, h) * 0.35))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad)
    y2 = min(ih, y + h + pad)
    outer = gray[y1:y2, x1:x2]

    roi_mean = float(np.mean(roi))
    outer_mean = float(np.mean(outer)) if outer.size else roi_mean
    roi_std = float(np.std(roi))
    dark_ratio = float(np.mean(roi < max(50, outer_mean - 28)))

    area = w * h
    area_score = min(area / 4500.0, 1.5)
    contrast_score = max(0.0, (outer_mean - roi_mean) / 70.0)
    dark_score = dark_ratio * 2.0

    # Prefer moderate real holes; penalize huge texture blocks and tiny noise.
    penalty = 0
    if area < 120:
        penalty += 2.0
    if area > iw * ih * 0.12:
        penalty += 3.0
    if w / max(h, 1) > 6 or h / max(w, 1) > 6:
        penalty += 1.0

    return area_score + contrast_score + dark_score + roi_std / 150.0 - penalty


def refine_holes_human_like(scan_img, holes, result):
    """
    UI-only post-process:
    - removes zero/tiny/huge false boxes
    - merges repeated boxes around the same defect
    - for hand-through-hole images, shows one primary opening
    - for complex fabric, limits false positives and marks manual check via result flag
    """
    if not holes:
        return [], False, "No hole candidates"

    merged = _merge_close_boxes(holes, scan_img.shape)
    if not merged:
        return [], False, "No valid hole candidate after filtering"

    for item in merged:
        item["_score"] = _hole_score(scan_img, item)

    merged = sorted(merged, key=lambda it: it.get("_score", 0), reverse=True)

    # Hand/skin visible case from hole_scanner should display one primary hole only.
    unknown_count = int(result.get("unknown_count", 0))
    if unknown_count > 0:
        best = dict(merged[0])
        best["id"] = 1
        best["display_note"] = "Primary opening selected because hand/skin is visible"
        best.pop("_score", None)
        return [best], True, "Hand / skin visible inside fabric area"

    # If many candidates are found, this is usually mesh/texture/annotation confusion.
    # Show max 2 strongest candidates and ask manual verification instead of claiming all are holes.
    manual_check = len(merged) > 2
    display_count = 2 if manual_check else len(merged)
    final_items = []

    for idx, item in enumerate(merged[:display_count], 1):
        item = dict(item)
        item["id"] = idx
        item["display_note"] = "Human-like filtered hole candidate"
        item.pop("_score", None)
        final_items.append(item)

    reason = "Multiple possible holes; complex texture/manual verification required" if manual_check else "Filtered hole candidate"
    return final_items, manual_check, reason


def scanner_card(title, icon, count, confidence, status, image, color):
    status_text = "FAIL" if status else "PASS"
    status_class = "fail-pill" if status else "pass-pill"

    st.markdown(
        f"""
        <div class="scanner-card" style="border-color:{color};">
            <div class="scanner-title" style="color:{color};">
                <span>{icon}</span> {title}
                <span class="{status_class}">{status_text}</span>
            </div>
            <div class="scanner-metrics">
                <div>
                    <small>Count</small>
                    <h2>{count}</h2>
                </div>
                <div>
                    <small>Confidence</small>
                    <h2>{confidence * 100:.1f}%</h2>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(image, use_container_width=True)

# ============================================
# STYLE
# ============================================

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .main-header {
        text-align: center;
        padding: 2rem;
        color: white;
    }

    .top-bar {
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:1rem 2rem;
        background:white;
        border-radius:16px;
        margin-bottom:1rem;
        box-shadow:0 6px 18px rgba(0,0,0,0.12);
    }

    .metric-card {
        background:white;
        border-radius:16px;
        padding:1rem;
        text-align:center;
        box-shadow:0 2px 8px rgba(0,0,0,0.1);
        min-height:100px;
    }

    .defect-card {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border-radius:20px;
        padding:1.5rem;
        color:white;
        margin:1rem 0;
        box-shadow:0 8px 24px rgba(0,0,0,0.18);
    }

    .success-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius:20px;
        padding:1.5rem;
        color:white;
        margin:1rem 0;
        box-shadow:0 8px 24px rgba(0,0,0,0.18);
    }

    .scanner-card {
        background:white;
        border:2px solid;
        border-radius:16px;
        padding:1rem;
        margin:0.5rem 0;
        box-shadow:0 3px 12px rgba(0,0,0,0.08);
    }

    .scanner-title {
        font-weight:800;
        font-size:19px;
        display:flex;
        justify-content:space-between;
        align-items:center;
    }

    .scanner-metrics {
        margin-top:12px;
        border:1px solid #e8e8e8;
        border-radius:12px;
        display:grid;
        grid-template-columns:1fr 1fr;
        overflow:hidden;
    }

    .scanner-metrics div {
        padding:12px;
        text-align:center;
        border-right:1px solid #e8e8e8;
    }

    .scanner-metrics div:last-child {
        border-right:none;
    }

    .scanner-metrics h2 {
        margin:0;
        font-weight:800;
    }

    .pass-pill {
        background:#d4edda;
        color:#087f23;
        padding:5px 11px;
        border-radius:10px;
        font-size:13px;
        border:1px solid #28a745;
    }

    .fail-pill {
        background:#f8d7da;
        color:#b00020;
        padding:5px 11px;
        border-radius:10px;
        font-size:13px;
        border:1px solid #dc3545;
    }

    .info-box {
        background:#f8f9fa;
        border-left:4px solid #007bff;
        padding:1rem;
        margin:1rem 0;
        border-radius:8px;
    }

    .final-box {
        background:white;
        border-radius:18px;
        padding:1.5rem;
        box-shadow:0 6px 20px rgba(0,0,0,0.16);
        margin-top:1rem;
    }
</style>
""",
    unsafe_allow_html=True
)

# ============================================
# AUTH / HOME
# ============================================

def home_page():
    st.markdown(
        """
        <div class="main-header">
            <h1>🔍 Professional Textile Defect Detection</h1>
            <p>One Image → Separate Scanners → Final Single Output</p>
            <p><small>Hole | Stain | Tear/Kilichu | Defect Free</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            '<div style="text-align:center;color:white;"><h3>Sign In</h3></div>',
            unsafe_allow_html=True
        )

        oauth2 = OAuth2Component(
            CLIENT_ID,
            CLIENT_SECRET,
            AUTHORIZE_URL,
            TOKEN_URL,
            REFRESH_TOKEN_URL,
            REVOKE_TOKEN_URL
        )

        result = oauth2.authorize_button(
            name="Continue with Google",
            icon="https://www.google.com/favicon.ico",
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="google_oauth",
            use_container_width=True,
        )

        if result and "token" in result:
            access_token = result["token"]["access_token"]
            user_info_response = requests.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if user_info_response.status_code == 200:
                user_info = user_info_response.json()
                user_email = user_info.get("email", "")
                user_name = user_info.get("name", user_email.split("@")[0])

                user_data = {
                    "email": user_email,
                    "name": user_name,
                    "role": "admin" if user_email in ADMINS else "user",
                    "created_at": datetime.now().isoformat(),
                }
                save_user(user_data)

                st.session_state["logged_in"] = True
                st.session_state["user_email"] = user_email
                st.session_state["user_name"] = user_name
                st.session_state["user_role"] = user_data["role"]
                st.rerun()

# ============================================
# DASHBOARD
# ============================================

def user_dashboard():
    st.markdown(
        f"""
        <div class="top-bar">
            <div>
                <span style="font-size:24px;font-weight:800;">🧵 Textile Defect Detection</span>
                <span style="background:#28a745;color:white;padding:3px 10px;border-radius:12px;font-size:12px;">MULTI SCANNER</span>
            </div>
            <div>Welcome, <b>{st.session_state.get("user_name", "User")}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Logout", key="logout_btn"):
        for key in ["logged_in", "user_email", "user_name", "user_role", "current_image"]:
            st.session_state[key] = DEFAULT_SESSION.get(key, None)
        st.rerun()

    reports = get_user_reports(st.session_state["user_email"])
    total = len(reports)
    defects = sum(1 for r in reports if r.get("is_defect", False))
    defect_rate = (defects / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h1>{total}</h1><p>Inspections</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h1 style="color:#dc3545;">{defects}</h1><p>Defects</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h1>{defect_rate:.1f}%</h1><p>Defect Rate</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h1>512px</h1><p>Scan Mode</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.2, 0.8])

    with left:
        st.subheader("📸 Upload Fabric Image")
        st.caption("One image upload pannunga. App separate-a Hole/Stain/Tear scan pannum.")

        tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])
        captured_image = None

        with tab1:
            uploaded = st.file_uploader(
                "Choose image",
                type=["jpg", "jpeg", "png", "bmp", "tiff"]
            )
            if uploaded:
                captured_image = Image.open(uploaded)
                st.image(captured_image, use_container_width=True)

        with tab2:
            camera = st.camera_input("Take a photo")
            if camera:
                captured_image = Image.open(camera)
                st.image(captured_image, use_container_width=True)

        if captured_image:
            if st.button("🔍 Analyze Fabric", type="primary", use_container_width=True):
                st.session_state["current_image"] = captured_image
                st.rerun()

    with right:
        st.subheader("💡 Scanning Process")
        st.markdown(
            """
            <div class="info-box">
            <b>1.</b> Image Preprocessing<br>
            <b>2.</b> Hole Scanner<br>
            <b>3.</b> Stain Scanner<br>
            <b>4.</b> Tear/Kilichu Scanner<br>
            <b>5.</b> Final Single Output
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    if st.session_state.get("current_image") is not None:
        show_analysis(st.session_state["current_image"])


def show_analysis(image):
    st.subheader("📊 Analysis Results")

    quality = check_image_quality(image)

    with st.spinner("🔬 Running all scanners separately using trained YOLO model..."):
        result = yolo_scan(image)

    if result.get("final_result") == "MODEL MISSING":
        st.error("❌ best.pt file missing. Please keep best.pt in the same folder as main_app.py")
        return

    if not quality["is_good"]:
        st.warning(f"⚠️ Image Quality: {quality['score']:.0f}% - {', '.join(quality['issues'])}")

    scan_img = result.get("scan_image")
    if scan_img is None:
        scan_img = cv2.resize(pil_to_rgb_np(image), (512, 512))

    holes = get_items(result, "holes")
    stains = get_items(result, "stains")
    tears = get_items(result, "tears")
    horizontals = get_items(result, "horizontal")
    verticals = get_items(result, "vertical")
    lines = get_items(result, "lines")

    hole_manual_check = False
    scores = result.get("confidence_scores", {})

    is_defect = result.get("is_defect", result.get("final_result") == "REJECT")
    final_result = result.get("final_result", "REJECT" if is_defect else "ACCEPT")
    main_defect = result.get("main_defect", result.get("final_class", "Defect Free"))

    # Save report
    report_data = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "user_email": st.session_state["user_email"],
        "timestamp": datetime.now().isoformat(),
        "is_defect": is_defect,
        "final_result": final_result,
        "main_defect": main_defect,
        "defect_type": result.get("defect_type", main_defect),
        "severity": result.get("severity", "GOOD"),
        "confidence": result.get("confidence", 0.0),
        "holes_count": len(holes),
        "stains_count": result.get("stains_count", len(stains)),
        "tears_count": result.get("tears_count", len(tears)),
        "horizontal_count": result.get("horizontal_count", len(horizontals)),
        "vertical_count": result.get("vertical_count", len(verticals)),
        "lines_count": result.get("lines_count", len(lines)),
        "processing_time": result.get("processing_time", 0),
        "confidence_scores": scores,
    }
    save_report(report_data)
    st.session_state["analysis_history"].insert(0, report_data)

    # Final top card
    if is_defect:
        st.markdown(
            f"""
            <div class="defect-card">
                <h2>❌ FINAL SINGLE OUTPUT: REJECT</h2>
                <h3>Main Defect: {main_defect}</h3>
                <p><b>Severity:</b> {result.get("severity", "CRITICAL")}</p>
                <p><b>Confidence:</b> {result.get("confidence", 0) * 100:.1f}%</p>
                <p><b>Status:</b> {result.get("final_status", "REJECT - Do NOT use for production")}</p>
                <p><b>Reason:</b> {result.get("decision_reason", "Defect detected by scanner")}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="success-card">
                <h2>✅ FINAL SINGLE OUTPUT: ACCEPT</h2>
                <h3>Main Result: Defect Free</h3>
                <p><b>Confidence:</b> {result.get("confidence", 0) * 100:.1f}%</p>
                <p><b>Status:</b> ACCEPT - Ready for production</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.balloons()

    # Original and processed image
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("### Original Image")
        st.image(image, use_container_width=True)
    with img_col2:
        st.markdown("### Preprocessed 512×512 Image")
        st.image(scan_img, use_container_width=True)

    st.markdown("## 🧪 Separate Scanner Outputs")

    # Separate scanner images
    hole_img = draw_single_detection(scan_img, holes, "hole")
    stain_img = draw_single_detection(scan_img, stains, "stain")
    tear_img = draw_single_detection(scan_img, tears, "tear")
    horizontal_img = draw_single_detection(scan_img, horizontals, "horizontal")
    vertical_img = draw_single_detection(scan_img, verticals, "vertical")
    lines_img = draw_single_detection(scan_img, lines, "lines")

    row1 = st.columns(3)
    with row1[0]:
        scanner_card("1. Hole Scanner", "🔴", len(holes), scores.get("hole", 0.03), len(holes) > 0, hole_img, "#ff2d2d")
    with row1[1]:
        scanner_card("2. Stain Scanner", "🟢", len(stains), scores.get("stain", 0.03), len(stains) > 0, stain_img, "#0a9f3d")
    with row1[2]:
        scanner_card("3. Tear/Kilichu Scanner", "🟠", len(tears), scores.get("tear", 0.03), len(tears) > 0, tear_img, "#ff7a00")

    # Defect free scanner
    st.markdown("## ✅ Defect Free Scanner")
    df_count = 1 if final_result == "ACCEPT" else 0
    scanner_card(
        "7. Defect Free Scanner",
        "✅",
        df_count,
        scores.get("defect_free", 0.03),
        final_result != "ACCEPT",
        scan_img,
        "#28a745"
    )

    # Detailed table
    st.markdown("## 📋 Scanner Table")
    scan_steps = result.get("scan_steps", [])
    scan_df = pd.DataFrame([
        {
            "Scanner": s.get("scanner", ""),
            "Count": s.get("count", 0),
            "Confidence": f"{s.get('confidence', 0) * 100:.1f}%",
            "Status": s.get("status", "PASS"),
        }
        for s in scan_steps
    ])
    st.dataframe(scan_df, use_container_width=True, hide_index=True)

    # Location table
    st.markdown("## 📍 Defect Location Table")
    locations = result.get("detected_locations", [])
    if locations:
        st.dataframe(pd.DataFrame(locations), use_container_width=True, hide_index=True)
    else:
        st.success("No defect locations found.")

    # Confidence row
    st.markdown("## 📊 Confidence Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Hole %", f"{scores.get('hole', 0) * 100:.0f}%")
    with c2:
        st.metric("Stain %", f"{scores.get('stain', 0) * 100:.0f}%")
    with c3:
        st.metric("Tear/Kilichu %", f"{scores.get('tear', 0) * 100:.0f}%")
    with c4:
        st.metric("Defect Free %", f"{scores.get('defect_free', 0) * 100:.0f}%")

    # Detailed counts
    with st.expander("🔴 Hole Details", expanded=len(holes) > 0):
        if holes:
            st.json(holes)
        else:
            st.success("No holes detected.")

    with st.expander("🟢 Stain Details", expanded=len(stains) > 0):
        if stains:
            st.json(stains)
        else:
            st.success("No stains detected.")

    with st.expander("🟠 Tear/Kilichu Details", expanded=len(tears) > 0):
        if tears:
            st.json(tears)
        else:
            st.success("No tear/kilichu detected.")

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(result.get("defect_score", 0.06)) * 100,
        title={"text": "Final Defect Severity Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#dc3545" if is_defect else "#28a745"},
            "steps": [
                {"range": [0, 30], "color": "#d4edda"},
                {"range": [30, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 70
            }
        }
    ))
    fig.update_layout(height=260)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 New Inspection", use_container_width=True):
        st.session_state["current_image"] = None
        st.rerun()


# ============================================
# ADMIN DASHBOARD
# ============================================

def admin_dashboard():
    st.title("👑 Admin Dashboard")

    if st.button("Logout"):
        for key in ["logged_in", "user_email", "user_name", "user_role", "current_image"]:
            st.session_state[key] = DEFAULT_SESSION.get(key, None)
        st.rerun()

    users = get_all_users()
    reports = get_all_reports()

    defects = sum(1 for r in reports if r.get("is_defect", False))
    rate = (defects / len(reports) * 100) if reports else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", len(users))
    c2.metric("Total Inspections", len(reports))
    c3.metric("Defects Found", defects)
    c4.metric("Defect Rate", f"{rate:.1f}%")

    st.subheader("User Management")
    if users:
        user_df = pd.DataFrame([
            {
                "Email": email,
                "Name": data.get("name", "N/A"),
                "Role": "Admin" if email in ADMINS else "User",
                "Joined": data.get("created_at", "N/A")[:10],
            }
            for email, data in users.items()
        ])
        st.dataframe(user_df, use_container_width=True)

    st.subheader("Recent Inspections")
    if reports:
        report_df = pd.DataFrame([
            {
                "User": r.get("user_email", "N/A")[:25],
                "Time": r.get("timestamp", "")[:16],
                "Final": r.get("final_result", "N/A"),
                "Main Defect": r.get("main_defect", r.get("defect_type", "N/A")),
                "Holes": r.get("holes_count", 0),
                "Stains": r.get("stains_count", 0),
                "Tears": r.get("tears_count", 0),
                "Horizontal": r.get("horizontal_count", 0),
                "Vertical": r.get("vertical_count", 0),
                "Lines": r.get("lines_count", 0),
                "Time(s)": r.get("processing_time", 0),
            }
            for r in reports[-50:]
        ])
        st.dataframe(report_df, use_container_width=True)

        csv = report_df.to_csv(index=False)
        st.download_button("📥 Download Reports CSV", csv, "reports.csv", "text/csv")

# ============================================
# MAIN
# ============================================

def main():
    if not st.session_state.get("logged_in"):
        home_page()
    else:
        if st.session_state.get("user_role") == "admin":
            admin_dashboard()
        else:
            user_dashboard()


if __name__ == "__main__":
    main()
