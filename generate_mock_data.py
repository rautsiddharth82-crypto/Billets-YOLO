"""
Automated Billet Quality Inspection System - Test Data Generator
================================================================
Generates calibrated, high-fidelity synthetic billet images for validation,
unit testing, and hardware-free demonstration.

Simulates:
- Continuous cast steel texture with thermal scaling
- Stamped heat numbers on billet end faces (e.g., 'H727622R1.0')
- 60-degree angle perspective projection
- NEU-DET surface defect archetypes (crazing, scratches, inclusions, patches)
- Dimension variations (IS 2830 compliance and violations)
"""

import os
import math
import random
import cv2
import numpy as np

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "test")


def create_realistic_billet_frame(
    width_mm=125.0,
    height_mm=125.0,
    heat_number="H727622R1.0",
    defect_type=None,
    angle_deg=60.0,
    pixels_per_mm=2.77
):
    """
    Creates a full 1920x1080 camera frame simulating the 60-degree industrial camera view.
    """
    # 1. Industrial conveyor table background (dark matte charcoal)
    frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 28

    # 2. Compute 60-degree perspective foreshortened dimensions
    theta = math.radians(angle_deg)
    w_meas_mm = width_mm * math.cos(theta)
    h_meas_mm = height_mm * math.sin(theta)

    w_px = int(round(w_meas_mm * pixels_per_mm))
    h_px = int(round(h_meas_mm * pixels_per_mm))

    # Center position of billet end face
    cx, cy = 960, 540
    x1 = cx - w_px // 2
    y1 = cy - h_px // 2
    x2 = x1 + w_px
    y2 = y1 + h_px

    # Draw billet body extending backward along conveyor (perspective trapezoid)
    top_offset_x = 260
    top_offset_y = -320
    body_pts = np.array([
        [x1, y1],
        [x2, y1],
        [x2 + top_offset_x, y1 + top_offset_y],
        [x1 + top_offset_x, y1 + top_offset_y]
    ], np.int32)
    # Billet top surface color (hot/warm rolled oxidized steel)
    cv2.fillPoly(frame, [body_pts], (80, 85, 95))
    cv2.polylines(frame, [body_pts], True, (55, 60, 68), 2)

    # Draw billet end face (facing camera)
    end_face = np.ones((h_px, w_px, 3), dtype=np.uint8) * 165
    # Steel grain & thermal scale texture
    grain_noise = np.random.normal(0, 8, end_face.shape).astype(np.int16)
    end_face = np.clip(end_face.astype(np.int16) + grain_noise, 0, 255).astype(np.uint8)

    # Crisp boundary around end face
    cv2.rectangle(end_face, (0, 0), (w_px - 1, h_px - 1), (110, 115, 120), 2)

    # Stamped Heat Number on end face
    if heat_number:
        font_scale = max(0.65, min(1.2, w_px / 220.0))
        thickness = 2
        (tw, th), _ = cv2.getTextSize(heat_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = max(8, (w_px - tw) // 2)
        ty = (h_px + th) // 2
        # Crisp stamped text
        cv2.putText(end_face, heat_number, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
        cv2.putText(end_face, heat_number, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 245), thickness, cv2.LINE_AA)

    # Embed end face into frame
    frame[y1:y2, x1:x2] = end_face

    # Draw bevel edge separating end face from top surface
    cv2.line(frame, (x1, y1), (x2, y1), (30, 32, 36), 3)

    # 3. Add Surface Defects (on top surface)
    if defect_type == "crazing":
        # Network of surface thermal stress micro-cracks
        for _ in range(12):
            pt1 = (random.randint(cx - 30, cx + 140), random.randint(cy - 220, cy - 100))
            pt2 = (pt1[0] + random.randint(-40, 40), pt1[1] + random.randint(-40, 40))
            cv2.line(frame, pt1, pt2, (15, 15, 20), 3)
            pt3 = (pt2[0] + random.randint(-25, 25), pt2[1] + random.randint(-25, 25))
            cv2.line(frame, pt2, pt3, (20, 20, 25), 2)

    elif defect_type == "scratches":
        # Longitudinal scratch from guide roll friction
        s_start = (cx - 20, cy - 230)
        s_end = (cx + 160, cy - 80)
        cv2.line(frame, s_start, s_end, (15, 15, 18), 5)
        cv2.line(frame, (s_start[0] + 1, s_start[1] + 1), (s_end[0] + 1, s_end[1] + 1), (180, 185, 190), 1)

    elif defect_type == "inclusion":
        # Non-metallic slag/oxide entrapment
        cv2.circle(frame, (cx + 60, cy - 200), 22, (15, 15, 20), -1)
        cv2.circle(frame, (cx + 60, cy - 200), 24, (50, 50, 55), 2)

    elif defect_type == "patches":
        # Dark mill scale patch
        pts = np.array([
            [cx + 20, cy - 190],
            [cx + 100, cy - 180],
            [cx + 120, cy - 130],
            [cx + 40, cy - 140]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (35, 35, 40))

    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame


def create_side_view_billet_frame(
    height_mm=125.0,
    defect_type=None,
    pixels_per_mm=2.77
):
    """
    Creates a 1920x1080 camera frame simulating the 90-degree orthogonal side camera view.
    Inspects the longitudinal side profile and side surface defects along the conveyor.
    """
    frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 28

    # Side view: Orthogonal (90 deg)
    # Height is directly 125 mm * pixels_per_mm
    h_px = int(round(height_mm * pixels_per_mm))
    cy = 540
    y1 = cy - h_px // 2
    y2 = y1 + h_px

    # Billet extends across the horizontal FOV along conveyor table
    x1 = 120
    x2 = 1800

    # Draw billet side profile (continuous longitudinal steel bar)
    side_face = np.ones((h_px, x2 - x1, 3), dtype=np.uint8) * 95
    grain_noise = np.random.normal(0, 7, side_face.shape).astype(np.int16)
    side_face = np.clip(side_face.astype(np.int16) + grain_noise, 0, 255).astype(np.uint8)

    # Bevel lines on top and bottom edges
    cv2.rectangle(side_face, (0, 0), (x2 - x1 - 1, h_px - 1), (65, 70, 75), 2)
    frame[y1:y2, x1:x2] = side_face

    # Side Surface Defects
    cx_defect = 960
    if defect_type == "scratches":
        # Guide-roll friction scratch running along rolling axis
        cv2.line(frame, (cx_defect - 300, cy - 20), (cx_defect + 350, cy - 15), (20, 20, 25), 4)
        cv2.line(frame, (cx_defect - 300, cy - 19), (cx_defect + 350, cy - 14), (170, 175, 180), 1)
    elif defect_type == "inclusion":
        cv2.circle(frame, (cx_defect, cy), 24, (18, 18, 22), -1)
        cv2.circle(frame, (cx_defect, cy), 26, (55, 60, 65), 2)
    elif defect_type == "crazing":
        for _ in range(15):
            pt1 = (random.randint(cx_defect - 150, cx_defect + 150), random.randint(cy - 80, cy + 80))
            pt2 = (pt1[0] + random.randint(-40, 40), pt1[1] + random.randint(-40, 40))
            cv2.line(frame, pt1, pt2, (20, 20, 25), 2)
    elif defect_type == "patches":
        cv2.rectangle(frame, (cx_defect - 100, cy - 50), (cx_defect + 120, cy + 40), (45, 45, 50), -1)

    return frame


def generate_all_test_billets():
    """Generates complete suites of both front (60°) and side (90°) test billet scenarios."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    scenarios = [
        {
            "filename": "billet_pass_01.jpg",
            "side_filename": "billet_side_pass_01.jpg",
            "width_mm": 125.1,
            "height_mm": 124.9,
            "heat_number": "H727622R1.0",
            "defect_type": None,
            "expected_status": "PASS"
        },
        {
            "filename": "billet_pass_02.jpg",
            "side_filename": "billet_side_pass_02.jpg",
            "width_mm": 126.2,
            "height_mm": 125.5,
            "heat_number": "H727623R1.0",
            "defect_type": None,
            "expected_status": "PASS"
        },
        {
            "filename": "billet_fail_crazing.jpg",
            "side_filename": "billet_side_fail_crazing.jpg",
            "width_mm": 125.0,
            "height_mm": 125.0,
            "heat_number": "H727624R2.1",
            "defect_type": "crazing",
            "expected_status": "FAIL (Defect: Crazing)"
        },
        {
            "filename": "billet_fail_scratches.jpg",
            "side_filename": "billet_side_fail_scratches.jpg",
            "width_mm": 125.2,
            "height_mm": 124.8,
            "heat_number": "H727625R1.0",
            "defect_type": "scratches",
            "expected_status": "FAIL (Defect: Scratches)"
        },
        {
            "filename": "billet_fail_dimension_wide.jpg",
            "side_filename": "billet_side_fail_dimension_wide.jpg",
            "width_mm": 131.8,  # > 128mm tolerance
            "height_mm": 125.2,
            "heat_number": "H727626R1.0",
            "defect_type": None,
            "expected_status": "FAIL (Width 131.8mm out of spec)"
        },
        {
            "filename": "billet_fail_inclusion.jpg",
            "side_filename": "billet_side_fail_inclusion.jpg",
            "width_mm": 125.0,
            "height_mm": 124.5,
            "heat_number": "H727627R3.0",
            "defect_type": "inclusion",
            "expected_status": "FAIL (Defect: Inclusion)"
        }
    ]

    print(f"[Test Generator] Generating Front (60°) and Side (90°) test billets in {TEST_DATA_DIR}...")
    for s in scenarios:
        # 1. Front View (60°)
        img_front = create_realistic_billet_frame(
            width_mm=s["width_mm"],
            height_mm=s["height_mm"],
            heat_number=s["heat_number"],
            defect_type=s["defect_type"],
            angle_deg=60.0
        )
        path_front = os.path.join(TEST_DATA_DIR, s["filename"])
        cv2.imwrite(path_front, img_front)

        # 2. Side View (90°)
        img_side = create_side_view_billet_frame(
            height_mm=s["height_mm"],
            defect_type=s["defect_type"]
        )
        path_side = os.path.join(TEST_DATA_DIR, s["side_filename"])
        cv2.imwrite(path_side, img_side)

        print(f"  + Created {s['filename']} (Front 60°) & {s['side_filename']} (Side 90°) -> {s['expected_status']}")

    print("[Test Generator] Dual-camera test suite generated successfully.")


def generate_test_conveyor_videos(fps=30, duration_sec=4):
    """
    Generates realistic conveyor video streams for testing video upload pipelines:
    - data/test/conveyor_front_60deg.mp4: Front camera (60° tilt) showing passing billets with stamped heat numbers
    - data/test/conveyor_side_90deg.mp4: Side camera (90° orthogonal) showing longitudinal billet profile
    """
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 1. Front Camera Video (60° tilt)
    front_path = os.path.join(TEST_DATA_DIR, "conveyor_front_60deg.mp4")
    out_front = cv2.VideoWriter(front_path, fourcc, fps, (1920, 1080))

    billet_templates = [
        {"w": 125.0, "h": 125.0, "heat": "H727622R1.0", "defect": None},
        {"w": 125.0, "h": 125.0, "heat": "H727624R2.1", "defect": "scratches"},
    ]

    print(f"[Test Generator] Generating Front Conveyor Video (60°): {front_path}...")
    for frame_i in range(total_frames):
        # Base conveyor
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 28
        # Conveyor roller lines
        for y_r in range(0, 1080, 80):
            cv2.line(frame, (0, y_r), (1920, y_r), (42, 45, 52), 2)

        # Decide which billet is currently moving through
        # Billet 1 from frame 0 to 60, Billet 2 from 60 to 120
        b_idx = min(len(billet_templates) - 1, frame_i // (total_frames // len(billet_templates)))
        b_info = billet_templates[b_idx]
        progress_in_slot = (frame_i % (total_frames // len(billet_templates))) / (total_frames // len(billet_templates))

        # Position transitions from left (x=300) to right (x=1600)
        cx = int(300 + progress_in_slot * 1300)
        cy = 540

        # Draw billet at (cx, cy)
        theta = math.radians(60.0)
        w_px = int(round(b_info["w"] * math.cos(theta) * 2.77))
        h_px = int(round(b_info["h"] * math.sin(theta) * 2.77))
        x1, y1 = cx - w_px // 2, cy - h_px // 2
        x2, y2 = x1 + w_px, y1 + h_px

        top_offset_x, top_offset_y = 260, -320
        body_pts = np.array([
            [x1, y1], [x2, y1],
            [x2 + top_offset_x, y1 + top_offset_y],
            [x1 + top_offset_x, y1 + top_offset_y]
        ], np.int32)
        cv2.fillPoly(frame, [body_pts], (80, 85, 95))
        cv2.polylines(frame, [body_pts], True, (55, 60, 68), 2)

        # Draw end face
        end_face = np.ones((h_px, w_px, 3), dtype=np.uint8) * 165
        grain = np.random.normal(0, 6, end_face.shape).astype(np.int16)
        end_face = np.clip(end_face.astype(np.int16) + grain, 0, 255).astype(np.uint8)

        # Add stamped text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = b_info["heat"]
        font_scale = 0.55 * (w_px / 173.0)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
        tx = max(4, (w_px - tw) // 2)
        ty = (h_px + th) // 2
        cv2.putText(end_face, text, (tx + 1, ty + 1), font, font_scale, (200, 200, 210), 2, cv2.LINE_AA)
        cv2.putText(end_face, text, (tx, ty), font, font_scale, (30, 30, 30), 2, cv2.LINE_AA)

        # Bevel line
        cv2.line(end_face, (0, 0), (w_px, 0), (25, 25, 28), 3)

        # Draw defect if present
        if b_info["defect"] == "scratches":
            cv2.line(end_face, (tx, ty - 15), (tx + 60, ty - 15), (20, 20, 20), 2)

        # Paste onto frame
        if 0 <= y1 and y2 <= 1080 and 0 <= x1 and x2 <= 1920:
            frame[y1:y2, x1:x2] = end_face

        out_front.write(frame)

    out_front.release()
    print(f"  + Generated {front_path}")

    # 2. Side Camera Video (90° orthogonal)
    side_path = os.path.join(TEST_DATA_DIR, "conveyor_side_90deg.mp4")
    out_side = cv2.VideoWriter(side_path, fourcc, fps, (1920, 1080))
    print(f"[Test Generator] Generating Side Conveyor Video (90°): {side_path}...")

    for frame_i in range(total_frames):
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 25
        for y_r in range(0, 1080, 80):
            cv2.line(frame, (0, y_r), (1920, y_r), (38, 40, 48), 2)

        b_idx = min(len(billet_templates) - 1, frame_i // (total_frames // len(billet_templates)))
        b_info = billet_templates[b_idx]
        progress_in_slot = (frame_i % (total_frames // len(billet_templates))) / (total_frames // len(billet_templates))

        cx = int(200 + progress_in_slot * 1500)
        cy = 540
        bh_px = int(round(b_info["h"] * 2.77))
        bw_px = 1200  # visible longitudinal section

        bx1 = cx - bw_px // 2
        by1 = cy - bh_px // 2
        bx2 = bx1 + bw_px
        by2 = by1 + bh_px

        # Draw side face
        side_crop_x1 = max(0, bx1)
        side_crop_x2 = min(1920, bx2)
        if side_crop_x2 > side_crop_x1 and 0 <= by1 and by2 <= 1080:
            sw = side_crop_x2 - side_crop_x1
            sh = bh_px
            surf = np.ones((sh, sw, 3), dtype=np.uint8) * 85
            noise = np.random.normal(0, 5, surf.shape).astype(np.int16)
            surf = np.clip(surf.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            if b_info["defect"] == "scratches":
                cv2.line(surf, (sw // 2 - 80, sh // 2), (sw // 2 + 80, sh // 2), (20, 20, 20), 3)

            frame[by1:by2, side_crop_x1:side_crop_x2] = surf
            cv2.rectangle(frame, (side_crop_x1, by1), (side_crop_x2, by2), (40, 45, 52), 2)

        out_side.write(frame)

    out_side.release()
    print(f"  + Generated {side_path}")


if __name__ == "__main__":
    generate_all_test_billets()
    generate_test_conveyor_videos()
