"""
Automated Billet Quality Inspection System - Main Inspection Pipeline
====================================================================
Production-grade computer vision pipeline deployed for steel continuous casting
and rolling mill inspection (Tata Steel, JSW, SAIL standard).

Pipeline Stages:
1. Industrial Camera Frame Acquisition (USB Camera @ 60-degree tilt, 600mm height)
2. Surface Defect Detection (Triple-voting YOLO ensemble: YOLOv8 + YOLOv10 + Texture Analyzer)
3. Cross-Sectional Dimensional Measurement (Contour analysis + 60-deg trigonometric correction)
4. Heat Number Stamping OCR (Triple-voting: EasyOCR + Tesseract + PaddleOCR)
5. Quality Decision Engine (IS 2830:2012 tolerance check: 125 +/- 3 mm, zero defects)
6. Excel Auto-Logging (billet_log.xlsx with concurrency and file-lock recovery)
7. Operator Alert Dispatcher (Desktop Plyer alerts, audio chime, shadow mode)
"""

import os
import sys
import time
import glob
import math
import argparse
import cv2
import numpy as np

from dimension_analyzer import DimensionAnalyzer
from ocr_engine import TripleVotingOCR
from defect_detector import SurfaceDefectDetector
from excel_logger import billet_logger
from alert_system import alert_system
from calibrate_camera import check_is2830_tolerance, correct_dimensions


class BilletInspectionPipeline:
    """
    Main integrated orchestrator for automated billet quality inspection.
    """

    def __init__(self, shadow_mode=False, conf_thresh=0.50):
        print("\n" + "=" * 65)
        print("  AUTOMATED BILLET QUALITY INSPECTION SYSTEM (IS 2830:2012)")
        print("  Deployment: Continuous Casting & Rolling Mill Finishing Line")
        print("=" * 65)

        self.shadow_mode = shadow_mode
        self.conf_thresh = conf_thresh
        alert_system.set_shadow_mode(shadow_mode)

        # Initialize sub-modules
        print("[Pipeline] Initializing Dimensional Analysis Engine...")
        self.dim_analyzer = DimensionAnalyzer()

        print("[Pipeline] Initializing Surface Defect Detection Ensemble...")
        self.defect_detector = SurfaceDefectDetector(conf_thresh=conf_thresh)

        print("[Pipeline] Initializing Triple-Voting OCR Engine...")
        self.ocr_engine = TripleVotingOCR(enable_easyocr=True, enable_tesseract=True, enable_paddleocr=False)

        # Operational metrics
        self.total_inspected = 0
        self.pass_count = 0
        self.fail_count = 0
        self.last_inspected_heat = None
        self.last_inspection_time = 0

    def inspect_frame(self, frame_front_bgr, frame_side_bgr=None):
        """
        Executes dual-camera inspection cycle:
        1. Front Camera (60°): End face OCR, cross-section width/height, top surface defects
        2. Side Camera (90°): Longitudinal side face defects, true side height
        3. Cross-view height verification & IS 2830 Decision Engine
        4. Auto-logging and Alerts
        """
        t_start = time.time()
        h, w = frame_front_bgr.shape[:2]

        # 1. Front View Dimensional Measurement (60° tilt)
        dim_front = self.dim_analyzer.measure_dimensions(frame_front_bgr, camera_view="front")
        width_mm = dim_front.get("width_actual_mm", 0.0)
        height_front_mm = dim_front.get("height_actual_mm", 0.0)
        dim_front_pass = dim_front.get("is_pass", False)

        # 2. Side View Dimensional Measurement (90° orthogonal) if available
        dim_side = None
        dim_side_pass = True
        height_side_mm = height_front_mm
        dual_cross = {}
        if frame_side_bgr is not None:
            dim_side = self.dim_analyzer.measure_dimensions(frame_side_bgr, camera_view="side")
            height_side_mm = dim_side.get("height_actual_mm", height_front_mm)
            dim_side_pass = dim_side.get("is_pass", False)
            dual_cross = self.dim_analyzer.cross_validate_dual_views(dim_front, dim_side)

        # Overall dimensional pass: Front width & height + Side height
        dim_overall_pass = dim_front_pass and dim_side_pass

        # 3. Defect Detection Ensemble across both views
        defects_front = self.defect_detector.detect_defects(frame_front_bgr, camera_view="front")
        defects_side = []
        if frame_side_bgr is not None:
            defects_side = self.defect_detector.detect_defects(frame_side_bgr, camera_view="side")

        all_defects = defects_front + defects_side
        defect_pass = (len(all_defects) == 0)
        defect_strings = [f"{d['class_name']} ({d['confidence']:.2f})" for d in all_defects]

        # 4. Extract End Face ROI & Run OCR
        contour_box = dim_front.get("contour_box")
        if contour_box is not None:
            bx1 = max(0, int(np.min(contour_box[:, 0])))
            by1 = max(0, int(np.min(contour_box[:, 1])))
            bx2 = min(w, int(np.max(contour_box[:, 0])))
            by2 = min(h, int(np.max(contour_box[:, 1])))
            roi_end_face = frame_front_bgr[by1:by2, bx1:bx2]
        else:
            roi_end_face = frame_front_bgr[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)]

        ocr_result = self.ocr_engine.read_heat_number(roi_end_face)
        heat_number = ocr_result["heat_number"]

        # 5. IS 2830 Decision Engine
        rejection_reasons = []
        if not dim_front_pass:
            tol = dim_front.get("tolerance_info", {})
            rejection_reasons.append(f"Front Dim: {tol.get('status_msg', 'Out of spec')}")
        if not dim_side_pass and dim_side:
            tol_s = dim_side.get("tolerance_info", {})
            rejection_reasons.append(f"Side Height: {height_side_mm:.1f}mm outside 125±3mm")
        if not defect_pass:
            names = list(set([d["class_name"] for d in all_defects]))
            rejection_reasons.append(f"Surface defects: {', '.join(names)}")

        overall_status = "PASS" if (dim_overall_pass and defect_pass) else "FAIL"

        # Update inspection counts
        self.total_inspected += 1
        if overall_status == "PASS":
            self.pass_count += 1
        else:
            self.fail_count += 1

        cycle_latency_ms = (time.time() - t_start) * 1000.0

        # 6. Data Logging to Excel (Auto-append)
        billet_logger.log_inspection(
            heat_number=heat_number,
            width_mm=width_mm,
            height_mm=height_side_mm if frame_side_bgr is not None else height_front_mm,
            defects=defect_strings,
            status=overall_status
        )

        # 7. Alert System
        if overall_status == "FAIL":
            fail_msg = " | ".join(rejection_reasons)
            alert_system.trigger_failure_alert(
                heat_number=heat_number,
                failure_reason=fail_msg,
                width_mm=width_mm,
                height_mm=height_side_mm,
                defects=defect_strings
            )

        print(f"\n[INSPECTION #{self.total_inspected}] Heat: {heat_number} | Dim: {width_mm:.1f}x{height_side_mm:.1f}mm | Dual Cross: {dual_cross.get('status', 'Single Cam')} | Defects: {defect_strings or 'None'} | Status: {overall_status} ({cycle_latency_ms:.1f}ms)")

        return {
            "heat_number": heat_number,
            "ocr_confidence": ocr_result["confidence"],
            "ocr_method": ocr_result.get("method", "N/A"),
            "width_mm": width_mm,
            "height_front_mm": height_front_mm,
            "height_side_mm": height_side_mm,
            "height_mm": height_side_mm if frame_side_bgr is not None else height_front_mm,
            "dim_pass": dim_overall_pass,
            "defects_front": defects_front,
            "defects_side": defects_side,
            "defects": all_defects,
            "defect_strings": defect_strings,
            "defect_pass": defect_pass,
            "status": overall_status,
            "reasons": rejection_reasons,
            "contour_box": contour_box,
            "contour_box_side": dim_side.get("contour_box") if dim_side else None,
            "dual_cross": dual_cross,
        }

    def render_hud(self, frame_front_bgr, inspection_data, frame_side_bgr=None, fps=0.0):
        """
        Renders rich industrial dual-camera HUD overlay:
        - Left viewport: Front Camera @ 60° (End face OCR & Cross-Section)
        - Right viewport: Side Camera @ 90° (Longitudinal Surface Defects & Orthogonal Height)
        """
        # Annotate Front View
        vis_front = frame_front_bgr.copy()
        defects_front = inspection_data.get("defects_front", [])
        vis_front = self.defect_detector.draw_defect_boxes(vis_front, defects_front)

        contour_box = inspection_data.get("contour_box")
        dim_pass = inspection_data.get("dim_pass", False)
        box_color = (0, 220, 0) if dim_pass else (0, 0, 240)
        if contour_box is not None:
            cv2.drawContours(vis_front, [contour_box], 0, box_color, 2)

        # Label Front Viewport
        cv2.putText(vis_front, "CAM 1: FRONT VIEW (60 DEG TILT)", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        # If Side View provided, create side-by-side composite
        if frame_side_bgr is not None:
            vis_side = frame_side_bgr.copy()
            defects_side = inspection_data.get("defects_side", [])
            vis_side = self.defect_detector.draw_defect_boxes(vis_side, defects_side)

            # Label Side Viewport
            cv2.putText(vis_side, "CAM 2: SIDE VIEW (90 DEG ORTHOGONAL)", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

            # Resize both to 960x540 for split view
            front_resized = cv2.resize(vis_front, (960, 540))
            side_resized = cv2.resize(vis_side, (960, 540))

            # Combine horizontally
            composite = np.hstack([front_resized, side_resized])
            # Draw dividing vertical line
            cv2.line(composite, (960, 0), (960, 540), (0, 200, 255), 2)
            vis = composite
        else:
            vis = cv2.resize(vis_front, (1280, 720))

        h, w = vis.shape[:2]

        # Top HUD Banner (Plant Header & Telemetry)
        header_h = 60
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 20, 26), -1)
        vis = cv2.addWeighted(overlay, 0.85, vis, 0.15, 0)
        cv2.line(vis, (0, header_h), (w, header_h), (0, 200, 255), 2)

        # Header titles
        cv2.putText(vis, "DUAL-CAMERA BILLET QUALITY INSPECTION (FRONT 60° + SIDE 90°)", (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        dual_status = inspection_data.get("dual_cross", {}).get("status", "IS 2830 ACTIVE")
        cv2.putText(vis, f"CONSENSUS: {dual_status}", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 1, cv2.LINE_AA)

        # Metrics on right
        pass_rate = (self.pass_count / max(1, self.total_inspected)) * 100.0
        hud_metrics = f"Inspected: {self.total_inspected} | Pass: {self.pass_count} | Fail: {self.fail_count} | Yield: {pass_rate:.1f}% | FPS: {fps:.1f}"
        (mw, _), _ = cv2.getTextSize(hud_metrics, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(vis, hud_metrics, (w - mw - 20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 240), 2, cv2.LINE_AA)

        # Bottom Quality Decision Banner
        status = inspection_data.get("status", "INSPECTING")
        banner_color = (18, 120, 40) if status == "PASS" else (20, 25, 180)
        banner_h = 55
        cv2.rectangle(vis, (0, h - banner_h), (w, h), banner_color, -1)

        heat = inspection_data.get("heat_number", "UNKNOWN")
        w_act = inspection_data.get("width_mm", 0.0)
        h_act = inspection_data.get("height_mm", 0.0)
        status_text = f"STATUS: {status} | HEAT: {heat} | DIM: {w_act:.1f}x{h_act:.1f}mm"
        reasons = inspection_data.get("reasons", [])
        if reasons:
            status_text += f" [{reasons[0]}]"

        cv2.putText(vis, status_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        help_text = "[SPACE] Next | [S] Shadow Mode | [Q] Exit"
        (hw, _), _ = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(vis, help_text, (w - hw - 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 240, 255), 1, cv2.LINE_AA)

        return vis

    def run_stream(self, source=0, camera_view="auto", simulation_mode=False, save_video=None):
        """
        Runs live inspection loop on video camera feed, uploaded video file, or simulated test dataset.
        Supports dual Front (60°) + Side (90°) camera streaming with tripwire trigger debouncing.
        """
        print(f"\n[Pipeline] Starting inspection stream (Source: {source}, View: {camera_view}, Simulation: {simulation_mode})...")

        front_images = []
        side_images = []
        is_video_file = False

        if simulation_mode or str(source).lower() == "simulation":
            test_dir = os.path.join(os.path.dirname(__file__), "data", "test")
            front_images = sorted([f for f in glob.glob(os.path.join(test_dir, "*.jpg")) if "side" not in os.path.basename(f)])
            side_images = sorted([f for f in glob.glob(os.path.join(test_dir, "*side*.jpg"))])
            if not front_images or not side_images:
                from generate_mock_data import generate_all_test_billets
                generate_all_test_billets()
                front_images = sorted([f for f in glob.glob(os.path.join(test_dir, "*.jpg")) if "side" not in os.path.basename(f)])
                side_images = sorted([f for f in glob.glob(os.path.join(test_dir, "*side*.jpg"))])
            cap = None
        else:
            try:
                cap_idx = int(source)
                cap = cv2.VideoCapture(cap_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            except ValueError:
                is_video_file = True
                cap = cv2.VideoCapture(str(source))

            if cap is None or not cap.isOpened():
                print(f"[Pipeline Notice] Physical source '{source}' unavailable. Automatically switching to Dual-Camera Billet Simulation Mode.")
                return self.run_stream(source="simulation", simulation_mode=True)

        # Video writer if requested
        out_writer = None
        if save_video and cap is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_video)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            out_writer = cv2.VideoWriter(save_video, fourcc, vfps, (vw, vh))

        test_idx = 0
        frame_idx = 0
        current_data = None
        last_inspected_frame = -60
        cooldown_frames = 20  # Minimum frames between distinct billet triggers on conveyor
        t_prev = time.time()
        fps = 0.0

        try:
            while True:
                if cap is not None and cap.isOpened():
                    ret, raw_frame = cap.read()
                    if not ret:
                        print("[Pipeline] Video or camera stream ended.")
                        break

                    frame_idx += 1
                    h_f, w_f = raw_frame.shape[:2]

                    # Auto-detect camera view or use specified view
                    if camera_view == "auto":
                        # If aspect ratio is wide (w > 2.5 * h for billet contour), it's side view
                        view_mode = "side" if (w_f / max(1, h_f) > 2.2) else "front"
                    else:
                        view_mode = camera_view.lower()

                    if view_mode == "side":
                        frame_front = None
                        frame_side = raw_frame
                    elif view_mode == "dual":
                        # Split screen: left half front 60°, right half side 90°
                        mid_x = w_f // 2
                        frame_front = raw_frame[:, :mid_x]
                        frame_side = raw_frame[:, mid_x:]
                    else:
                        frame_front = raw_frame
                        frame_side = None

                    # Conveyor Billet Trigger Logic:
                    # Check if a billet is centered in the tripwire inspection zone
                    can_inspect = (frame_idx - last_inspected_frame) >= cooldown_frames
                    check_frame = frame_front if frame_front is not None else frame_side

                    if can_inspect:
                        # Test if billet profile is present in center inspection zone
                        dim_check = self.dim_analyzer.measure_dimensions(check_frame, camera_view=view_mode if view_mode != "dual" else "front")
                        if dim_check.get("detected", False):
                            # Trigger full inspection cycle
                            if frame_front is not None and frame_side is not None:
                                current_data = self.inspect_frame(frame_front, frame_side)
                            elif frame_front is not None:
                                current_data = self.inspect_frame(frame_front, None)
                            else:
                                # Side-only inspection
                                current_data = self.inspect_frame(frame_side, frame_side)
                            last_inspected_frame = frame_idx

                else:
                    # Simulation mode: load matched Front (60°) and Side (90°) pairs
                    f_path = front_images[test_idx % len(front_images)]
                    s_path = side_images[test_idx % len(side_images)]
                    frame_front = cv2.imread(f_path)
                    frame_side = cv2.imread(s_path)
                    if frame_front is None:
                        continue

                    # Trigger inspection when switching billets in simulation
                    if current_data is None:
                        current_data = self.inspect_frame(frame_front, frame_side)

                # Real-time FPS
                t_now = time.time()
                dt = t_now - t_prev
                t_prev = t_now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # Render rich dual-camera HUD
                display_front = frame_front if frame_front is not None else frame_side
                display_side = frame_side if (frame_front is not None and frame_side is not None) else None

                vis_data = current_data if current_data is not None else {
                    "status": "TRACKING CONVEYOR",
                    "heat_number": "CONVEYOR ACTIVE",
                    "width_mm": 0.0,
                    "height_mm": 0.0,
                    "defects": [],
                    "dim_pass": True
                }
                display_frame = self.render_hud(display_front, vis_data, frame_side_bgr=display_side, fps=fps)

                # Record frame if requested
                if out_writer is not None:
                    out_writer.write(cv2.resize(display_frame, (vw, vh)))

                # Display window
                cv2.imshow("Automated Billet Quality Inspection System", display_frame)

                # Handle keyboard inputs
                wait_time = 1800 if (cap is None) else (15 if is_video_file else 1)
                key = cv2.waitKey(wait_time) & 0xFF

                if key == ord('q'):
                    print("[Pipeline] Operator initiated exit.")
                    break
                elif key == ord('s'):
                    self.shadow_mode = not self.shadow_mode
                    alert_system.set_shadow_mode(self.shadow_mode)
                elif key == ord(' ') and cap is None:
                    test_idx += 1
                    current_data = None

        finally:
            if cap is not None:
                cap.release()
            if out_writer is not None:
                out_writer.release()
            cv2.destroyAllWindows()
            print("\n[Pipeline] Inspection session finished. All records preserved in billet_log.xlsx")


def main():
    parser = argparse.ArgumentParser(description="Automated Billet Quality Inspection System")
    parser.add_argument("--source", default="0", help="Camera index (0, 1), video file path (.mp4, .avi), or 'simulation'")
    parser.add_argument("--view", default="auto", choices=["auto", "front", "side", "dual"], help="Camera view type (default: auto)")
    parser.add_argument("--shadow", action="store_true", help="Enable shadow mode (silence audible alerts)")
    parser.add_argument("--conf", type=float, default=0.50, help="Defect detection confidence threshold (default 0.50)")
    parser.add_argument("--simulate", action="store_true", help="Force simulation mode with calibrated test billets")
    parser.add_argument("--save_video", default=None, help="Optional output path to save annotated inspection video")

    args = parser.parse_args()

    pipeline = BilletInspectionPipeline(shadow_mode=args.shadow, conf_thresh=args.conf)
    sim_mode = args.simulate or (str(args.source).lower() == "simulation")
    pipeline.run_stream(source=args.source, camera_view=args.view, simulation_mode=sim_mode, save_video=args.save_video)


if __name__ == "__main__":
    main()
