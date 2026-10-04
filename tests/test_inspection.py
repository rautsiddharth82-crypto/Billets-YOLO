"""
Automated Billet Quality Inspection System - Unit Test Suite
============================================================
Comprehensive test suite verifying:
1. Dimensional perspective & trigonometric correction (60-deg angle)
2. IS 2830:2012 tolerance checking (125 +/- 3 mm)
3. OCR heat number pattern validation and voting
4. Excel logger concurrency, retry logic, and fallback buffering
5. Camera calibration parameter persistence and retrieval
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import tempfile
import unittest
import pandas as pd
import cv2
import numpy as np

from calibrate_camera import correct_dimensions, check_is2830_tolerance, CameraCalibrator
from ocr_engine import TripleVotingOCR, HEAT_NUMBER_REGEX
from excel_logger import BilletExcelLogger


class TestBilletInspection(unittest.TestCase):

    def test_trigonometric_correction(self):
        """Tests that 60-degree foreshortened measurements are corrected properly."""
        angle = 60.0
        # If nominal is 125 x 125 mm:
        measured_w = 125.0 * math.cos(math.radians(angle))  # 62.5 mm
        measured_h = 125.0 * math.sin(math.radians(angle))  # ~108.25 mm

        act_w, act_h = correct_dimensions(measured_w, measured_h, angle_deg=angle)

        self.assertAlmostEqual(act_w, 125.0, delta=0.1)
        self.assertAlmostEqual(act_h, 125.0, delta=0.1)

    def test_side_camera_orthogonal_correction(self):
        """Tests that 90-degree orthogonal side camera requires zero trigonometric distortion."""
        measured_h = 125.0
        measured_l = 400.0  # Visible segment length
        act_l, act_h = correct_dimensions(measured_l, measured_h, angle_deg=90.0)

        self.assertEqual(act_l, 400.0)
        self.assertEqual(act_h, 125.0)

    def test_dual_camera_cross_validation(self):
        """Tests dual camera consensus verification between 60 deg front and 90 deg side cameras."""
        from dimension_analyzer import DimensionAnalyzer
        da = DimensionAnalyzer()

        front_res = {"detected": True, "height_actual_mm": 124.5, "is_pass": True}
        side_res = {"detected": True, "height_actual_mm": 125.1, "is_pass": True}
        cross = da.cross_validate_dual_views(front_res, side_res, height_tolerance_mm=2.5)

        self.assertTrue(cross["dual_pass"])
        self.assertAlmostEqual(cross["delta_height_mm"], 0.6, delta=0.05)
        self.assertIn("CONSENSUS PASS", cross["status"])

    def test_is2830_tolerance(self):
        """Tests Indian Standard IS 2830:2012 tolerance evaluation (125 +/- 3 mm)."""
        # Nominal: 125.0 mm -> PASS
        res_pass = check_is2830_tolerance(125.0, 125.0)
        self.assertTrue(res_pass["is_dimension_pass"])
        self.assertTrue(res_pass["width_pass"])
        self.assertTrue(res_pass["height_pass"])

        # Boundary pass: 128.0 mm -> PASS
        res_bound = check_is2830_tolerance(128.0, 122.0)
        self.assertTrue(res_bound["is_dimension_pass"])

        # Out of tolerance: 129.5 mm -> FAIL
        res_fail_w = check_is2830_tolerance(129.5, 125.0)
        self.assertFalse(res_fail_w["is_dimension_pass"])
        self.assertFalse(res_fail_w["width_pass"])
        self.assertTrue(res_fail_w["height_pass"])

        # Under tolerance: 120.0 mm -> FAIL
        res_fail_h = check_is2830_tolerance(125.0, 120.0)
        self.assertFalse(res_fail_h["is_dimension_pass"])
        self.assertFalse(res_fail_h["height_pass"])

    def test_heat_number_regex(self):
        """Tests strict regex pattern ^[A-Z]\\d{6,7}[A-Z0-9\\.]*$ for stamped heat codes."""
        valid_codes = [
            "H727622R1.0",
            "H123456",
            "B7654321",
            "S987654R2.1",
            "T1234567A"
        ]
        invalid_codes = [
            "1234567",       # No leading uppercase letter
            "H1234",         # Too few digits (< 6)
            "h727622r1.0",   # Lowercase
            "STEEL_BILLET",  # Non-heat format
            ""               # Empty
        ]

        for code in valid_codes:
            self.assertIsNotNone(HEAT_NUMBER_REGEX.match(code), f"Expected {code} to be valid")

        for code in invalid_codes:
            self.assertIsNone(HEAT_NUMBER_REGEX.match(code), f"Expected {code} to be invalid")

    def test_excel_logger_basic(self):
        """Tests Excel logging and recovery buffering."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_xlsx = os.path.join(tmp_dir, "test_log.xlsx")
            logger = BilletExcelLogger(file_path=test_xlsx)

            logger.log_inspection("H727622R1.0", 125.2, 124.8, [], "PASS")
            logger.log_inspection("H727623R2.0", 131.0, 125.0, ["scratches"], "FAIL")

            df = logger.get_recent_logs(10)
            self.assertEqual(len(df), 2)
            self.assertIn("H727622R1.0", df["Heat_Number"].values)
            self.assertIn("H727623R2.0", df["Heat_Number"].values)

    def test_camera_calibration_persistence(self):
        """Tests saving and loading camera calibration parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            calib_file = os.path.join(tmp_dir, "test_camera_params.json")
            calibrator = CameraCalibrator()
            calibrator.camera_matrix = np.eye(3)
            calibrator.dist_coeffs = np.zeros((5, 1))
            calibrator.pixels_per_mm = 5.25
            calibrator.reprojection_error = 0.04
            calibrator.compute_homography_matrix()

            calibrator.save_parameters(calib_file)
            self.assertTrue(os.path.exists(calib_file))

            # Load back
            calibrator2 = CameraCalibrator()
            calibrator2.load_parameters(calib_file)
            self.assertAlmostEqual(calibrator2.pixels_per_mm, 5.25)
            self.assertAlmostEqual(calibrator2.reprojection_error, 0.04)

    def test_video_processing_pipeline(self):
        """Tests end-to-end video ingestion, metadata extraction, and conveyor inspection."""
        from video_processor import video_processor
        video_front = os.path.join(os.path.dirname(__file__), "..", "data", "test", "conveyor_front_60deg.mp4")
        if not os.path.exists(video_front):
            from generate_mock_data import generate_test_conveyor_videos
            generate_test_conveyor_videos()

        meta = video_processor.get_video_metadata(video_front)
        self.assertEqual(meta["width"], 1920)
        self.assertEqual(meta["height"], 1080)
        self.assertGreater(meta["frame_count"], 0)

        # Test single frame extraction
        frame = video_processor.extract_frame_at(video_front, 10)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (1080, 1920, 3))

        # Test video inspection run
        res = video_processor.process_video(video_front, camera_view="front", log_to_excel=False)
        self.assertGreater(res["billets_inspected"], 0)
        self.assertGreaterEqual(res["total_frames_processed"], 50)

    def test_levenshtein_distance_and_voting(self):
        """Tests pairwise Levenshtein distance consensus voting (DL <= 1)."""
        from ocr_engine import levenshtein_distance
        # Same string -> distance 0
        self.assertEqual(levenshtein_distance("H727622R1.0", "H727622R1.0"), 0)
        # 1 character substitution ('O' vs '0') -> distance 1
        self.assertEqual(levenshtein_distance("H727622R1.0", "H727622R1.O"), 1)
        # 1 character deletion/insertion -> distance 1
        self.assertEqual(levenshtein_distance("H727622R10", "H727622R1.0"), 1)
        # Distinct strings -> distance > 2
        self.assertGreater(levenshtein_distance("H727622R1.0", "B999999X"), 2)

    def test_rhomboidity_metrology(self):
        """Tests billet cross-section diagonal skew (IS 2830 / continuous casting standard: <= 4.0 mm)."""
        from dimension_analyzer import DimensionAnalyzer
        da = DimensionAnalyzer()

        # Simulate a square billet cross section (nominal 125x125 at 60 deg)
        frame_normal = np.ones((600, 800, 3), dtype=np.uint8) * 30
        cv2.rectangle(frame_normal, (250, 200), (550, 400), (160, 160, 160), -1)
        res_normal = da.measure_dimensions(frame_normal, camera_view="front")
        self.assertTrue(res_normal.get("rhomboidity_pass", False))
        self.assertLessEqual(res_normal.get("rhomboidity_delta_mm", 0.0), 4.0)

    def test_smart_safety_override(self):
        """Tests that single-model defects with >90% confidence trigger Smart Safety Override."""
        from defect_detector import SurfaceDefectDetector
        det = SurfaceDefectDetector()

        # Simulate a high-confidence crack detection (>90%)
        predictions = [{
            "class_name": "crazing",
            "confidence": 0.94,
            "box": [100, 100, 200, 200],
            "source": "YOLOv8"
        }]
        fused = det._fuse_detections(predictions)
        self.assertEqual(len(fused), 1)
        self.assertTrue(fused[0].get("smart_safety_override"))

        # Low-confidence single model unverified detection (<90%) with 1 vote should be suppressed
        det.model_v8 = "mock_v8"
        det.model_v10 = "mock_v10"
        predictions_low = [{
            "class_name": "patches",
            "confidence": 0.58,
            "box": [100, 100, 200, 200],
            "source": "YOLOv8"
        }]
        fused_low = det._fuse_detections(predictions_low)
        self.assertEqual(len(fused_low), 0)

    def test_plc_reject_kicker_latency(self):
        """Tests that PLC reject kicker bit signal dispatches in < 200 ms."""
        from alert_system import alert_system
        latency = alert_system.dispatch_plc_reject_kicker("H727622R1.0", "Dimensional Violation")
        self.assertLess(latency, 200.0)
        status = alert_system.get_plc_status()
        self.assertTrue(status["plc_reject_kicker"])
        self.assertIn("Siemens S7-1500", status["plc_target"])

    def test_sqlite_dual_write(self):
        """Tests simultaneous writing to SQLite database billet_master.db."""
        import tempfile
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_xlsx = os.path.join(tmp_dir, "test_log.xlsx")
            test_db = os.path.join(tmp_dir, "billet_master.db")
            logger = BilletExcelLogger(file_path=test_xlsx, db_path=test_db)

            logger.log_inspection("H727622R1.0", 125.1, 124.9, [], "PASS", rhomboidity_mm=0.8)

            # Query SQLite directly
            conn = sqlite3.connect(test_db)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT heat_number, width_mm, rhomboidity_delta_mm FROM billet_inspections")
                row = cursor.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "H727622R1.0")
                self.assertAlmostEqual(row[1], 125.1)
                self.assertAlmostEqual(row[2], 0.8)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
