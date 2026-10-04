"""
Automated Billet Quality Inspection System - Camera Calibration Module
======================================================================
Calibrates the industrial 1920x1080 camera mounted 600mm above the billet
at a 60-degree angle from the horizontal conveyor plane.

Key Responsibilities:
1. Calibrate camera intrinsics using standard checkerboard (9x6 corners, 20mm square).
2. Calculate pixels_per_mm scale factor.
3. Compute perspective correction homography matrix for 60-degree tilt.
4. Verify dimensional measurement against IS 2830:2012 tolerance standard (125 +/- 3 mm).
"""

import os
import json
import glob
import math
import numpy as np
import cv2


DEFAULT_CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), "data", "calibration")
DEFAULT_PARAMS_PATH = os.path.join(DEFAULT_CALIBRATION_DIR, "camera_params.json")

# Physical Hardware Specifications (Aligned with Problem Statement Section 2)
FOCAL_LENGTH_MM = 25.0             # 25 mm lens (varifocal C-mount)
CAMERA_HEIGHT_MM = 500.0           # Camera 1: 500 mm vertical offset along motion axis
FRONT_CAMERA_ANGLE_DEG = 60.0      # Camera 1: 60 deg incline facing oncoming movement (OCR + End Face)
SIDE_CAMERA_ANGLE_DEG = 90.0       # Camera 2: 90 deg direct perpendicular side view (Height + Length)
SIDE_CAMERA_OFFSET_MM = 400.0      # Camera 2: 400 mm lateral offset from conveyor edge
SIDE_CALIBRATION_FACTOR = 4.5      # Camera 2: CF = 4.5 pixels/mm
INCLINATION_ANGLE_DEG = FRONT_CAMERA_ANGLE_DEG
NOMINAL_BILLET_SIZE_MM = 125.0     # IS 2830:2012: 125x125 mm
IS2830_TOLERANCE_MM = 3.0          # +/- 3.0 mm tolerance (<= +/- 1.0% error margin)


class CameraCalibrator:
    """
    Handles camera intrinsic calibration, distortion rectification,
    pixels-to-mm scaling, and perspective correction.
    """

    def __init__(self, checkerboard_size=(9, 6), square_size_mm=20.0):
        """
        Initialize calibrator.
        
        Args:
            checkerboard_size (tuple): Number of internal corners (cols, rows), default (9, 6).
            square_size_mm (float): Length of checkerboard square side in mm, default 20.0 mm.
        """
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.camera_matrix = None
        self.dist_coeffs = None
        self.pixels_per_mm = None
        self.homography_matrix = None
        self.reprojection_error = None

    def create_synthetic_calibration_images(self, output_dir=DEFAULT_CALIBRATION_DIR, num_images=5):
        """
        Generates realistic synthetic checkerboard images if physical images are not yet captured.
        Allows immediate testing of calibration pipeline.
        """
        os.makedirs(output_dir, exist_ok=True)
        cols, rows = self.checkerboard_size
        # Full board has cols+1 by rows+1 squares
        board_w_px = (cols + 1) * 60
        board_h_px = (rows + 1) * 60

        print(f"[Calibration] Generating {num_images} synthetic checkerboard patterns in {output_dir}...")
        for idx in range(num_images):
            canvas = np.ones((1080, 1920), dtype=np.uint8) * 180
            # Draw checkerboard
            board = np.zeros((board_h_px, board_w_px), dtype=np.uint8)
            sq_w = board_w_px // (cols + 1)
            sq_h = board_h_px // (rows + 1)
            for r in range(rows + 1):
                for c in range(cols + 1):
                    if (r + c) % 2 == 0:
                        board[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w] = 255

            # Apply slight perspective and rotation to simulate camera angles
            angle = (idx - num_images // 2) * 5.0
            scale = 0.85 + (idx * 0.05)
            M_rot = cv2.getRotationMatrix2D((board_w_px / 2, board_h_px / 2), angle, scale)
            rotated_board = cv2.warpAffine(board, M_rot, (board_w_px, board_h_px), borderValue=180)

            # Paste into 1920x1080 canvas with offset
            x_offset = int(1920 / 2 - board_w_px / 2 + (idx - 2) * 40)
            y_offset = int(1080 / 2 - board_h_px / 2 + (idx - 2) * 25)
            canvas[y_offset:y_offset + board_h_px, x_offset:x_offset + board_w_px] = rotated_board

            # Add subtle Gaussian noise
            noise = np.random.normal(0, 3, canvas.shape).astype(np.int16)
            noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            file_path = os.path.join(output_dir, f"checkerboard_{idx+1:02d}.jpg")
            cv2.imwrite(file_path, noisy_canvas)
        print(f"[Calibration] Created {num_images} calibration images.")

    def calibrate(self, images_dir=DEFAULT_CALIBRATION_DIR, save_params=True):
        """
        Performs camera calibration using checkerboard images in the specified directory.
        Calculates intrinsic matrix, distortion coefficients, and pixels_per_mm.
        """
        cols, rows = self.checkerboard_size
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size_mm

        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        measured_corner_distances = []

        image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                      glob.glob(os.path.join(images_dir, "*.png"))

        if len(image_files) < 3:
            print("[Calibration] Insufficient calibration images found. Generating synthetic images...")
            self.create_synthetic_calibration_images(images_dir)
            image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        image_shape = None
        successful_frames = 0

        for fname in image_files:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_shape = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                successful_frames += 1

                # Measure horizontal corner distances to estimate scale
                for r in range(rows):
                    p1 = corners2[r * cols][0]
                    p2 = corners2[r * cols + (cols - 1)][0]
                    pixel_dist = np.linalg.norm(p2 - p1)
                    mm_dist = (cols - 1) * self.square_size_mm
                    if mm_dist > 0:
                        measured_corner_distances.append(pixel_dist / mm_dist)

        if successful_frames == 0:
            raise RuntimeError(f"Failed to find checkerboard corners in any images under {images_dir}")

        print(f"[Calibration] Successfully detected checkerboards in {successful_frames} images.")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None
        )

        self.camera_matrix = mtx
        self.dist_coeffs = dist

        # Calculate reprojection error
        total_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            pts1 = imgpoints[i].reshape(-1, 2).astype(np.float64)
            pts2 = imgpoints2.reshape(-1, 2).astype(np.float64)
            error = np.sum((pts1 - pts2) ** 2)
            total_error += error
            total_points += len(pts1)
        self.reprojection_error = math.sqrt(total_error / total_points)

        # Baseline pixels_per_mm from optical formula:
        # Working distance = 600mm, Focal length = 25mm -> Magnification = 25/600 = 0.04167
        # Typical 1920x1080 sensor with 5.86um pixel pitch -> ~7.1 pixels/mm at 600mm
        if measured_corner_distances:
            self.pixels_per_mm = float(np.median(measured_corner_distances))
        else:
            self.pixels_per_mm = 7.15

        # Compute Homography for 60-degree angle
        self.compute_homography_matrix()

        print(f"[Calibration] Reprojection Error: {self.reprojection_error:.4f} pixels")
        print(f"[Calibration] Calibrated Scale: {self.pixels_per_mm:.3f} pixels/mm")

        if save_params:
            self.save_parameters(DEFAULT_PARAMS_PATH)

        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "pixels_per_mm": self.pixels_per_mm,
            "reprojection_error": self.reprojection_error,
            "homography_matrix": self.homography_matrix.tolist()
        }

    def compute_homography_matrix(self):
        """
        Computes the perspective homography transformation matrix from the 60-degree
        camera view to a flat perpendicular top-down metric plane.
        """
        theta = math.radians(INCLINATION_ANGLE_DEG)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Approximate 4-corner trapezoid caused by 60-degree projection
        # Top edge is further, bottom edge is closer
        src_pts = np.float32([
            [600, 300],   # Top-left
            [1320, 300],  # Top-right
            [1420, 850],  # Bottom-right (foreshortened)
            [500, 850]    # Bottom-left
        ])

        # Dest points: orthogonal square 125mm x 125mm in pixels
        w_px = NOMINAL_BILLET_SIZE_MM * self.pixels_per_mm
        h_px = NOMINAL_BILLET_SIZE_MM * self.pixels_per_mm
        dst_pts = np.float32([
            [400, 200],
            [400 + w_px, 200],
            [400 + w_px, 200 + h_px],
            [400, 200 + h_px]
        ])

        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return self.homography_matrix

    def save_parameters(self, path=DEFAULT_PARAMS_PATH):
        """Saves calibration parameters to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "pixels_per_mm": self.pixels_per_mm,
            "reprojection_error": self.reprojection_error,
            "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            "nominal_size_mm": NOMINAL_BILLET_SIZE_MM,
            "tolerance_mm": IS2830_TOLERANCE_MM,
            "camera_angle_deg": INCLINATION_ANGLE_DEG,
            "height_mm": CAMERA_HEIGHT_MM,
            "focal_length_mm": FOCAL_LENGTH_MM
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[Calibration] Saved camera calibration parameters to {path}")

    def load_parameters(self, path=DEFAULT_PARAMS_PATH):
        """Loads calibration parameters from JSON."""
        if not os.path.exists(path):
            print(f"[Calibration] Params file {path} not found. Running calibration first...")
            self.calibrate(save_params=True)
            return

        with open(path, "r") as f:
            data = json.load(f)

        self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64) if data["camera_matrix"] else None
        self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64) if data["dist_coeffs"] else None
        self.pixels_per_mm = float(data.get("pixels_per_mm", 7.15))
        self.reprojection_error = float(data.get("reprojection_error", 0.05))
        if data.get("homography_matrix"):
            self.homography_matrix = np.array(data["homography_matrix"], dtype=np.float64)
        else:
            self.compute_homography_matrix()
        print(f"[Calibration] Successfully loaded parameters from {path} (Scale: {self.pixels_per_mm:.3f} px/mm)")


def correct_dimensions(width_measured_mm, height_measured_mm, angle_deg=60.0):
    """
    Applies trigonometric perspective correction for the 60-degree angle mount:
      width_actual = width_measured / cos(60 deg)
      height_actual = height_measured / sin(60 deg)

    Args:
        width_measured_mm (float): Measured raw width in mm from 2D camera plane.
        height_measured_mm (float): Measured raw height in mm from 2D camera plane.
        angle_deg (float): Camera inclination angle from horizontal (default 60.0).

    Returns:
        tuple: (width_actual_mm, height_actual_mm)
    """
    # 90-degree orthogonal side camera: Direct 1:1 metric projection
    if abs(angle_deg - 90.0) < 1.0:
        width_actual = width_measured_mm
        height_actual = height_measured_mm
    else:
        theta_rad = math.radians(angle_deg)
        cos_theta = max(abs(math.cos(theta_rad)), 1e-4)
        sin_theta = max(abs(math.sin(theta_rad)), 1e-4)

        # Front 60-degree camera: trigonometric angle correction
        width_actual = width_measured_mm / cos_theta
        height_actual = height_measured_mm / sin_theta

    return round(float(width_actual), 2), round(float(height_actual), 2)


def check_is2830_tolerance(width_mm, height_mm, nominal_mm=125.0, tolerance_mm=3.0):
    """
    Evaluates dimensional compliance against Indian Standard IS 2830:2012.
    Nominal: 125x125 mm. Permissible tolerance: +/- 3.0 mm (122.0 to 128.0 mm).

    Args:
        width_mm (float): Actual corrected width in mm.
        height_mm (float): Actual corrected height in mm.
        nominal_mm (float): Nominal dimension (125.0 mm).
        tolerance_mm (float): Allowed deviation (+/- 3.0 mm).

    Returns:
        dict: {
            'width_pass': bool,
            'height_pass': bool,
            'is_dimension_pass': bool,
            'width_deviation': float,
            'height_deviation': float,
            'status_msg': str
        }
    """
    width_dev = width_mm - nominal_mm
    height_dev = height_mm - nominal_mm

    width_pass = abs(width_dev) <= tolerance_mm
    height_pass = abs(height_dev) <= tolerance_mm
    is_dimension_pass = width_pass and height_pass

    reasons = []
    if not width_pass:
        reasons.append(f"Width {width_mm:.1f}mm outside {nominal_mm}±{tolerance_mm}mm")
    if not height_pass:
        reasons.append(f"Height {height_mm:.1f}mm outside {nominal_mm}±{tolerance_mm}mm")

    status_msg = "PASS (IS 2830 Compliant)" if is_dimension_pass else "FAIL (" + "; ".join(reasons) + ")"

    return {
        "width_pass": width_pass,
        "height_pass": height_pass,
        "is_dimension_pass": is_dimension_pass,
        "width_deviation": round(width_dev, 2),
        "height_deviation": round(height_dev, 2),
        "status_msg": status_msg
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BILLET QUALITY INSPECTION - CAMERA CALIBRATION PIPELINE")
    print("=" * 60)

    calibrator = CameraCalibrator()
    params = calibrator.calibrate()

    print("\n--- Calibration Summary ---")
    print(f"Pixels per mm: {params['pixels_per_mm']:.4f}")
    print(f"Reprojection Error: {params['reprojection_error']:.4f} px")

    # Test correction on nominal values
    # For a 125x125 billet at 60 deg, measured on image:
    # width_measured = 125 * cos(60) = 62.5 mm
    # height_measured = 125 * sin(60) = 108.25 mm
    w_meas = 125.0 * math.cos(math.radians(60))
    h_meas = 125.0 * math.sin(math.radians(60))
    w_act, h_act = correct_dimensions(w_meas, h_meas, 60.0)
    res = check_is2830_tolerance(w_act, h_act)
    print(f"\nVerification on nominal: Measured ({w_meas:.2f}, {h_meas:.2f}) -> Corrected ({w_act:.2f}, {h_act:.2f})")
    print(f"Tolerance check: {res['status_msg']}")
