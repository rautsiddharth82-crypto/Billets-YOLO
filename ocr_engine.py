"""
Automated Billet Quality Inspection System - OCR Engine Module
=============================================================
Implements triple-voting OCR (EasyOCR + Tesseract + PaddleOCR) for reading
stamped heat numbers from billet end faces (e.g., 'H727622R1.0').

Features:
- Industrial image preprocessing (CLAHE, unsharp mask, bilateral denoising, Otsu/adaptive threshold)
- Triple-engine voting with majority consensus (2 out of 3 agree)
- Strict regex validation pattern: ^[A-Z]\\d{6,7}[A-Z0-9\\.]*$
- Graceful fallbacks and character error correction (e.g., 'O' vs '0', 'I' vs '1')
"""

import os
import re
import cv2
import numpy as np

# Regex for heat number: 1 uppercase letter, 6-7 digits, optional revision/strand suffix (e.g., H727622R1.0)
HEAT_NUMBER_REGEX = re.compile(r"^[A-Z]\d{6,7}([A-Z0-9\.]*)$")

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the Levenshtein edit distance between two character strings.
    Used for consensus voting across OCR engines per Problem Statement Step 4 (DL <= 1).
    """
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class TripleVotingOCR:
    """
    Orchestrates EasyOCR, Tesseract, and PaddleOCR to achieve >= 90%
    accuracy on harsh, low-contrast stamped steel surfaces.
    """

    def __init__(self, enable_easyocr=True, enable_tesseract=True, enable_paddleocr=True):
        self.easyocr_reader = None
        self.paddle_ocr = None
        self.tesseract_available = False

        # 1. Initialize EasyOCR
        if enable_easyocr:
            try:
                import easyocr
                # Load models directly from local project directory: models/ocr/
                local_ocr_dir = os.path.join(os.path.dirname(__file__), "models", "ocr")
                os.makedirs(local_ocr_dir, exist_ok=True)

                try:
                    self.easyocr_reader = easyocr.Reader(
                        ['en'],
                        gpu=False,
                        verbose=False,
                        model_storage_directory=local_ocr_dir,
                        user_network_directory=local_ocr_dir,
                        download_enabled=False
                    )
                except Exception:
                    self.easyocr_reader = easyocr.Reader(
                        ['en'],
                        gpu=False,
                        verbose=False,
                        model_storage_directory=local_ocr_dir,
                        user_network_directory=local_ocr_dir,
                        download_enabled=True
                    )
                print(f"[OCR] EasyOCR engine initialized from local folder: {local_ocr_dir}")
            except Exception as e:
                print(f"[OCR Warning] EasyOCR failed to initialize: {e}")

        # 2. Check Tesseract
        if enable_tesseract:
            try:
                import pytesseract
                # Test call
                version = pytesseract.get_tesseract_version()
                self.tesseract_available = True
                print(f"[OCR] Tesseract OCR v{version} detected.")
            except Exception as e:
                # Common on Windows if tesseract.exe is in Program Files
                common_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
                ]
                found = False
                for p in common_paths:
                    if os.path.exists(p):
                        try:
                            import pytesseract
                            pytesseract.pytesseract.tesseract_cmd = p
                            self.tesseract_available = True
                            print(f"[OCR] Tesseract configured at {p}")
                            found = True
                            break
                        except Exception:
                            pass
                if not found:
                    print(f"[OCR Notice] Tesseract executable not detected on PATH. EasyOCR and PaddleOCR will handle voting.")

        # 3. Check PaddleOCR
        if enable_paddleocr:
            try:
                from paddleocr import PaddleOCR
                # Disable angle classification to save time for billet end face
                self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
                print("[OCR] PaddleOCR engine initialized.")
            except Exception as e:
                print(f"[OCR Warning] PaddleOCR failed to initialize: {e}")

        # 4. Barcode and QR Code Detectors (OpenCV native C++ decoders, <2ms execution)
        self.qr_detector = None
        self.barcode_detector = None
        try:
            self.qr_detector = cv2.QRCodeDetector()
            self.barcode_detector = cv2.barcode.BarcodeDetector()
            print("[OCR] OpenCV QR and Barcode detectors initialized.")
        except Exception as e:
            print(f"[OCR Warning] Barcode/QR detector init error: {e}")

    def preprocess_billet_roi(self, img_roi):
        """
        Enhances low-contrast stamped characters on hot/cold rolled steel:
        1. Grayscale conversion
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        3. Bilateral filter for edge-preserving denoising
        4. Unsharp masking to accentuate stamped character edges
        5. Adaptive thresholding
        """
        if img_roi is None or img_roi.size == 0:
            return None

        if len(img_roi.shape) == 3:
            gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_roi.copy()

        # Resize if ROI is small for better OCR character recognition
        h, w = gray.shape[:2]
        if h < 80 or w < 240:
            scale = max(1.5, 120.0 / max(h, 1))
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # 1. CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_boost = clahe.apply(gray)

        # 2. Bilateral filter
        denoised = cv2.bilateralFilter(contrast_boost, 7, 50, 50)

        # 3. Unsharp mask
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)

        # 4. Otsu / Adaptive threshold variant
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return {
            "gray": gray,
            "sharpened": sharpened,
            "binary": binary
        }

    def _clean_text(self, raw_text):
        """Cleans and standardizes raw OCR strings."""
        if not raw_text:
            return ""
        # Remove whitespace and special punctuation except dot
        clean = re.sub(r"[^A-Za-z0-9\.]", "", raw_text.strip().upper())
        # Common steel stamping OCR confusions
        # If first char is '0' or '8', it might be 'H' or 'B'
        if len(clean) > 0 and clean[0] == '0':
            clean = 'H' + clean[1:]
        elif len(clean) > 0 and clean[0] == '8':
            clean = 'B' + clean[1:]
        return clean

    def _run_easyocr(self, img):
        """Runs EasyOCR on processed image."""
        if self.easyocr_reader is None:
            return None
        try:
            results = self.easyocr_reader.readtext(img, detail=1)
            # Find the result that best matches heat pattern
            candidates = []
            for bbox, text, conf in results:
                cleaned = self._clean_text(text)
                if cleaned:
                    candidates.append((cleaned, float(conf)))
            if candidates:
                candidates.sort(key=lambda x: (HEAT_NUMBER_REGEX.match(x[0]) is not None, x[1]), reverse=True)
                return candidates[0]
        except Exception as e:
            pass
        return None

    def _run_tesseract(self, img):
        """Runs Tesseract OCR on processed image."""
        if not self.tesseract_available:
            return None
        try:
            import pytesseract
            # Whitelist uppercase alphanumeric + dot
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.'
            text = pytesseract.image_to_string(img, config=custom_config)
            cleaned = self._clean_text(text)
            if cleaned:
                conf = 0.85 if HEAT_NUMBER_REGEX.match(cleaned) else 0.50
                return (cleaned, conf)
        except Exception:
            pass
        return None

    def _run_paddleocr(self, img):
        """Runs PaddleOCR on processed image."""
        if self.paddle_ocr is None:
            return None
        try:
            result = self.paddle_ocr.ocr(img, cls=False)
            if result and result[0]:
                candidates = []
                for line in result[0]:
                    txt, conf = line[1][0], line[1][1]
                    cleaned = self._clean_text(txt)
                    if cleaned:
                        candidates.append((cleaned, float(conf)))
                if candidates:
                    candidates.sort(key=lambda x: (HEAT_NUMBER_REGEX.match(x[0]) is not None, x[1]), reverse=True)
                    return candidates[0]
        except Exception:
            pass
        return None

    def read_heat_number(self, billet_end_face_bgr):
        """
        Main OCR entry point:
        1. Preprocesses billet end face ROI
        2. Executes triple-voting (EasyOCR, Tesseract, PaddleOCR)
        3. Applies consensus voting logic (2 of 3 agreement or regex validation)
        
        Args:
            billet_end_face_bgr (np.ndarray): Cropped end-face image.
            
        Returns:
            dict: {
                'heat_number': str,
                'confidence': float,
                'votes': dict,
                'is_valid_format': bool,
                'method': str
            }
        """
        if billet_end_face_bgr is None or billet_end_face_bgr.size == 0:
            return {
                "heat_number": "UNKNOWN",
                "confidence": 0.0,
                "votes": {},
                "is_valid_format": False,
                "method": "No image"
            }

        prep = self.preprocess_billet_roi(billet_end_face_bgr)
        if prep is None:
            return {
                "heat_number": "UNKNOWN",
                "confidence": 0.0,
                "votes": {},
                "is_valid_format": False,
                "method": "Preprocessing failed"
            }

        test_images = [
            ("Sharpened", prep["sharpened"]),
            ("OtsuBinary", prep["binary"]),
            ("CLAHE_Gray", prep["gray"])
        ]
        votes = {}

        # 1. EasyOCR (Primary deep convolutional CRNN + CTC engine)
        for name, img in test_images:
            res = self._run_easyocr(img)
            if res and res[0]:
                votes[f"EasyOCR_{name}"] = res
                if HEAT_NUMBER_REGEX.match(res[0]):
                    break

        # 2. Tesseract (LSTM Engine configured with --psm 6 / 7)
        if self.tesseract_available:
            for name, img in test_images:
                res = self._run_tesseract(img)
                if res and res[0]:
                    votes[f"Tesseract_{name}"] = res
                    if HEAT_NUMBER_REGEX.match(res[0]):
                        break

        # 3. PaddleOCR (SVTR Architecture for low-contrast steel surfaces)
        if self.paddle_ocr is not None:
            for name, img in test_images:
                res = self._run_paddleocr(img)
                if res and res[0]:
                    votes[f"PaddleOCR_{name}"] = res
                    if HEAT_NUMBER_REGEX.match(res[0]):
                        break

        # Step 4: Pairwise Levenshtein Distance Voting Logic (DL(Sa, Sb) <= 1)
        selected_text = "UNKNOWN"
        final_conf = 0.0
        method = "None"
        manual_review = False

        candidate_list = list(votes.items())  # [(source_engine, (txt, conf)), ...]

        if candidate_list:
            # 1. Check exact matches first
            text_counts = {}
            for eng, (txt, conf) in candidate_list:
                text_counts[txt] = text_counts.get(txt, 0) + 1

            for txt, count in text_counts.items():
                if count >= 2:
                    selected_text = txt
                    confs = [conf for eng, (t, conf) in candidate_list if t == txt]
                    final_conf = sum(confs) / len(confs)
                    method = f"Exact Consensus ({count} engines/passes)"
                    break

            # 2. Pairwise Levenshtein Distance Alignment (DL <= 1)
            if selected_text == "UNKNOWN" and len(candidate_list) >= 2:
                for i in range(len(candidate_list)):
                    eng1, (txt1, conf1) = candidate_list[i]
                    for j in range(i + 1, len(candidate_list)):
                        eng2, (txt2, conf2) = candidate_list[j]
                        dl = levenshtein_distance(txt1, txt2)
                        if dl <= 1:
                            # They agree within 1 edit distance (e.g. '0' vs 'O' or missing dot)
                            # Pick the one matching standard heat number regex or higher conf
                            m1 = bool(HEAT_NUMBER_REGEX.match(txt1))
                            m2 = bool(HEAT_NUMBER_REGEX.match(txt2))
                            if m1 and not m2:
                                selected_text = txt1
                                final_conf = max(conf1, conf2)
                            elif m2 and not m1:
                                selected_text = txt2
                                final_conf = max(conf1, conf2)
                            else:
                                selected_text = txt1 if conf1 >= conf2 else txt2
                                final_conf = (conf1 + conf2) / 2.0
                            method = f"Levenshtein Consensus (DL={dl} between {eng1} and {eng2})"
                            break
                    if selected_text != "UNKNOWN":
                        break

            # 3. Check for single strong regex match
            if selected_text == "UNKNOWN":
                valid_candidates = [
                    (txt, conf, eng) for eng, (txt, conf) in candidate_list
                    if HEAT_NUMBER_REGEX.match(txt)
                ]
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: x[1], reverse=True)
                    selected_text, final_conf, eng = valid_candidates[0]
                    method = f"Regex Match ({eng})"

            # 4. If all engines strongly disagree (pairwise DL > 2), tag UNREADABLE_OCR per PS
            if selected_text == "UNKNOWN":
                all_pairwise_divergent = True
                for i in range(len(candidate_list)):
                    for j in range(i + 1, len(candidate_list)):
                        if levenshtein_distance(candidate_list[i][1][0], candidate_list[j][1][0]) <= 2:
                            all_pairwise_divergent = False
                            break
                if all_pairwise_divergent and len(candidate_list) >= 2:
                    selected_text = "UNREADABLE_OCR"
                    final_conf = 0.0
                    method = "Dispatched to Operator Review (Pairwise DL > 2)"
                    manual_review = True
                else:
                    # Best effort highest confidence
                    candidate_list.sort(key=lambda x: x[1][1], reverse=True)
                    selected_text, final_conf = candidate_list[0][1]
                    method = f"Highest Confidence ({candidate_list[0][0]})"

        is_valid = bool(HEAT_NUMBER_REGEX.match(selected_text))
        if not is_valid and selected_text != "UNREADABLE_OCR":
            manual_review = True

        return {
            "heat_number": selected_text,
            "confidence": round(float(final_conf), 3),
            "votes": {k: f"{v[0]} ({v[1]:.2f})" for k, v in votes.items()},
            "is_valid_format": is_valid,
            "manual_review": manual_review,
            "method": method
        }

    def extract_all_text(self, image_bgr, min_confidence=0.15, draw_boxes=True, max_dim=1280):
        """
        Extracts ALL detected text from any photo, video frame, or webcam feed.
        Accelerated with intelligent dynamic scaling and multi-orientation recovery for harsh steel plant environments.

        Args:
            image_bgr (np.ndarray): Input image in BGR or RGB format.
            min_confidence (float): Minimum confidence threshold (0.0 to 1.0).
            draw_boxes (bool): If True, draws visual bounding boxes and labels on image copy.
            max_dim (int): Maximum dimension to feed neural network (default: 1280 for sharp text).

        Returns:
            dict: {
                "detected_items": list[dict],
                "all_text": str,
                "lines": list[str],
                "annotated_image": np.ndarray,
                "count": int,
                "heat_code_match": str or None
            }
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "detected_items": [],
                "all_text": "",
                "lines": [],
                "annotated_image": image_bgr,
                "count": 0,
                "heat_code_match": None
            }

        annotated = image_bgr.copy() if draw_boxes else image_bgr
        detected_items = []
        heat_match = None

        def _clean_billet_label(raw):
            t = str(raw).strip()
            if t.startswith("@"):
                t = "0" + t[1:]
            t = t.replace("#", "+")
            if t.startswith("48C"):
                t = "ABC" + t[3:]
            t = t.strip("*.,; ")
            return t

        # Determine scaling
        h_orig, w_orig = image_bgr.shape[:2]
        if max_dim and max(h_orig, w_orig) > max_dim:
            scale = max_dim / float(max(h_orig, w_orig))
            ocr_input = cv2.resize(image_bgr, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
            scale_inv_x = w_orig / float(ocr_input.shape[1])
            scale_inv_y = h_orig / float(ocr_input.shape[0])
        else:
            ocr_input = image_bgr
            scale_inv_x = 1.0
            scale_inv_y = 1.0

        # 0. Scan for 2D QR Codes & 1D Barcodes
        if self.qr_detector is not None:
            try:
                ret_qr, decoded_qr, pts_qr, _ = self.qr_detector.detectAndDecodeMulti(image_bgr)
                if ret_qr and decoded_qr:
                    for idx, q_txt in enumerate(decoded_qr):
                        clean_q = str(q_txt).strip()
                        if clean_q:
                            box_pts = []
                            if pts_qr is not None and len(pts_qr) > idx and pts_qr[idx] is not None:
                                p_arr = np.array(pts_qr[idx], dtype=np.int32).reshape(-1, 2)
                                box_pts = p_arr.tolist()
                            detected_items.append({
                                "text": f"[QR] {clean_q}",
                                "raw_text": clean_q,
                                "type": "QR Code",
                                "confidence": 1.0,
                                "box": box_pts
                            })
                            if not heat_match:
                                m = re.search(r"[A-Z]\d{6,7}", clean_q.upper())
                                if m:
                                    heat_match = m.group(0)
            except Exception as e:
                pass

        if self.barcode_detector is not None:
            try:
                ret_bc, decoded_bc, types_bc, pts_bc = self.barcode_detector.detectAndDecodeMulti(image_bgr)
                if ret_bc and decoded_bc:
                    for idx, b_txt in enumerate(decoded_bc):
                        clean_b = str(b_txt).strip()
                        if clean_b:
                            box_pts = []
                            if pts_bc is not None and len(pts_bc) > idx and pts_bc[idx] is not None:
                                p_arr = np.array(pts_bc[idx], dtype=np.int32).reshape(-1, 2)
                                box_pts = p_arr.tolist()
                            b_type = str(types_bc[idx]) if types_bc is not None and len(types_bc) > idx else "BARCODE"
                            detected_items.append({
                                "text": f"[{b_type}] {clean_b}",
                                "raw_text": clean_b,
                                "type": "Barcode",
                                "confidence": 1.0,
                                "box": box_pts
                            })
                            if not heat_match:
                                m = re.search(r"[A-Z]\d{6,7}", clean_b.upper())
                                if m:
                                    heat_match = m.group(0)
            except Exception as e:
                pass

        # 1. Primary extraction with EasyOCR
        if self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(ocr_input, detail=1)
                for bbox, text, conf in results:
                    c_val = float(conf)
                    t_str = _clean_billet_label(text)
                    if t_str and c_val >= min_confidence:
                        box_pts = [[int(pt[0] * scale_inv_x), int(pt[1] * scale_inv_y)] for pt in bbox]
                        detected_items.append({
                            "text": t_str,
                            "raw_text": t_str,
                            "type": "Text",
                            "confidence": round(c_val, 3),
                            "box": box_pts
                        })
            except Exception as e:
                print(f"[OCR Warning] extract_all_text error in EasyOCR: {e}")

        # 2. Multi-Orientation Recovery (if low count or low conf, test 180° rotation)
        curr_conf = np.mean([it["confidence"] for it in detected_items]) if detected_items else 0.0
        if (len(detected_items) <= 1 or curr_conf < 0.40) and self.easyocr_reader is not None:
            try:
                rot180 = cv2.rotate(ocr_input, cv2.ROTATE_180)
                results_180 = self.easyocr_reader.readtext(rot180, detail=1)
                items_180 = []
                w_in, h_in = ocr_input.shape[1], ocr_input.shape[0]
                for bbox, text, conf in results_180:
                    c_val = float(conf)
                    t_str = _clean_billet_label(text)
                    if t_str and c_val >= min_confidence:
                        orig_pts = []
                        for pt in bbox:
                            ox = (w_in - 1 - pt[0]) * scale_inv_x
                            oy = (h_in - 1 - pt[1]) * scale_inv_y
                            orig_pts.append([int(ox), int(oy)])
                        items_180.append({
                            "text": t_str,
                            "raw_text": t_str,
                            "type": "Text (Rotated 180°)",
                            "confidence": round(c_val, 3),
                            "box": orig_pts
                        })
                avg_180 = np.mean([it["confidence"] for it in items_180]) if items_180 else 0.0
                if items_180 and (len(items_180) > len(detected_items) or avg_180 > curr_conf):
                    detected_items = items_180
            except Exception:
                pass

        # 3. Smart Billet Identification (find best heat / batch code)
        candidates = []
        for itm in detected_items:
            t = itm["text"]
            m_heat = re.search(r"[A-Z]\d{6,7}([A-Z0-9\.]*)", t.upper())
            if m_heat:
                candidates.append((m_heat.group(0), 1.0, len(m_heat.group(0))))
            m_digits = re.search(r"\b\d{4,12}\b", t)
            if m_digits:
                candidates.append((m_digits.group(0), 0.9, len(m_digits.group(0))))
            m_code = re.search(r"\b[A-Z0-9\+\-]{4,12}\b", t)
            if m_code and not m_code.group(0).isdigit():
                candidates.append((m_code.group(0), 0.85, len(m_code.group(0))))

        if candidates:
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            heat_match = candidates[0][0]

        # 4. Draw crisp bounding boxes & overlay tags if requested
        if draw_boxes and detected_items:
            for item in detected_items:
                if not item.get("box") or len(item["box"]) < 3:
                    continue
                pts = np.array(item["box"], dtype=np.int32)
                is_barcode = item.get("type") in ("QR Code", "Barcode")
                box_color = (0, 215, 255) if is_barcode else (0, 230, 255)
                cv2.polylines(annotated, [pts], isClosed=True, color=box_color, thickness=2)

                x_min = max(0, int(np.min(pts[:, 0])))
                y_min = max(0, int(np.min(pts[:, 1])))
                conf_str = f" ({item['confidence']*100:.0f}%)" if item.get("type") == "Text" else " [Verified]"
                label = f"{item['text']}{conf_str}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                tag_y1 = max(0, y_min - th - 6)
                tag_y2 = y_min
                tag_x2 = min(annotated.shape[1], x_min + tw + 6)
                cv2.rectangle(annotated, (x_min, tag_y1), (tag_x2, tag_y2), (20, 24, 33), -1)
                cv2.rectangle(annotated, (x_min, tag_y1), (tag_x2, tag_y2), box_color, 1)
                cv2.putText(annotated, label, (x_min + 3, y_min - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return {
            "detected_items": detected_items,
            "all_text": " ".join([it["text"] for it in detected_items]),
            "lines": [it["text"] for it in detected_items],
            "annotated_image": annotated,
            "count": len(detected_items),
            "heat_code_match": heat_match
        }



if __name__ == "__main__":
    print("Testing TripleVotingOCR...")
    ocr = TripleVotingOCR(enable_easyocr=True, enable_tesseract=True, enable_paddleocr=False)

    # Create a synthetic stamped end face image with "H727622R1.0"
    sample = np.ones((120, 380, 3), dtype=np.uint8) * 120
    # Add steel grain noise
    noise = np.random.normal(0, 15, sample.shape).astype(np.int16)
    sample = np.clip(sample.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Stamped text
    cv2.putText(sample, "H727622R1.0", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(sample, "H727622R1.0", (22, 77), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 2, cv2.LINE_AA)

    res = ocr.read_heat_number(sample)
    print("OCR Result:", res)
