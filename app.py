"""
Automated Billet Quality Inspection System - Streamlit Plant Dashboard
======================================================================
Real-time web monitoring and control dashboard for steel plant quality engineers.
Deployed at continuous casting discharge and rolling mill entry stations.

Features:
- Live camera stream / conveyor simulation with defect bounding boxes & HUD
- Real-time production KPIs: Total Inspected, Pass Count, Fail Count, Pass Rate %
- IS 2830:2012 tolerance monitoring (125 +/- 3 mm)
- Real-time inspection log viewer with live auto-refresh
- Operator settings sidebar: Tolerance sliders, confidence thresholds, shadow mode toggle
- Downloadable inspection reports (Excel / CSV)
"""

import os
import time
import glob
import cv2
import datetime
import pandas as pd
import numpy as np
import streamlit as st

import importlib
import ocr_engine as ocr_eng_mod
import alert_system as alert_sys_mod
import excel_logger as excel_log_mod
try:
    importlib.reload(ocr_eng_mod)
    importlib.reload(alert_sys_mod)
    importlib.reload(excel_log_mod)
except Exception:
    pass
from dimension_analyzer import DimensionAnalyzer
from ocr_engine import TripleVotingOCR
from defect_detector import SurfaceDefectDetector, DEFECT_COLORS
from excel_logger import billet_logger
from alert_system import alert_system, BilletAlertSystem
from calibrate_camera import check_is2830_tolerance

# Defensive fallback to eliminate any cached class AttributeError
if not hasattr(BilletAlertSystem, "get_plc_status"):
    def _fallback_get_plc_status(self):
        return {
            "plc_reject_kicker": getattr(self, "plc_reject_kicker", False),
            "plc_target": getattr(self, "plc_target", "Siemens S7-1500 (OPC-UA Node ns=2;s=PLC_Reject_Kicker)"),
            "last_dispatch_latency_ms": getattr(self, "plc_last_dispatch_ms", 0.0),
            "last_trigger_time": getattr(self, "plc_last_trigger_time", "None") or "None"
        }
    BilletAlertSystem.get_plc_status = _fallback_get_plc_status

if not hasattr(alert_system, "get_plc_status"):
    alert_system.get_plc_status = lambda: {
        "plc_reject_kicker": getattr(alert_system, "plc_reject_kicker", False),
        "plc_target": getattr(alert_system, "plc_target", "Siemens S7-1500 (OPC-UA Node ns=2;s=PLC_Reject_Kicker)"),
        "last_dispatch_latency_ms": getattr(alert_system, "plc_last_dispatch_ms", 0.0),
        "last_trigger_time": getattr(alert_system, "plc_last_trigger_time", "None") or "None"
    }

import io

def get_current_plc_status():
    """Safely retrieves PLC kicker status with 100% fail-safe fallback."""
    if hasattr(alert_system, "get_plc_status"):
        try:
            return alert_system.get_plc_status()
        except Exception:
            pass
    return {
        "plc_reject_kicker": False,
        "plc_target": "Siemens S7-1500 (OPC-UA Node ns=2;s=PLC_Reject_Kicker)",
        "last_dispatch_latency_ms": 0.0,
        "last_trigger_time": "None"
    }

def get_excel_export_bytes():
    """Safely returns Excel bytes, even if billet_log.xlsx is open or locked by MS Excel."""
    if os.path.exists(billet_logger.file_path):
        try:
            with open(billet_logger.file_path, "rb") as f:
                return f.read()
        except (PermissionError, IOError):
            pass
    try:
        df_all = billet_logger.get_recent_logs(limit=5000)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_all.to_excel(writer, sheet_name="InspectionLog", index=False)
        return output.getvalue()
    except Exception:
        return b""

# Page configuration
st.set_page_config(
    page_title="Billet Vision Inspector | Tata Steel & JSW Spec",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial CSS Styling
st.markdown("""
<style>
    /* Dark steel industrial aesthetic */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .status-pass-badge {
        background-color: #065F46;
        color: #34D399;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .status-fail-badge {
        background-color: #7F1D1D;
        color: #F87171;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .plant-header {
        background: linear-gradient(90deg, #1E293B 0%, #334155 100%);
        padding: 18px 24px;
        border-radius: 10px;
        border-left: 6px solid #38BDF8;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_inspection_engines():
    """Initializes and caches the CV and AI inspection modules."""
    dim_analyzer = DimensionAnalyzer()
    ocr_engine = TripleVotingOCR(enable_easyocr=True, enable_tesseract=False, enable_paddleocr=False)
    defect_detector = SurfaceDefectDetector(conf_thresh=0.50)
    return dim_analyzer, ocr_engine, defect_detector


dim_analyzer, ocr_engine, defect_detector = load_inspection_engines()

# Ensure latest extract_all_text method is always bound to ocr_engine
import types
ocr_engine.extract_all_text = types.MethodType(TripleVotingOCR.extract_all_text, ocr_engine)


def highlight_status(val):
    """Styles inspection status badges for dataframes."""
    color = '#166534' if val == 'PASS' else '#991B1B'
    bg = '#DCFCE7' if val == 'PASS' else '#FEE2E2'
    return f'background-color: {bg}; color: {color}; font-weight: bold;'


def render_log_table(placeholder, limit=12):
    """Renders recent inspection rows from Excel/SQLite into Streamlit placeholder."""
    recent_df = billet_logger.get_recent_logs(limit=limit)
    if not recent_df.empty:
        if hasattr(recent_df.style, "map"):
            styled_df = recent_df.style.map(highlight_status, subset=['Status'])
        else:
            styled_df = recent_df.style.applymap(highlight_status, subset=['Status'])
        placeholder.dataframe(styled_df, use_container_width=True)
    else:
        placeholder.caption("No inspection records logged yet. Start line or inspect billets above.")


def render_video_billet_table(placeholder, records):
    """Renders billets detected ONLY in the currently active video session."""
    if records:
        df = pd.DataFrame(records)
        if hasattr(df.style, "map"):
            styled_df = df.style.map(highlight_status, subset=['Status'])
        else:
            styled_df = df.style.applymap(highlight_status, subset=['Status'])
        placeholder.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        placeholder.info("⏳ Ready for video inspection. Click '▶️ Play Live Inspection Video' or '🚀 Process Entire Video' to inspect billets.")


# Initialize session state variables
if "running" not in st.session_state:
    st.session_state.running = False
if "sim_index" not in st.session_state:
    st.session_state.sim_index = 0
if "total_inspected" not in st.session_state:
    st.session_state.total_inspected = 0
if "pass_count" not in st.session_state:
    st.session_state.pass_count = 0
if "fail_count" not in st.session_state:
    st.session_state.fail_count = 0
if "defect_counts" not in st.session_state:
    st.session_state.defect_counts = {}
if "camera_session_texts" not in st.session_state:
    st.session_state.camera_session_texts = []
if "current_video_billets" not in st.session_state:
    st.session_state.current_video_billets = []
if "current_active_video_id" not in st.session_state:
    st.session_state.current_active_video_id = None
if "video_batch_result" not in st.session_state:
    st.session_state.video_batch_result = None

def record_session_text(text, confidence, tag="Text/Logo"):
    """Stores unique recognized text into the active camera session history."""
    clean_t = str(text).strip()
    if not clean_t:
        return
    recent = [entry.get("Text", entry.get("Detected Content", "")) for entry in st.session_state.camera_session_texts[-6:]]
    if clean_t not in recent:
        conf_float = float(confidence) if isinstance(confidence, (int, float)) else 0.85
        conf_display = f"{conf_float*100:.1f}%" if conf_float <= 1.0 else f"{conf_float:.1f}%"
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.camera_session_texts.append({
            "Time": now_str,
            "Type": tag,
            "Text": clean_t,
            "Confidence": conf_display
        })

# --- SIDEBAR CONTROLS ---
# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/steel-industry.png", width=54)
    st.title("Inspection Station")
    st.caption("Standard: IS 2830:2012 (Continuous Cast Billets)")

    source_option = st.radio(
        "Select Camera Input",
        [
            "📁 Upload Video or Image",
            "🧪 Calibrated Demo Billets",
            "🏭 USB Industrial Camera"
        ]
    )

    video_path = None
    frame_scrub_idx = 0
    total_video_frames = 1
    uploaded_video_file = None
    is_image_input = False

    if source_option == "📁 Upload Video or Image":
        uploaded_video_file = st.file_uploader(
            "Upload Billet Video or Image",
            type=["mp4", "avi", "mov", "mkv", "webm", "jpg", "png", "jpeg"]
        )

        repo_videos_dir = os.path.join(os.path.dirname(__file__), "Videos")
        repo_images_dir = os.path.join(os.path.dirname(__file__), "Billets Images")
        test_video_dir = os.path.join(os.path.dirname(__file__), "data", "test")
        os.makedirs(test_video_dir, exist_ok=True)

        local_repo_videos = {}
        if os.path.exists(repo_images_dir):
            for img_name in sorted(os.listdir(repo_images_dir)):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    local_repo_videos[f"🖼️ Billets Images/{img_name}"] = os.path.join(repo_images_dir, img_name)

        if os.path.exists(repo_videos_dir):
            for v_name in sorted(os.listdir(repo_videos_dir)):
                if v_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    local_repo_videos[f"📹 Videos/{v_name}"] = os.path.join(repo_videos_dir, v_name)

        if os.path.exists(test_video_dir):
            for v_name in sorted(os.listdir(test_video_dir)):
                if v_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")) and not v_name.startswith("temp_"):
                    local_repo_videos[f"🧪 {v_name}"] = os.path.join(test_video_dir, v_name)

        sample_options = ["None (Use Uploaded File)"] + list(local_repo_videos.keys())
        sample_choice = st.selectbox(
            "Or Test with Preloaded Conveyor Video or Billets Image",
            sample_options,
            help="Select any video or image from Billets Images, Videos, or sample recordings"
        )

        temp_video_dir = test_video_dir
        active_vid_id = None
        if uploaded_video_file is not None:
            is_vid = any(uploaded_video_file.name.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"])
            if is_vid:
                temp_path = os.path.join(temp_video_dir, "temp_uploaded_video.mp4")
                active_vid_id = f"upload_{uploaded_video_file.name}_{uploaded_video_file.size}"
                if (not os.path.exists(temp_path)
                    or os.path.getsize(temp_path) != uploaded_video_file.size
                    or st.session_state.get("current_active_video_id") != active_vid_id):
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_video_file.getvalue())
                video_path = temp_path
                is_image_input = False
            else:
                temp_img_path = os.path.join(temp_video_dir, f"temp_uploaded_{uploaded_video_file.name}")
                active_vid_id = f"image_{uploaded_video_file.name}_{uploaded_video_file.size}"
                if (not os.path.exists(temp_img_path)
                    or os.path.getsize(temp_img_path) != uploaded_video_file.size
                    or st.session_state.get("current_active_video_id") != active_vid_id):
                    with open(temp_img_path, "wb") as f:
                        f.write(uploaded_video_file.getvalue())
                video_path = temp_img_path
                is_image_input = True
        elif sample_choice != "None (Use Uploaded File)":
            video_path = local_repo_videos[sample_choice]
            active_vid_id = f"sample_{sample_choice}"
            is_image_input = any(video_path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"])

        # Wipe previous video data as soon as a new video or sample is uploaded/selected
        if active_vid_id is not None and active_vid_id != st.session_state.get("current_active_video_id"):
            st.session_state.current_active_video_id = active_vid_id
            st.session_state.video_batch_result = None
            st.session_state.current_video_billets = []
            st.session_state.total_inspected = 0
            st.session_state.pass_count = 0
            st.session_state.fail_count = 0
            st.session_state.defect_counts = {}

        if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            if is_image_input:
                st.caption(f"ℹ️ **{os.path.basename(video_path)}** — High-Resolution Billet Optical Snapshot (Instant OCR)")
            else:
                from video_processor import video_processor
                v_meta = video_processor.get_video_metadata(video_path)
                total_video_frames = max(1, v_meta["frame_count"])
                st.caption(f"ℹ️ **{os.path.basename(video_path)}** — {v_meta['resolution']} @ {v_meta['fps']} FPS | {v_meta['duration_seconds']}s ({total_video_frames} frames)")
                frame_scrub_idx = st.slider("🎞️ Frame Scrubber", 0, total_video_frames - 1, value=0)

        media_perspective = st.selectbox(
            "Camera Perspective",
            [
                "Auto-Detect (Smart)",
                "Side Camera (90° Longitudinal Bar)",
                "Front Camera (60° End Face + OCR)"
            ]
        )

    elif source_option == "🏭 USB Industrial Camera":
        st.markdown("##### 🎥 Camera Mode")
        cam_source_type = st.radio(
            "Select Camera Interface",
            [
                "📹 Live Continuous Camera Stream (Instant Real-Time OCR)",
                "🌐 Browser Webcam (Photo Snapshot)",
                "🖼️ Upload Test Image / Photo (Instant OCR)",
                "🧪 Pre-Loaded Stamped OCR Sample Card"
            ],
            help="Live Continuous Stream reads frames continuously and shows OCR on screen immediately."
        )
        cam_idx = st.number_input("USB Device Index", min_value=0, max_value=4, value=0, step=1,
                                  help="Index 0 is default laptop camera. Index 1 or 2 is external USB webcam.")
    else:
        cam_source_type = "None"
        cam_idx = 0

    # Clean expandable settings section
    with st.expander("⚙️ Advanced Parameters (Optional)", expanded=False):
        conf_thresh = st.slider("Defect Sensitivity Threshold", min_value=0.20, max_value=0.90, value=0.50, step=0.05)
        defect_detector.conf_thresh = conf_thresh

        nominal_dim = st.number_input("Nominal Size (mm)", min_value=100.0, max_value=200.0, value=125.0, step=1.0)
        tolerance = st.slider("IS 2830 Tolerance (± mm)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        dim_analyzer.nominal_size_mm = nominal_dim
        dim_analyzer.tolerance_mm = tolerance

        shadow_mode = st.toggle("Silent Shadow Mode", value=False)
        alert_system.set_shadow_mode(shadow_mode)

    st.divider()
    # Excel Report Download, Direct Open & Clear Button
    st.markdown("##### 📊 Production Quality Logs")
    st.caption(f"📁 **Active Path:** `{billet_logger.file_path}`\n\n*(Pure Local SSD • OneDrive Lock-Free • Real-Time COM Sync)*")
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if st.button("🟢 Open Live Excel", use_container_width=True):
            ok, target = billet_logger.launch_live_excel_window()
            if ok:
                st.toast(f"Opening {os.path.basename(target)} in Microsoft Excel...")
            else:
                st.error(f"Could not launch: {target}")
    with col_dl2:
        excel_bytes = get_excel_export_bytes()
        st.download_button(
            label="📥 Download",
            data=excel_bytes,
            file_name="steel_quality_records.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    if st.button("🗑️ Clear Logs", use_container_width=True):
        billet_logger.clear_log()
        st.session_state.total_inspected = 0
        st.session_state.pass_count = 0
        st.session_state.fail_count = 0
        st.session_state.defect_counts = {}
        st.session_state.video_batch_result = None
        st.rerun()

# --- MAIN DASHBOARD HEADER ---
st.markdown("""
<div class="plant-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0; font-size: 1.7rem; font-weight: 700; color: #FFFFFF;">
                Automated Billet Quality Inspection System
            </h2>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
                Dual Optical Vision (Front 60° + Side 90°) | IS 2830:2012 Compliance | Automated Defect Detection & OCR
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background-color: #0369A1; color: #E0F2FE; font-size: 0.8rem; font-weight: 600; padding: 6px 14px; border-radius: 20px;">
                STATION: FINISHING CONVEYOR 02
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- REAL-TIME KPI METRIC CARDS ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

total_insp = st.session_state.total_inspected
pass_c = st.session_state.pass_count
fail_c = st.session_state.fail_count
pass_rate = (pass_c / max(1, total_insp)) * 100.0 if total_insp > 0 else 100.0
plc_stat = get_current_plc_status()

with kpi_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Inspected</div>
        <div class="metric-value">{total_insp}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    rate_color = "#4ADE80" if pass_rate >= 90.0 else "#FBBF24" if pass_rate >= 80.0 else "#F87171"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Line Quality Yield</div>
        <div class="metric-value" style="color: {rate_color};">{pass_rate:.1f}% <span style="font-size: 0.85rem; color: #94A3B8;">({pass_c}P / {fail_c}R)</span></div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    total_defects = sum(st.session_state.defect_counts.values())
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Defects Detected</div>
        <div class="metric-value" style="color: {'#F87171' if total_defects > 0 else '#4ADE80'};">{total_defects}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    kicker_active = plc_stat["plc_reject_kicker"]
    k_color = "#F87171" if kicker_active else "#34D399"
    k_text = "ACTIVE (Diverting)" if kicker_active else "STANDBY (<200ms)"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">PLC Reject Kicker</div>
        <div class="metric-value" style="font-size: 1.45rem; color: {k_color};">{k_text}</div>
        <div style="font-size: 0.72rem; color: #94A3B8;">Siemens S7-1500 | SQLite Synced</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --- LIVE CAMERA FEED & INSPECTION DETAIL ---
col_feed, col_details = st.columns([1.6, 1.0])

# Prepare test frames
test_dir = os.path.join(os.path.dirname(__file__), "data", "test")
front_images = sorted([f for f in glob.glob(os.path.join(test_dir, "*.jpg")) if "side" not in os.path.basename(f)])
side_images = sorted([f for f in glob.glob(os.path.join(test_dir, "*side*.jpg"))])
if not front_images or not side_images:
    from generate_mock_data import generate_all_test_billets
    generate_all_test_billets()
    front_images = sorted([f for f in glob.glob(os.path.join(test_dir, "*.jpg")) if "side" not in os.path.basename(f)])
    side_images = sorted([f for f in glob.glob(os.path.join(test_dir, "*side*.jpg"))])

# Acquire frames based on selection
active_front_frame = None
active_side_frame = None

if source_option == "📁 Upload Video or Image":
    from video_processor import video_processor
    target_frame = None
    if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        if is_image_input:
            target_frame = cv2.imread(video_path)
        else:
            target_frame = video_processor.extract_frame_at(video_path, frame_scrub_idx)
    elif uploaded_video_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_video_file.getvalue()), dtype=np.uint8)
        target_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        is_image_input = True

    if target_frame is not None:
        if is_image_input:
            active_front_frame = target_frame
            active_side_frame = None
        elif media_perspective == "Side Camera (90° Longitudinal Bar)":
            active_side_frame = target_frame
            active_front_frame = None
        elif media_perspective == "Front Camera (60° End Face + OCR)":
            active_front_frame = target_frame
            active_side_frame = None
        else:
            detected_v = video_processor.detect_camera_view(target_frame)
            if detected_v == "side":
                active_side_frame = target_frame
                active_front_frame = None
            else:
                active_front_frame = target_frame
                active_side_frame = None
    else:
        active_front_frame = cv2.imread(front_images[0])
        active_side_frame = cv2.imread(side_images[0])

elif source_option == "🏭 USB Industrial Camera":
    cam_picture = None
    if "Live Continuous Camera" in cam_source_type:
        with col_feed:
            st.subheader("📹 Real-Time Live Optical Stream (Instant OCR & Barcode)")
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.0, 1.0])
            with col_ctrl1:
                start_live = st.button("▶️ Start Live Real-Time OCR Stream", type="primary", use_container_width=True)
            with col_ctrl2:
                grab_frame = st.button("📸 Grab Current Frame", use_container_width=True)
            with col_ctrl3:
                stop_stream = st.button("⏹️ Stop Stream", use_container_width=True)

            stream_box = st.empty()
            live_hud = st.empty()

            if start_live:
                st.info("🔴 Live Camera Feed Running. Hold any paper, card, phone, or stamped metal in front of the camera!")
                cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                for _ in range(5):
                    cap.read()

                cached_ocr = {"detected_items": [], "all_text": "", "lines": [], "count": 0, "heat_code_match": None}
                last_ocr_t = 0.0

                try:
                    f_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                        f_count += 1

                        now = time.time()
                        if now - last_ocr_t >= 1.0:
                            last_ocr_t = now
                            cached_ocr = ocr_engine.extract_all_text(frame, min_confidence=0.15, draw_boxes=False, max_dim=400)
                            for itm in cached_ocr.get("detected_items", []):
                                itm_raw = itm.get("raw_text", itm["text"])
                                is_h = bool(cached_ocr.get("heat_code_match") and itm_raw in cached_ocr["heat_code_match"])
                                itm_type = itm.get("type", "Text")
                                if is_h:
                                    itm_type = "Stamped Heat Code"
                                record_session_text(itm_raw, itm["confidence"], tag=itm_type)

                        # Render overlay on live frame
                        vis = frame.copy()
                        for item in cached_ocr.get("detected_items", []):
                            if not item.get("box") or len(item["box"]) < 3:
                                continue
                            pts = np.array(item["box"], dtype=np.int32)
                            is_barcode = item.get("type") in ("QR Code", "Barcode")
                            box_color = (0, 215, 255) if is_barcode else (0, 230, 255)
                            cv2.polylines(vis, [pts], True, box_color, 2)
                            x_min = max(0, int(np.min(pts[:, 0])))
                            y_min = max(0, int(np.min(pts[:, 1])))
                            conf_str = f" ({item['confidence']*100:.0f}%)" if item.get("type") == "Text" else " [Verified]"
                            lbl = f"{item['text']}{conf_str}"
                            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            cv2.rectangle(vis, (x_min, max(0, y_min - th - 6)), (x_min + tw + 6, y_min), (20, 24, 33), -1)
                            cv2.rectangle(vis, (x_min, max(0, y_min - th - 6)), (x_min + tw + 6, y_min), box_color, 1)
                            cv2.putText(vis, lbl, (x_min + 3, y_min - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                        # Top banner
                        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (15, 23, 42), -1)
                        cv2.line(vis, (0, 40), (vis.shape[1], 40), (0, 230, 255), 2)
                        heat_lbl = f" | Heat: {cached_ocr['heat_code_match']}" if cached_ocr.get('heat_code_match') else ""
                        cv2.putText(vis, f"LIVE OPTICAL OCR & BARCODE | Items: {cached_ocr['count']}{heat_lbl}", (12, 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)

                        stream_box.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

                        if cached_ocr["count"] > 0:
                            live_hud.markdown(
                                f"<div style='background-color:#0F172A; border-left: 5px solid #00E5FF; padding:10px 14px; border-radius:6px; margin-top:8px;'>"
                                f"<b style='color:#38BDF8;'>📝 Live Detected Items:</b> <span style='color:#F8FAFC; font-weight:bold; font-size:1.15rem;'>{cached_ocr['all_text']}</span><br>"
                                f"<span style='font-size:0.8rem; color:#94A3B8;'>Items: <b>{cached_ocr['count']}</b> | Primary Code: <b style='color:#4ADE80;'>{cached_ocr.get('heat_code_match') or (cached_ocr['lines'][0] if cached_ocr['lines'] else 'N/A')}</b></span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            live_hud.markdown(
                                f"<div style='background-color:#1E293B; border-left: 5px solid #F59E0B; padding:10px 14px; border-radius:6px; margin-top:8px;'>"
                                f"<b style='color:#FBBF24;'>🔍 Camera Ready:</b> <span style='color:#CBD5E1;'>Point camera at any document, brand label, barcode, or stamped metal...</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        time.sleep(0.02)
                finally:
                    cap.release()

            # Session text & barcode history display
            if st.session_state.camera_session_texts:
                with st.expander(f"📜 Camera Session Stored History ({len(st.session_state.camera_session_texts)} items)", expanded=True):
                    import pandas as pd
                    df_session = pd.DataFrame(st.session_state.camera_session_texts)
                    st.dataframe(df_session, use_container_width=True, height=200)
                    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
                    with col_d1:
                        csv_data = df_session.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download CSV", data=csv_data, file_name="camera_session_ocr.csv", mime="text/csv", use_container_width=True)
                    with col_d2:
                        import io
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                            df_session.to_excel(writer, index=False, sheet_name="CameraSession")
                        st.download_button("📊 Download Excel", data=buf.getvalue(), file_name="camera_session_ocr.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                    with col_d3:
                        if st.button("🗑️ Clear Session Memory", use_container_width=True):
                            st.session_state.camera_session_texts = []
                            st.rerun()
            else:
                try:
                    cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
                    if cap.isOpened():
                        for _ in range(5):
                            ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None and frame.mean() > 2.0:
                            active_front_frame = frame
                            active_side_frame = None
                except Exception:
                    pass
                if active_front_frame is None:
                    st.info("💡 Click **'▶️ Start Live Real-Time OCR Stream'** above to start streaming live with instant OCR text detection!")

    elif "Browser Webcam" in cam_source_type:
        with col_feed:
            st.subheader("📹 Optical Station Live Stream")
            st.info("💡 **Camera Instructions:** Position any text, document, card, iron rod, or stamped metal in front of your camera and click **'Take Photo'** below to extract all text.")
            cam_picture = st.camera_input("📸 Live Webcam Viewfinder (Click 'Take Photo' to Inspect)")
        if cam_picture is not None:
            file_bytes = np.asarray(bytearray(cam_picture.read()), dtype=np.uint8)
            active_front_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            active_side_frame = None
    elif "Upload Test Image" in cam_source_type:
        with col_feed:
            st.subheader("📹 Optical Station Live Stream")
            st.info("📁 **Upload Photo / Document:** Select any image file from your device to extract and highlight all text.")
            uploaded_cam_file = st.file_uploader("Upload Image for Instant OCR", type=["jpg", "png", "jpeg", "webp"])
            if uploaded_cam_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_cam_file.read()), dtype=np.uint8)
                active_front_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                active_side_frame = None
    elif "Sample Card" in cam_source_type:
        with col_feed:
            st.subheader("📹 Optical Station Live Stream")
            st.success("🧪 Pre-loaded Stamped OCR Test Sample Card Active")
        sample_card_path = os.path.join(test_dir, "sample_text_card.jpg")
        if os.path.exists(sample_card_path):
            active_front_frame = cv2.imread(sample_card_path)
            active_side_frame = None

    if active_front_frame is None and ("Live Continuous" not in cam_source_type) and ("Browser Webcam" not in cam_source_type or cam_picture is None) and ("Upload" not in cam_source_type) and ("Sample" not in cam_source_type):
        with col_feed:
            st.warning(f"⚠️ USB camera {cam_idx} not responding or dark. Ensure camera is connected, privacy shutter is open, or switch to 'Live Continuous Camera Stream'.")
        active_front_frame = None
        active_side_frame = None
else:
    # Calibrated demo billets (simulation)
    active_front_frame = cv2.imread(front_images[st.session_state.sim_index % len(front_images)])
    active_side_frame = cv2.imread(side_images[st.session_state.sim_index % len(side_images)])

# Run Inspection
primary_frame = active_front_frame if active_front_frame is not None else active_side_frame

if primary_frame is not None:
    t0 = time.time()

    # 1. Front View Dimensions (60° tilt)
    if active_front_frame is not None:
        dim_front = dim_analyzer.measure_dimensions(active_front_frame, camera_view="front")
        width_act = dim_front.get("width_actual_mm", 0.0)
        height_front_act = dim_front.get("height_actual_mm", 0.0)
        dim_front_pass = dim_front.get("is_pass", False)
        front_detected = dim_front.get("detected", False)
    else:
        dim_front = None
        width_act = dim_analyzer.nominal_size_mm
        height_front_act = 0.0
        dim_front_pass = True
        front_detected = False

    # 2. Side View Dimensions (90° orthogonal)
    if active_side_frame is not None:
        dim_side = dim_analyzer.measure_dimensions(active_side_frame, camera_view="side")
        height_side_act = dim_side.get("height_actual_mm", height_front_act)
        dim_side_pass = dim_side.get("is_pass", False)
        side_detected = dim_side.get("detected", False)
        if active_front_frame is None:
            width_act = dim_analyzer.nominal_size_mm
    else:
        dim_side = None
        height_side_act = height_front_act
        dim_side_pass = True
        side_detected = False

    billet_present = front_detected or side_detected

    # Cross-view height validation (if dual views active)
    if dim_front and dim_side and front_detected and side_detected:
        cross_val = dim_analyzer.cross_validate_dual_views(dim_front, dim_side)
    else:
        cross_val = {"dual_pass": dim_front_pass and dim_side_pass, "delta_height_mm": 0.0, "status": "Single Camera View"}

    # 3. Defect Detection
    defects_front = defect_detector.detect_defects(active_front_frame, camera_view="front") if (active_front_frame is not None and front_detected) else []
    defects_side = defect_detector.detect_defects(active_side_frame, camera_view="side") if (active_side_frame is not None and side_detected) else []
    all_defects = defects_front + defects_side
    defect_pass = (len(all_defects) == 0)
    defect_strings = [f"{d['class_name']} ({d['confidence']:.2f})" for d in all_defects]

    # 4. OCR Heat Stamping & Full Text Extraction
    ocr_all_results = None
    heat_number = "NO BILLET RECOGNISED (STANDBY)"
    ocr_res = {"heat_number": heat_number, "confidence": 0.0, "method": "Standby", "is_valid_format": False}

    if is_image_input and active_front_frame is not None:
        ocr_all_results = ocr_engine.extract_all_text(active_front_frame, min_confidence=0.15, draw_boxes=True, max_dim=1280)
        if ocr_all_results and ocr_all_results.get("count", 0) > 0:
            billet_present = True
            top_candidate = ocr_all_results.get("heat_code_match")
            if not top_candidate and ocr_all_results.get("detected_items"):
                top_candidate = ocr_all_results["detected_items"][0]["text"]
            heat_number = top_candidate or "BILLET_TAG"
            top_conf = ocr_all_results["detected_items"][0]["confidence"] if ocr_all_results.get("detected_items") else 0.95
            ocr_res = {
                "heat_number": heat_number,
                "confidence": top_conf,
                "method": f"Full-Frame Optical OCR ({ocr_all_results['count']} texts detected)",
                "is_valid_format": True
            }
            overall_status = "PASS"
            dim_pass = True
        else:
            heat_number = "UNREADABLE_OCR"
            ocr_res = {"heat_number": heat_number, "confidence": 0.0, "method": "No Text Detected", "is_valid_format": False}
    elif not billet_present:
        heat_number = "NO BILLET RECOGNISED (STANDBY)"
        ocr_res = {"heat_number": heat_number, "confidence": 0.0, "method": "Standby", "is_valid_format": False}
    elif dim_front and dim_front.get("shape") == "cylindrical":
        cyl_diam = int(round(dim_front.get("diameter_mm", width_act)))
        heat_number = f"CYL-R{cyl_diam}"
        ocr_res = {
            "heat_number": heat_number,
            "confidence": 0.95,
            "method": "Cylindrical Metrology",
            "is_valid_format": True
        }
    elif source_option == "🏭 USB Industrial Camera" and active_front_frame is not None:
        ocr_all_results = ocr_engine.extract_all_text(active_front_frame, min_confidence=0.15, draw_boxes=True)
        if ocr_all_results.get("heat_code_match"):
            heat_number = ocr_all_results["heat_code_match"]
            ocr_res = {
                "heat_number": heat_number,
                "confidence": 0.95,
                "method": "Triple-Voting Consensus",
                "is_valid_format": True
            }
        elif ocr_all_results.get("lines"):
            heat_number = ocr_all_results["lines"][0]
            top_item = ocr_all_results["detected_items"][0]
            ocr_res = {
                "heat_number": heat_number,
                "confidence": top_item["confidence"],
                "method": f"Full-Frame OCR ({ocr_all_results['count']} texts detected)",
                "is_valid_format": False
            }
        for it in ocr_all_results.get("detected_items", []):
            is_h = bool(ocr_all_results.get("heat_code_match") and it["text"] in ocr_all_results["heat_code_match"])
            record_session_text(it["text"], it["confidence"], tag="Stamped Heat Code" if is_h else "Text/Brand")
    elif active_front_frame is not None:
        contour_box = dim_front.get("contour_box") if dim_front else None
        h_f, w_f = active_front_frame.shape[:2]
        if contour_box is not None:
            bx1 = max(0, int(np.min(contour_box[:, 0])))
            by1 = max(0, int(np.min(contour_box[:, 1])))
            bx2 = min(w_f, int(np.max(contour_box[:, 0])))
            by2 = min(h_f, int(np.max(contour_box[:, 1])))
            roi_end = active_front_frame[by1:by2, bx1:bx2]
        else:
            roi_end = active_front_frame[int(h_f * 0.20):int(h_f * 0.80), int(w_f * 0.15):int(w_f * 0.85)]
        ocr_res = ocr_engine.read_heat_number(roi_end)
        heat_number = ocr_res["heat_number"]
    else:
        ocr_res = {"heat_number": "N/A (Side View)", "confidence": 1.0, "method": "Side Orthogonal Camera", "is_valid_format": True}
        heat_number = "N/A (Side View)"

    # 5. Status Decision
    if is_image_input:
        if ocr_all_results and ocr_all_results.get("count", 0) > 0:
            overall_status = "PASS"
            dim_pass = True
            billet_present = True
        else:
            overall_status = "FAIL"
            dim_pass = False
    elif not billet_present:
        if source_option == "🏭 USB Industrial Camera" and ocr_all_results and ocr_all_results["count"] > 0:
            overall_status = "PASS"
            dim_pass = True
        else:
            overall_status = "STANDBY"
            dim_pass = False
    else:
        dim_pass = dim_front_pass and dim_side_pass
        overall_status = "PASS" if (dim_pass and defect_pass) else "FAIL"

    cycle_time_ms = (time.time() - t0) * 1000.0

    # Auto-log if running and newly triggered
    if st.session_state.running and overall_status in ("PASS", "FAIL"):
        st.session_state.total_inspected += 1
        if overall_status == "PASS":
            st.session_state.pass_count += 1
        else:
            st.session_state.fail_count += 1

        for d in all_defects:
            c = d["class_name"]
            st.session_state.defect_counts[c] = st.session_state.defect_counts.get(c, 0) + 1

        billet_logger.log_inspection(
            heat_number=heat_number,
            width_mm=width_act,
            height_mm=height_side_act,
            defects=defect_strings,
            status=overall_status,
            rhomboidity_mm=dim_front.get("rhomboidity_delta_mm", 0.0) if dim_front else 0.0
        )

        if overall_status == "FAIL":
            alert_system.trigger_failure_alert(
                heat_number=heat_number,
                failure_reason=f"Status: {overall_status} (Defects: {len(all_defects)})",
                width_mm=width_act,
                height_mm=height_side_act,
                defects=defect_strings
            )

    # Render overlays on camera views
    annotated_front = None
    if active_front_frame is not None:
        if is_image_input and ocr_all_results and ocr_all_results.get("annotated_image") is not None:
            annotated_front = ocr_all_results["annotated_image"].copy()
            annotated_front = defect_detector.draw_defect_boxes(annotated_front, defects_front)
        elif source_option == "📁 Upload Video or Image":
            insp_data = {
                "detected": front_detected,
                "shape": dim_front.get("shape", "square") if dim_front else "square",
                "camera_view": "front",
                "heat_number": heat_number,
                "ocr_confidence": ocr_res.get("confidence", 0.0),
                "width_mm": width_act,
                "height_mm": height_front_act,
                "diameter_mm": dim_front.get("diameter_mm", width_act) if dim_front else width_act,
                "ovality_mm": dim_front.get("ovality_mm", 0.0) if dim_front else 0.0,
                "dim_pass": dim_front_pass,
                "contour_box": dim_front.get("contour_box") if dim_front else None,
                "contour": dim_front.get("contour") if dim_front else None,
                "defects": defects_front,
                "status": overall_status,
                "rhomboidity_delta_mm": dim_front.get("rhomboidity_delta_mm", 0.0) if dim_front else 0.0,
                "latency_ms": round(cycle_time_ms, 1)
            }
            annotated_front = video_processor.render_overlay(active_front_frame, insp_data, frame_num=frame_scrub_idx)
        else:
            if ocr_all_results and ocr_all_results.get("annotated_image") is not None:
                annotated_front = ocr_all_results["annotated_image"].copy()
            else:
                annotated_front = active_front_frame.copy()
            annotated_front = defect_detector.draw_defect_boxes(annotated_front, defects_front)
            cbox = dim_front.get("contour_box") if dim_front else None
            if cbox is not None:
                cv2.drawContours(annotated_front, [cbox], 0, (0, 220, 0) if dim_front_pass else (0, 0, 240), 2)

    annotated_side = None
    if active_side_frame is not None:
        if source_option == "📁 Upload Video or Image":
            insp_data = {
                "detected": side_detected,
                "shape": dim_side.get("shape", "square") if dim_side else "square",
                "camera_view": "side",
                "heat_number": "N/A (Side View)",
                "ocr_confidence": 1.0,
                "width_mm": width_act,
                "height_mm": height_side_act,
                "diameter_mm": width_act,
                "ovality_mm": 0.0,
                "dim_pass": dim_side_pass,
                "contour_box": dim_side.get("contour_box") if dim_side else None,
                "contour": dim_side.get("contour") if dim_side else None,
                "defects": defects_side,
                "status": overall_status,
                "rhomboidity_delta_mm": 0.0,
                "latency_ms": round(cycle_time_ms, 1)
            }
            annotated_side = video_processor.render_overlay(active_side_frame, insp_data, frame_num=frame_scrub_idx)
        else:
            annotated_side = active_side_frame.copy()
            annotated_side = defect_detector.draw_defect_boxes(annotated_side, defects_side)
            contour_side = dim_side.get("contour_box") if dim_side else None
            if contour_side is not None:
                cv2.drawContours(annotated_side, [contour_side], 0, (0, 220, 0) if dim_side_pass else (0, 0, 240), 2)

    with col_feed:
        if is_image_input:
            st.subheader("🔬 Billet Optical Text & Defect Inspection")
            if annotated_front is not None:
                front_rgb = cv2.cvtColor(annotated_front, cv2.COLOR_BGR2RGB)
                txt_c = ocr_all_results["count"] if ocr_all_results else 0
                st.image(front_rgb, use_container_width=True, caption=f"Billet Image Overlay | Detected Texts: {txt_c} | Latency: {cycle_time_ms:.1f}ms")
            else:
                st.info("No billet image loaded.")
        elif source_option != "🏭 USB Industrial Camera":
            st.subheader("📹 Optical Station Live Stream")
            tab_front, tab_side = st.tabs(["📷 Front Camera (60° Tilt | OCR + End Face)", "📸 Side Camera (90° Orthogonal | Longitudinal Body)"])

            with tab_front:
                if annotated_front is not None:
                    front_rgb = cv2.cvtColor(annotated_front, cv2.COLOR_BGR2RGB)
                    st.image(front_rgb, use_container_width=True, caption=f"Front View @ 60° | Dim: {width_act:.1f}x{height_front_act:.1f}mm | Latency: {cycle_time_ms:.1f}ms")
                else:
                    st.info("Front camera view not active for current perspective.")

            with tab_side:
                if annotated_side is not None:
                    side_rgb = cv2.cvtColor(annotated_side, cv2.COLOR_BGR2RGB)
                    st.image(side_rgb, use_container_width=True, caption=f"Side View @ 90° | True Orthogonal Height: {height_side_act:.1f}mm | Latency: {cycle_time_ms:.1f}ms")
                else:
                    st.info("Side camera view not active for current perspective.")
        else:
            if annotated_front is not None:
                st.markdown("##### 🔬 Inspected Camera Snapshot (With OCR Bounding Boxes)")
                front_rgb = cv2.cvtColor(annotated_front, cv2.COLOR_BGR2RGB)
                txt_c = ocr_all_results["count"] if ocr_all_results else 0
                st.image(front_rgb, use_container_width=True, caption=f"Camera / Document Overlay | Detected Texts: {txt_c} | Latency: {cycle_time_ms:.1f}ms")

        if source_option == "📁 Upload Video or Image" and (video_path or uploaded_video_file):
            st.markdown("---")
            if is_image_input:
                col_ibtn1, col_ibtn2 = st.columns([1.5, 1.0])
                with col_ibtn1:
                    log_img_btn = st.button("📥 Log Billet to Master Excel Sheet", type="primary", use_container_width=True)
                with col_ibtn2:
                    if st.button("🔄 Re-Scan Image", use_container_width=True):
                        st.rerun()
                if log_img_btn:
                    st.session_state.total_inspected += 1
                    if overall_status == "PASS":
                        st.session_state.pass_count += 1
                    else:
                        st.session_state.fail_count += 1
                    billet_logger.log_inspection(
                        heat_number=heat_number,
                        width_mm=width_act,
                        height_mm=height_side_act or 150.0,
                        defects=defect_strings,
                        status=overall_status,
                        rhomboidity_mm=0.0
                    )
                    st.success(f"✅ Recorded Billet '{heat_number}' into Excel master spreadsheet!")
                    st.rerun()
            else:
                col_vbtn1, col_vbtn2, col_vbtn3 = st.columns([1.3, 1.1, 1.0])
                with col_vbtn1:
                    play_live = st.button("▶️ Play Live Inspection Video", type="primary", use_container_width=True)
                with col_vbtn2:
                    run_video_batch = st.button("🚀 Process Entire Video", use_container_width=True)
                with col_vbtn3:
                    log_current_frame = st.button("📥 Log Current Frame", use_container_width=True)

            if play_live and video_path:
                st.info("🔴 Live Stream Playing: Real-time neural inference & dimensioning active on video feed.")
                stream_box = st.empty()
                live_hud = st.empty()
                st.markdown("##### 📋 Live Conveyor Billet Telemetry (Current Video Session)")
                live_table_box = st.empty()
                st.session_state.current_video_billets = []
                render_video_billet_table(live_table_box, st.session_state.current_video_billets)

                cap_stream = cv2.VideoCapture(video_path)
                f_count = 0
                step = 2  # Sample every 2nd frame for smooth 20+ FPS streaming in browser

                fps = cap_stream.get(cv2.CAP_PROP_FPS) or 30.0
                cooldown_frames = int(1.2 * fps)  # Debounce per passing billet (1.2s window)
                last_logged_frame = -cooldown_frames
                
                try:
                    while cap_stream.isOpened():
                        ret, f_raw = cap_stream.read()
                        if not ret:
                            break
                        f_count += 1
                        if f_count % step != 0:
                            continue

                        # Inspect frame live
                        live_insp = video_processor.inspect_frame(f_raw, camera_view="both")
                        vis_live = video_processor.render_overlay(f_raw, live_insp, frame_num=f_count)
                        vis_live_rgb = cv2.cvtColor(vis_live, cv2.COLOR_BGR2RGB)
                        stream_box.image(vis_live_rgb, use_container_width=True)

                        # REAL-TIME LOGGING TO EXCEL AS VIDEO PROCESSES
                        if live_insp.get("detected") and (f_count - last_logged_frame >= cooldown_frames):
                            last_logged_frame = f_count

                            # 1. Append directly to Excel & SQLite dual-store in real time
                            billet_logger.log_inspection(
                                heat_number=live_insp["heat_number"],
                                width_mm=live_insp["width_mm"],
                                height_mm=live_insp["height_mm"],
                                defects=live_insp["defect_strings"],
                                status=live_insp["status"],
                                rhomboidity_mm=live_insp.get("rhomboidity_delta_mm", 0.0)
                            )

                            # 2. Update KPI metrics
                            st.session_state.total_inspected += 1
                            if live_insp["status"] == "PASS":
                                st.session_state.pass_count += 1
                            else:
                                st.session_state.fail_count += 1
                                alert_system.trigger_failure_alert(
                                    heat_number=live_insp["heat_number"],
                                    failure_reason=f"Status: FAIL (Defects: {len(live_insp['defects'])})",
                                    width_mm=live_insp["width_mm"],
                                    height_mm=live_insp["height_mm"],
                                    defects=live_insp["defect_strings"]
                                )

                            for d in live_insp["defects"]:
                                c = d["class_name"]
                                st.session_state.defect_counts[c] = st.session_state.defect_counts.get(c, 0) + 1

                            # 3. Update the live table for current video
                            dim_str = f"Ø{live_insp.get('diameter_mm', live_insp['width_mm']):.1f} mm ({live_insp['width_mm']:.1f} × {live_insp['height_mm']:.1f})" if live_insp.get("shape") == "cylindrical" else f"{live_insp['width_mm']:.1f} × {live_insp['height_mm']:.1f}"
                            b_rec = {
                                "Billet #": len(st.session_state.current_video_billets) + 1,
                                "Frame": f_count,
                                "Time": f"{round(f_count / fps, 2)}s",
                                "Heat Number": live_insp["heat_number"],
                                "Dimensions (mm)": dim_str,
                                "Defects": ", ".join(live_insp["defect_strings"]) if live_insp["defect_strings"] else "None (Clean)",
                                "Status": live_insp["status"]
                            }
                            st.session_state.current_video_billets.append(b_rec)
                            render_video_billet_table(live_table_box, st.session_state.current_video_billets)

                        # Update live status bar
                        if not live_insp.get("detected"):
                            live_hud.markdown(
                                f"<div style='background-color:#1E293B; border-left: 5px solid #64748B; padding:8px 14px; border-radius:6px; margin-bottom:8px;'>"
                                f"<b>Frame {f_count}:</b> Status: <span style='color:#94A3B8; font-weight:bold;'>NO BILLET RECOGNISED (STANDBY)</span> | "
                                f"<span style='color:#CBD5E1;'>Conveyor Monitoring • Waiting for billet to arrive at inspection table...</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            d_names = [d["class_name"] for d in live_insp["defects"]]
                            badge_color = "#10B981" if live_insp["status"] == "PASS" else "#EF4444"
                            shape_lbl = "CYLINDRICAL" if live_insp.get("shape") == "cylindrical" else "SQUARE"
                            dim_lbl = f"Ø{live_insp.get('diameter_mm', live_insp['width_mm']):.1f} mm" if live_insp.get("shape") == "cylindrical" else f"{live_insp['width_mm']:.1f} × {live_insp['height_mm']:.1f} mm"
                            live_hud.markdown(
                                f"<div style='background-color:#1E293B; border-left: 5px solid {badge_color}; padding:8px 14px; border-radius:6px; margin-bottom:8px;'>"
                                f"<b>Frame {f_count}:</b> Status: <span style='color:{badge_color}; font-weight:bold;'>{live_insp['status']} ({shape_lbl})</span> | "
                                f"Heat: <b>{live_insp['heat_number']}</b> | "
                                f"Dim: <b>{dim_lbl}</b> | "
                                f"Defects: <b>{', '.join(d_names) if d_names else 'None (Clean)'}</b> | "
                                f"📊 <span style='color:#38BDF8;'>Excel Updating Real-Time ({st.session_state.total_inspected} Logged)</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        time.sleep(0.015)
                finally:
                    cap_stream.release()
                    st.success(f"✅ Live video inspection complete! All passing billets logged to Excel (Total: {st.session_state.total_inspected}).")
                    render_video_billet_table(live_table_box, st.session_state.current_video_billets)

            if log_current_frame:
                st.session_state.total_inspected += 1
                if overall_status == "PASS":
                    st.session_state.pass_count += 1
                else:
                    st.session_state.fail_count += 1
                billet_logger.log_inspection(
                    heat_number=heat_number,
                    width_mm=width_act,
                    height_mm=height_side_act,
                    defects=defect_strings,
                    status=overall_status,
                    rhomboidity_mm=dim_front.get("rhomboidity_delta_mm", 0.0) if dim_front else 0.0
                )
                st.success(f"Recorded Billet #{st.session_state.total_inspected} as {overall_status} in steel_quality_records.xlsx!")
                st.rerun()

            if run_video_batch and video_path:
                st.markdown("##### 📋 Live Video Inspection Telemetry (Current Video Session)")
                batch_hud = st.empty()
                prog_bar = st.progress(0.0)
                status_text = st.empty()
                batch_table_box = st.empty()
                st.session_state.current_video_billets = []
                render_video_billet_table(batch_table_box, st.session_state.current_video_billets)

                def on_progress(f_idx, f_total, n_billets, new_billet=None):
                    pct = f_idx / max(1, f_total)
                    prog_bar.progress(pct)
                    status_text.caption(f"Inspecting frame {f_idx}/{f_total} — Detected Billets Logged: {n_billets}")
                    if new_billet is not None:
                        st.session_state.total_inspected += 1
                        if new_billet.get("status") == "PASS":
                            st.session_state.pass_count += 1
                        else:
                            st.session_state.fail_count += 1
                        for d in new_billet.get("defects", []):
                            c = d.get("class_name", "defect")
                            st.session_state.defect_counts[c] = st.session_state.defect_counts.get(c, 0) + 1
                        
                        is_cyl = (new_billet.get("shape") == "cylindrical")
                        dim_str = f"Ø{new_billet.get('diameter_mm', new_billet.get('width_mm', 0.0)):.1f} mm ({new_billet.get('width_mm', 0.0):.1f} × {new_billet.get('height_mm', 0.0):.1f})" if is_cyl else f"{new_billet.get('width_mm', 0.0):.1f} × {new_billet.get('height_mm', 0.0):.1f}"
                        b_entry = {
                            "Billet #": len(st.session_state.current_video_billets) + 1,
                            "Frame": f_idx,
                            "Time": f"{new_billet.get('timestamp_sec', round(f_idx / 30.0, 2))}s",
                            "Heat Number": new_billet.get('heat_number', 'N/A'),
                            "Dimensions (mm)": dim_str,
                            "Defects": ", ".join(new_billet.get('defect_strings', [])) if new_billet.get('defect_strings') else "None (Clean)",
                            "Status": new_billet.get('status', 'STANDBY')
                        }
                        st.session_state.current_video_billets.append(b_entry)
                        render_video_billet_table(batch_table_box, st.session_state.current_video_billets)
                        badge_c = "#10B981" if new_billet.get('status') == "PASS" else "#EF4444"
                        dim_hud = f"Ø{new_billet.get('diameter_mm', new_billet.get('width_mm', 0.0)):.1f} mm" if is_cyl else f"{new_billet.get('width_mm', 0.0):.1f} × {new_billet.get('height_mm', 0.0):.1f} mm"
                        batch_hud.markdown(
                            f"<div style='background-color:#1E293B; border-left: 5px solid {badge_c}; padding:8px 14px; border-radius:6px; margin-bottom:8px;'>"
                            f"<b>Frame {f_idx}:</b> Status: <span style='color:{badge_c}; font-weight:bold;'>{new_billet['status']} ({'CYLINDRICAL' if is_cyl else 'SQUARE'})</span> | "
                            f"Heat: <b>{new_billet['heat_number']}</b> | "
                            f"Dim: <b>{dim_hud}</b> | "
                            f"Defects: <b>{', '.join(new_billet['defect_strings']) if new_billet['defect_strings'] else 'None (Clean)'}</b> | "
                            f"📊 <span style='color:#38BDF8;'>Logged to Excel ({st.session_state.total_inspected} Total)</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                annotated_out = os.path.join(temp_video_dir, "annotated_inspection_result.mp4")
                batch_res = video_processor.process_video(
                    video_path=video_path,
                    camera_view="both",
                    log_to_excel=True,
                    save_annotated_path=annotated_out,
                    progress_callback=on_progress
                )
                st.session_state.video_batch_result = batch_res
                st.success(f"✅ Video Inspection Complete! {batch_res['billets_inspected']} billets detected & logged to Excel ({batch_res['pass_count']} Pass / {batch_res['fail_count']} Fail | Yield: {batch_res['yield_rate']}%)")
                render_video_billet_table(batch_table_box, st.session_state.current_video_billets)

            if "video_batch_result" in st.session_state and st.session_state.video_batch_result:
                vb = st.session_state.video_batch_result
                st.markdown("##### 📊 Video Inspection Summary")
                st.write(f"**Frames Processed:** {vb['total_frames_processed']} | **Billets Detected:** {vb['billets_inspected']} | **Yield:** {vb['yield_rate']}%")

                if vb["inspected_records"]:
                    b_rows = []
                    for b in vb["inspected_records"]:
                        dim_label = f"Ø{b.get('diameter_mm', b['width_mm']):.1f} mm ({b['width_mm']:.1f} × {b['height_mm']:.1f})" if b.get("shape") == "cylindrical" else f"{b['width_mm']:.1f} x {b['height_mm']:.1f}"
                        b_rows.append({
                            "Frame": b["frame_number"],
                            "Time (s)": b["timestamp_sec"],
                            "Heat Number": b["heat_number"],
                            "Dim (W x H mm)": dim_label,
                            "Defects": ", ".join(b["defect_strings"]) if b["defect_strings"] else "None",
                            "Status": b["status"]
                        })
                    st.dataframe(pd.DataFrame(b_rows), use_container_width=True)

                if vb.get("annotated_video_path") and os.path.exists(vb["annotated_video_path"]):
                    with open(vb["annotated_video_path"], "rb") as vf:
                        st.download_button(
                            label="📥 Download Annotated Inspection Video (.mp4)",
                            data=vf,
                            file_name="annotated_billet_inspection.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

        elif source_option == "📁 Upload Video or Image":
            st.markdown("---")
            st.info("💡 **Ready to Inspect:** Upload a video file in the sidebar or pick one from the preloaded list (`📹 Videos/circle saamne se dekho.mp4`) to start real-time telemetry and defect scanning.")

        elif source_option == "🏭 USB Industrial Camera":
            st.markdown("---")
            st.markdown("#### 📜 Camera Session Stored Text History")
            session_records = st.session_state.get("camera_session_texts", [])
            if session_records:
                st.caption(f"💾 **{len(session_records)}** unique text segments recorded during this active camera session.")
                df_session = pd.DataFrame(session_records)
                st.dataframe(df_session, use_container_width=True, hide_index=True)

                col_cs1, col_cs2, col_cs3 = st.columns([1.2, 1.2, 1.0])
                with col_cs1:
                    csv_bytes = df_session.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Session (CSV)",
                        data=csv_bytes,
                        file_name="camera_session_ocr_text.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_cs2:
                    out_ex = io.BytesIO()
                    with pd.ExcelWriter(out_ex, engine='openpyxl') as wrt:
                        df_session.to_excel(wrt, sheet_name="CameraSession", index=False)
                    st.download_button(
                        "📊 Download Session (Excel)",
                        data=out_ex.getvalue(),
                        file_name="camera_session_ocr_text.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                with col_cs3:
                    if st.button("🗑️ Clear Session Memory", use_container_width=True):
                        st.session_state.camera_session_texts = []
                        st.rerun()
            else:
                st.info("💡 Any text, brand name (e.g. ASUS), card, or code shown on camera is automatically recorded in this session table.")

            st.write("")
            col_u1, col_u2 = st.columns([1.5, 1.0])
            with col_u1:
                log_usb_frame = st.button("📥 Log Current Billet to Master Excel Sheet", use_container_width=True)
            with col_u2:
                if st.button("🔄 Refresh View", use_container_width=True):
                    st.rerun()

            if log_usb_frame:
                st.session_state.total_inspected += 1
                if overall_status == "PASS":
                    st.session_state.pass_count += 1
                else:
                    st.session_state.fail_count += 1
                billet_logger.log_inspection(
                    heat_number=heat_number,
                    width_mm=width_act,
                    height_mm=height_side_act,
                    defects=defect_strings,
                    status=overall_status,
                    rhomboidity_mm=dim_front.get("rhomboidity_delta_mm", 0.0) if dim_front else 0.0
                )
                st.success(f"Recorded '{heat_number}' as {overall_status} in steel_quality_records.xlsx & SQLite master database!")
                st.rerun()
        else:
            col_nav1, col_nav2, col_nav3 = st.columns([1.2, 1.2, 0.8])
            with col_nav1:
                log_demo_btn = st.button("📥 Log Billet to Excel", type="primary", use_container_width=True)
            with col_nav2:
                next_demo_btn = st.button("⏭ Next Billet & Auto-Log", use_container_width=True)
            with col_nav3:
                st.caption(f"Sample #{st.session_state.sim_index + 1} of {len(front_images)}")

            if log_demo_btn or next_demo_btn:
                st.session_state.total_inspected += 1
                if overall_status == "PASS":
                    st.session_state.pass_count += 1
                else:
                    st.session_state.fail_count += 1
                billet_logger.log_inspection(
                    heat_number=heat_number,
                    width_mm=width_act,
                    height_mm=height_side_act,
                    defects=defect_strings,
                    status=overall_status,
                    rhomboidity_mm=dim_front.get("rhomboidity_delta_mm", 0.0) if dim_front else 0.0
                )
                if next_demo_btn:
                    st.session_state.sim_index += 1
                st.toast(f"✅ Recorded Billet '{heat_number}' ({overall_status}) in Excel!")
                st.rerun()

    with col_details:
        st.subheader("🔍 Quality Telemetry")

        # Status Banner
        if is_image_input:
            if ocr_all_results and ocr_all_results.get("count", 0) > 0:
                st.success(f"### ✅ BILLET TEXT IDENTIFIED ({ocr_all_results['count']} Segments Detected)")
            else:
                st.error("### ❌ NO BILLET TEXT DETECTED")
        elif overall_status == "PASS":
            if source_option == "🏭 USB Industrial Camera" and (ocr_all_results and ocr_all_results["count"] > 0) and not billet_present:
                st.success(f"### ✅ OCR EXTRACTED ({ocr_all_results['count']} Text Segments Detected)")
            else:
                st.success("### ✅ STATUS: PASS (IS 2830:2012 COMPLIANT)")
        elif overall_status == "STANDBY":
            st.info("### ⏸️ STATUS: STANDBY (NO BILLET / TEXT IN FRAME)")
        else:
            st.error("### ❌ STATUS: REJECTED (DEFECT OR OUT OF TOLERANCE)")

        # All-Text OCR Display Card
        if ocr_all_results and ocr_all_results["count"] > 0:
            st.markdown(f"""
            <div style="background-color: #0F172A; border: 1px solid #38BDF8; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase;">
                        📝 All Extracted OCR Text ({ocr_all_results['count']} Found)
                    </span>
                    <span style="background-color: #0369A1; color: #E0F2FE; font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 12px;">
                        Offline Engine Active
                    </span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 8px; word-break: break-word;">
                    {ocr_all_results['all_text']}
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8;">
                    Top Line / Stamped Code: <b style="color: #4ADE80;">{heat_number}</b> (Conf: {ocr_res['confidence']*100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Detailed OCR Line-by-Line Breakdown", expanded=True):
                for idx, it in enumerate(ocr_all_results["detected_items"], 1):
                    col_t1, col_t2 = st.columns([3, 1])
                    with col_t1:
                        st.markdown(f"**{idx}.** `{it['text']}`")
                    with col_t2:
                        conf_pct = it['confidence'] * 100.0
                        c_color = "green" if conf_pct >= 85 else ("orange" if conf_pct >= 60 else "red")
                        st.markdown(f":{c_color}[**{conf_pct:.1f}%**]")
        else:
            # Heat Number Box Fallback
            st.markdown(f"""
            <div style="background-color: #1E293B; border: 1px solid #38BDF8; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Stamped Heat Code (OCR Triple-Voting)</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #38BDF8;">{heat_number}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Confidence: {ocr_res['confidence']*100:.1f}% | Consensus: {ocr_res.get('method', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Camera Session Stored Text Summary Card
        session_recs = st.session_state.get("camera_session_texts", [])
        if session_recs:
            latest_rec = session_recs[-1]
            latest_txt = latest_rec.get("Text") or latest_rec.get("Detected Content", "N/A")
            latest_conf = latest_rec.get("Confidence", "N/A")
            st.markdown(f"""
            <div style="background-color: #0F172A; border: 1px solid #10B981; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: #34D399; text-transform: uppercase;">
                        💾 Session Stored Memory
                    </span>
                    <span style="background-color: #065F46; color: #A7F3D0; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 10px;">
                        {len(session_recs)} Records
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #F1F5F9; margin-top: 5px;">
                    Latest: <b>{latest_txt}</b> ({latest_conf})
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Cross Section Metering Box (with Rhomboidity & Ovality Metrology)
        is_cylindrical = (dim_front.get("shape") == "cylindrical") if dim_front else False
        box_border = '#10B981' if dim_pass else ('#64748B' if overall_status == 'STANDBY' else '#EF4444')

        if is_image_input:
            st.markdown(f"""
            <div style="background-color: #1E293B; border: 1px solid #10B981; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Billet Identification Status</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #38BDF8;">{heat_number}</div>
                <div style="font-size: 0.8rem; color: #34D399; margin-top: 4px;">
                    Optical Stamping / Heat Code Recognized • {ocr_all_results['count'] if ocr_all_results else 0} Text Elements
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif overall_status == "STANDBY":
            st.markdown(f"""
            <div style="background-color: #1E293B; border: 1px solid #64748B; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Conveyor Station Metrology</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #94A3B8;">NO BILLET IN ZONE (STANDBY)</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 3px;">
                    Conveyor active • Optical station waiting for billet to reach inspection table
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif is_cylindrical:
            dia_val = dim_front.get("diameter_mm", width_act)
            ov_val = dim_front.get("ovality_mm", 0.0)
            st.markdown(f"""
            <div style="background-color: #1E293B; border: 1px solid {box_border}; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Cylindrical Billet Dimensions</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">Ø {dia_val:.1f} mm <span style="font-size: 0.85rem; color: #94A3B8;">({width_act:.1f} × {height_front_act:.1f} mm)</span></div>
                <div style="font-size: 0.8rem; color: {'#34D399' if dim_pass else '#F87171'};">
                    Shape: <b>Cylindrical / Round Cast</b> | Dynamic Ovality: <b>{ov_val:.1f} mm</b> ({'PASS' if dim_pass else 'FAIL'})
                </div>
                <div style="font-size: 0.75rem; color: #38BDF8; margin-top: 3px;">
                    Continuous Casting Round Billet Metrology
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            w_dev = width_act - nominal_dim
            h_dev = height_side_act - nominal_dim
            rhomb_delta = dim_front.get("rhomboidity_delta_mm", 0.0) if dim_front else 0.0
            rhomb_pass = dim_front.get("rhomboidity_pass", True) if dim_front else True
            rhomb_color = "#34D399" if rhomb_pass else "#F87171"
            st.markdown(f"""
            <div style="background-color: #1E293B; border: 1px solid {box_border}; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Cross-Section Dimensions (IS 2830)</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #FFFFFF;">{width_act:.1f} × {height_side_act:.1f} mm</div>
                <div style="font-size: 0.8rem; color: {'#34D399' if dim_pass else '#F87171'};">
                    Width: {w_dev:+.1f}mm | Height: {h_dev:+.1f}mm (Limit: ±{tolerance:.1f}mm)
                </div>
                <div style="font-size: 0.78rem; color: {rhomb_color}; margin-top: 5px;">
                    📐 Rhomboidity Skew (ΔD): <b>{rhomb_delta:.1f} mm</b> ({'PASS ≤ 4.0mm' if rhomb_pass else 'FAIL > 4.0mm'})
                </div>
                <div style="font-size: 0.75rem; color: #38BDF8; margin-top: 3px;">
                    {cross_val.get('status', 'Height Cross-Checked')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Industrial SCADA & PLC Interface Status Box
        plc_info = get_current_plc_status()
        plc_kicker_on = plc_info["plc_reject_kicker"]
        plc_badge_bg = "#7F1D1D" if plc_kicker_on else "#065F46"
        plc_badge_fg = "#FCA5A5" if plc_kicker_on else "#6EE7B7"
        plc_action_text = "ACTUATED (Diverting to Rework Table)" if plc_kicker_on else "STANDBY (Line Clear)"
        st.markdown(f"""
        <div style="background-color: #1E293B; border: 1px solid #475569; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Industrial PLC Interface (OPC-UA)</span>
                <span style="background-color: {plc_badge_bg}; color: {plc_badge_fg}; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;">
                    {plc_action_text}
                </span>
            </div>
            <div style="font-size: 0.75rem; color: #CBD5E1; margin-top: 4px;">
                Target: <b>{plc_info['plc_target']}</b>
            </div>
            <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 2px;">
                Dispatch Latency: <b>{plc_info['last_dispatch_latency_ms']:.2f} ms</b> (Guaranteed &lt; 200 ms) | Dual-Store: SQLite Synced
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Defects Box
        st.markdown("##### Detected Surface Defects (Dual Camera)")
        if all_defects:
            for d in all_defects:
                cname = d["class_name"]
                conf = d["confidence"]
                votes = d.get("votes", 1)
                is_override = d.get("smart_safety_override", False)
                if is_override:
                    st.error(f"🚨 **{cname.upper()}** (Conf: {conf*100:.0f}%) — **SMART SAFETY OVERRIDE** (&gt;90% Conf: Urgent Operator Review)")
                else:
                    st.warning(f"⚠️ **{cname.upper()}** (Conf: {conf*100:.0f}%, Ensemble Votes: {votes}/3)")
        else:
            if overall_status == "STANDBY":
                st.caption("Waiting for billet on conveyor...")
            else:
                st.info("✨ No surface defects detected. High-quality billet surface confirmed.")
else:
    with col_details:
        st.subheader("🔍 Quality Telemetry")
        if source_option == "🏭 USB Industrial Camera":
            st.info("### 📸 Camera Ready\n\nPlease capture a frame using the camera viewfinder on the left or select a test card to extract text.")
            st.markdown("""
            **How to test OCR with Webcam / Any Text:**
            1. In the **Optical Station** on the left, grant camera permissions if prompted by your browser.
            2. Hold any text, paper, iron rod, business card, ID badge, or phone in front of the lens.
            3. Click the **'Take Photo'** button.
            4. The system extracts **ALL text strings** detected by OCR, outlines them with bounding boxes on your photo, and displays them right here!
            """)
        else:
            st.info("### ⏸️ Optical Station Standby\n\nPlease select an active video or image source from the sidebar.")

st.divider()

# --- RECENT INSPECTION LOG TABLE ---
recent_df = billet_logger.get_recent_logs(limit=100)
total_logged = len(recent_df)

col_tbl1, col_tbl2, col_tbl3, col_tbl4 = st.columns([2.2, 1.1, 1.1, 0.9])
with col_tbl1:
    st.subheader("📋 Real-Time Billet Inspection Log")
    st.caption(f"📁 **Live Local File:** `{billet_logger.file_path}` *(OneDrive Lock-Free • Real-Time COM Sync)* | **Total Recorded: {total_logged} Billets**")
with col_tbl2:
    if st.button("🟢 Open Live Excel", key="open_excel_tbl_btn", use_container_width=True):
        ok, target = billet_logger.launch_live_excel_window()
        if ok:
            st.toast(f"Opening {os.path.basename(target)} in Microsoft Excel...")
        else:
            st.error(f"Could not open: {target}")
with col_tbl3:
    t_bytes = get_excel_export_bytes()
    st.download_button(
        label="📥 Download Excel",
        data=t_bytes,
        file_name="steel_quality_records.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
with col_tbl4:
    if st.button("🗑️ Clear Log", key="clear_log_tbl_btn", use_container_width=True):
        billet_logger.clear_log()
        st.session_state.total_inspected = 0
        st.session_state.pass_count = 0
        st.session_state.fail_count = 0
        st.session_state.defect_counts = {}
        st.rerun()

render_log_table(st, limit=15)
