import os
import base64
import datetime
import json
from pathlib import Path
import shutil
import time

import streamlit as st
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
import plotly.graph_objects as go

import firebase_admin
from firebase_admin import credentials, firestore, storage

def apply_ui():
    st.set_page_config(
        page_title="Flight Analysis",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    st.markdown("""
        <style>
        [data-testid="stHeader"] { display: none !important; }
        html, body, [class*="css"] {
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        [data-testid="InputInstructions"] { display: none !important; }
        [data-testid="stHeaderActionElements"] { display: none !important; }
        .stMarkdown a.anchor-link { display: none !important; }
        
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: flex;
            justify-content: flex-end;
            gap: 24px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 10px;
            background: transparent;
        }
        div[data-testid="stRadio"] label > div:first-child { display: none !important; }
        div[data-testid="stRadio"] label p {
            font-size: 1rem !important;
            font-weight: 500 !important;
            color: #94a3b8 !important;
            transition: color 0.25s ease-in-out !important; 
        }
        div[data-testid="stRadio"] label:hover p { color: #334155 !important; }
        div[data-testid="stRadio"] label:has(input:checked) p {
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        button[title="View fullscreen"] { display: none !important; }
        [data-testid="stImage"] button { display: none !important; }
        [data-testid="stFileUploader"] { margin-top: 0px; }
        .big-label {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        </style>
    """, unsafe_allow_html=True)

apply_ui()

BASE_DIR = Path(__file__).resolve().parent

def _pick_persist_dir() -> Path:
    if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("VERCEL"):
        p = Path("/tmp/res")
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    candidates = []
    env_dir = os.getenv("PERSIST_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(BASE_DIR / "res")
    
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            continue
    return BASE_DIR / "res"

RES_DIR = _pick_persist_dir()

LOGO_PATH = BASE_DIR / "flightdata-logo.svg"
MISSION_PATH = BASE_DIR / "mission.png"
WEIGHTS_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "250921_best.pt")))

PAGES = ["Home", "Analyze", "History"]

@st.cache_resource
def init_firebase():
    try:
        if firebase_admin._apps:
            return firebase_admin.get_app()

        cred_json = os.environ.get("FIREBASE_CREDENTIALS")
        if not cred_json and "FIREBASE_CREDENTIALS" in st.secrets:
            cred_json = st.secrets["FIREBASE_CREDENTIALS"]

        if not cred_json:
            st.error("Firebase credentials not found.")
            return None

        if isinstance(cred_json, str):
            try:
                cred_dict = json.loads(cred_json)
            except json.JSONDecodeError:
                st.error("Invalid JSON format in FIREBASE_CREDENTIALS.")
                return None
        else:
            cred_dict = cred_json

        bucket_name = os.environ.get("FIREBASE_BUCKET_NAME")
        if not bucket_name and "FIREBASE_BUCKET_NAME" in st.secrets:
            bucket_name = st.secrets["FIREBASE_BUCKET_NAME"]
        
        if not bucket_name:
            bucket_name = f"{cred_dict.get('project_id')}.appspot.com"

        cred = credentials.Certificate(cred_dict)
        return firebase_admin.initialize_app(cred, {
            'storageBucket': bucket_name
        })
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        return None

init_firebase()

def get_firestore_db():
    if not firebase_admin._apps:
        if init_firebase() is None:
            return None
    return firestore.client()

def get_storage_bucket():
    if not firebase_admin._apps:
        if init_firebase() is None:
            return None
    return storage.bucket()

def upload_to_firebase(local_folder: Path, folder_name: str):
    bucket = get_storage_bucket()
    if bucket is None:
        st.error("Firebase Storage connection failed.")
        return False
    
    try:
        video_path = local_folder / "input.mp4"
        if video_path.exists():
            blob_vid = bucket.blob(f"{folder_name}/input.mp4")
            blob_vid.upload_from_filename(str(video_path), content_type="video/mp4")

        html_path = local_folder / "trajectory_plot.html"
        if html_path.exists():
            blob_html = bucket.blob(f"{folder_name}/trajectory_plot.html")
            blob_html.upload_from_filename(str(html_path), content_type="text/html")
            
        return True
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return False

def get_cloud_results(folder_name: str):
    bucket = get_storage_bucket()
    if bucket is None:
        return None, None
    
    video_url = None
    html_content = None

    try:
        blob_vid = bucket.blob(f"{folder_name}/input.mp4")
        if blob_vid.exists():
            video_url = blob_vid.generate_signed_url(expiration=datetime.timedelta(hours=1))

        blob_html = bucket.blob(f"{folder_name}/trajectory_plot.html")
        if blob_html.exists():
            html_bytes = blob_html.download_as_bytes()
            html_content = html_bytes.decode("utf-8")

    except Exception as e:
        st.warning(f"Error retrieving files: {e}")
        
    return video_url, html_content

def load_history():
    db = get_firestore_db()
    if db is None:
        return []
    try:
        docs = (
            db.collection("history")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(500)
            .stream()
        )
        
        rows = []
        for doc in docs:
            data = doc.to_dict()
            if not data.get("analysis_name"):
                continue
            
            c_at = data.get("created_at")
            if isinstance(c_at, datetime.datetime):
                c_at_str = c_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                c_at_str = str(c_at)

            rows.append({
                "folder_name": data.get("folder_name", ""),
                "analysis_name": data.get("analysis_name", ""),
                "created_at": c_at_str,
                "points": int(data.get("points", 0) or 0),
            })
        return rows
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return []

def add_history(record: dict):
    db = get_firestore_db()
    if db is None:
        st.toast("Firebase DB not configured", icon="⚠️")
        return
    try:
        created_at = record.get("created_at")
        if not created_at:
            created_at = datetime.datetime.utcnow()
        
        payload = {
            "folder_name": str(record.get("folder_name", "")),
            "analysis_name": str(record.get("analysis_name", "")),
            "created_at": created_at, 
            "points": int(record.get("points", 0) or 0),
        }
        
        db.collection("history").add(payload)
        st.toast("Saved to Firestore & Storage", icon="☁️")
    except Exception as e:
        print(f"History insert failed: {e}")
        st.error(f"History save error: {e}")

def show_analysis_results(video_source, html_content):
    col_left, col_center, col_right = st.columns([0.15, 0.7, 0.15])
    
    with col_center:
        st.divider()
        st.markdown("### Video")
        
        if isinstance(video_source, (str, Path)):
            st.video(str(video_source))
        else:
            st.warning("Video source unavailable.")

        st.markdown("### Trajectory")
        if html_content:
            st.components.v1.html(html_content, height=550, scrolling=False)
        else:
            st.info("No trajectory plot available.")
            
        st.divider()

def render_logo_svg(width_px: int = 120):
    if not LOGO_PATH.exists():
        st.markdown("### FlightData")
        return
    b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    st.markdown(
        f"""<div style="display:flex; align-items:center;"><img src="data:image/svg+xml;base64,{b64}" style="width:{width_px}px; height:auto;" /></div>""",
        unsafe_allow_html=True,
    )

def topbar():
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    if st.session_state["page"] not in PAGES:
        st.session_state["page"] = "Home"

    c1, c2 = st.columns([0.2, 0.8], vertical_alignment="bottom")
    with c1:
        render_logo_svg(width_px=140)
    with c2:
        current_page = st.session_state["page"]
        display_map = {"Home": "About", "Analyze": "Analyze", "History": "History"}
        try:
            idx = PAGES.index(current_page)
        except ValueError:
            idx = 0
        display_options = [display_map[p] for p in PAGES]
        selected_display = st.radio("nav_radio", display_options, index=idx, horizontal=True, label_visibility="collapsed", key="nav_radio_key")
        reverse_map = {v: k for k, v in display_map.items()}
        selected_page = reverse_map[selected_display]
        if selected_page != current_page:
            st.session_state["page"] = selected_page
            st.rerun()

def home_page():
    left, right = st.columns([1.2, 0.8], gap="large", vertical_alignment="center")
    with left:
        st.title("Flight Analysis")
        st.markdown("""<div style='color:#64748b; font-size:1.1rem; margin-bottom:20px;'>Upload a launch video and turn it into clean trajectory + performance signals, instantly.</div>""", unsafe_allow_html=True)
        st.markdown("""<div style="display:flex; gap:10px; margin-bottom:20px;">
              <span style="background:#f1f5f9; padding:6px 12px; border-radius:20px; font-size:0.9rem; color:#334155;">📈 Trajectory</span>
              <span style="background:#f1f5f9; padding:6px 12px; border-radius:20px; font-size:0.9rem; color:#334155;">📏 Distance</span>
              <span style="background:#f1f5f9; padding:6px 12px; border-radius:20px; font-size:0.9rem; color:#334155;">🧭 Angle & Speed</span>
            </div>""", unsafe_allow_html=True)
    with right:
        if MISSION_PATH.exists():
            st.image(str(MISSION_PATH), use_column_width=True)
    st.divider()
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.subheader("Quick Start")
        st.caption("Drop a video, name the run, and generate a trajectory in one flow.")
    with c2:
        st.subheader("Precise Tracking") 
        st.caption("Detection-based tracking to collect points and visualize the path accurately.")
    with c3:
        st.subheader("Cloud Storage") 
        st.caption("Runs are automatically uploaded to Firebase so you never lose data.")
    st.divider()
    st.markdown("Feedback welcome: **palkiayp@gmail.com**")

@st.cache_resource
def get_model():
    if not WEIGHTS_PATH.exists():
        return None
    return YOLO(str(WEIGHTS_PATH))

def analyze_page():
    st.title("Analyze")
    st.markdown("""<div style="background:#f8fafc; padding:15px; border-radius:8px; border:1px solid #e2e8f0; margin-bottom:25px; color:#475569;">Please upload an MP4 or MOV file, enter a name, and press the Start button.</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown('<div class="big-label">Video file</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov"], label_visibility="collapsed")
    with c2:
        st.markdown('<div class="big-label">Analysis name</div>', unsafe_allow_html=True)
        analysis_name = st.text_input("Name", placeholder="e.g. Test_Flight_01", label_visibility="collapsed")
        st.write("") 
        start_btn = st.button("Start Analysis", type="primary", use_container_width=True)

    result_container = st.container()

    if start_btn:
        if not uploaded_file:
            st.warning("Please upload a video file first.")
            return
        if not analysis_name:
            st.warning("Please enter an analysis name.")
            return

        RES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in analysis_name if c.isalnum() or c in ("-", "_")).strip()
        folder_name = f"{timestamp}_{safe_name}" if safe_name else f"{timestamp}_analysis"
        output_folder = RES_DIR / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        temp_video = output_folder / "input.mp4"
        temp_video.write_bytes(uploaded_file.read())

        st.info(f"Processing '{analysis_name}'... Please wait.")
        
        model = get_model()
        if model is None:
            st.error(f"Model weights not found at {WEIGHTS_PATH}.")
            return

        with st.spinner("Running detection & tracking..."):
            try:
                results = model(str(temp_video), save=True, show=False, project=str(RES_DIR), name=folder_name, exist_ok=True)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        trajectory_points = []
        for frame_result in results:
            if frame_result.boxes is None: continue
            boxes = frame_result.boxes.xyxy
            if boxes is None: continue
            arr = boxes.cpu().numpy()
            for box in arr:
                x1, y1, x2, y2 = box[:4]
                trajectory_points.append(((x1 + x2) / 2, (y1 + y2) / 2))

        pt_count = len(trajectory_points)
        html_content = None
        if pt_count > 1:
            xs, ys = zip(*trajectory_points)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Path", line=dict(color='#0f172a', width=2), marker=dict(size=4)))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                margin=dict(l=60, r=20, t=40, b=40), title="Trajectory",
                xaxis_title="X (px)", yaxis_title="Y (px)", height=500, hovermode="closest", dragmode=False 
            )
            html_path = output_folder / "trajectory_plot.html"
            fig.write_html(str(html_path), config={'displayModeBar': False})
            html_content = html_path.read_text(encoding="utf-8")

        upload_success = False
        with st.spinner("Uploading results to Cloud Storage..."):
            upload_success = upload_to_firebase(output_folder, folder_name)
        
        if upload_success:
            add_history({
                "folder_name": folder_name,
                "analysis_name": analysis_name,
                "created_at": datetime.datetime.now(),
                "points": pt_count,
            })
            st.success("Analysis & Upload Complete!")
            
            st.session_state["current_analysis"] = {
                "video_path": str(temp_video),
                "html_content": html_content,
                "timestamp": datetime.datetime.now(),
                "name": analysis_name
            }
            st.rerun()
            
        else:
            st.error("Analysis done, but Cloud Upload failed. Check Firebase config.")
            st.session_state["current_analysis"] = {
                "video_path": str(temp_video),
                "html_content": html_content,
                "timestamp": datetime.datetime.now(),
                "name": analysis_name
            }
            st.rerun()

    if "current_analysis" in st.session_state and st.session_state["current_analysis"]:
        ca = st.session_state["current_analysis"]
        elapsed = datetime.datetime.now() - ca["timestamp"]
        if elapsed.total_seconds() < 86400: # 24 hours
            with result_container:
                st.markdown(f"### Recent Result: {ca['name']}")
                st.caption(f"Analyzed at: {ca['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                show_analysis_results(ca["video_path"], ca["html_content"])
                
                if st.button("Clear Result", help="Clear this result from the view"):
                    del st.session_state["current_analysis"]
                    st.rerun()
        else:
            del st.session_state["current_analysis"]
            st.rerun()


def history_page():
    st.title("History")
    
    items = load_history()
    if not items:
        st.info("No history found in Firestore.")
        return

    search_q = st.text_input("Search", placeholder="Type analysis name...", label_visibility="collapsed")
    
    filtered_items = items
    if search_q:
        q_lower = search_q.lower()
        filtered_items = [it for it in items if q_lower in it["analysis_name"].lower()]
    
    if not filtered_items:
        st.warning("No matching records.")
        return

    options = {
        f"{it['analysis_name']} ({it['created_at'][:19]}) - {it['points']} pts": it 
        for it in filtered_items
    }
    
    selected_label = st.selectbox("Select a launch", list(options.keys()), label_visibility="collapsed")
    selected_data = options[selected_label]
    folder_name = selected_data["folder_name"]
    
    with st.spinner("Loading from Cloud..."):
        vid_url, html_data = get_cloud_results(folder_name)
    
    if vid_url:
        show_analysis_results(vid_url, html_data)
    else:
        st.error("Could not retrieve files from Cloud Storage.")

topbar()

if st.session_state["page"] == "Home":
    home_page()
elif st.session_state["page"] == "Analyze":
    analyze_page()
elif st.session_state["page"] == "History":
    history_page()