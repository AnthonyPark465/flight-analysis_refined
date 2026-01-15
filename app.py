import os
import base64
import datetime
from pathlib import Path
import shutil
import time

import streamlit as st
# Vercel/Streamlit Cloud 등 클라우드 환경에서 에러 방지를 위해 matplotlib 백엔드 설정
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
import plotly.graph_objects as go
from supabase import create_client

# --- 기본 설정 ---
BASE_DIR = Path(__file__).resolve().parent

# 로컬 저장소 설정 (분석 중 임시 파일 생성용)
def _pick_persist_dir() -> Path:
    # Vercel 환경
    if os.environ.get("VERCEL"):
        p = Path("/tmp/res")
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    # Streamlit Cloud 등 기타 환경
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
BUCKET_NAME = "flight-results"  # 생성하신 버킷 이름

LOGO_PATH = BASE_DIR / "flightdata-logo.svg"
MISSION_PATH = BASE_DIR / "mission.png"
WEIGHTS_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "250921_best.pt")))

PAGES = ["Home", "Analyze", "History"]

# --- Supabase 헬퍼 ---
def _get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets.get(name, default))
    except Exception:
        pass
    return str(os.getenv(name, default))

@st.cache_resource
def _make_supabase(url: str, key: str):
    return create_client(url, key)

def _get_supabase():
    url = _get_secret("SUPABASE_URL").strip()
    key = _get_secret("SUPABASE_SERVICE_ROLE_KEY").strip()
    if not url or not key:
        return None
    try:
        return _make_supabase(url, key)
    except Exception:
        return None

# [NEW] 파일 업로드 함수
def upload_to_supabase(local_folder: Path, folder_name: str):
    sb = _get_supabase()
    if sb is None:
        st.error("Supabase connection failed.")
        return False
    
    try:
        # 1. MP4 업로드
        video_path = local_folder / "input.mp4"
        if video_path.exists():
            with open(video_path, "rb") as f:
                sb.storage.from_(BUCKET_NAME).upload(
                    path=f"{folder_name}/input.mp4",
                    file=f,
                    file_options={"content-type": "video/mp4", "x-upsert": "true"}
                )

        # 2. HTML 업로드
        html_path = local_folder / "trajectory_plot.html"
        if html_path.exists():
            with open(html_path, "rb") as f:
                sb.storage.from_(BUCKET_NAME).upload(
                    path=f"{folder_name}/trajectory_plot.html",
                    file=f,
                    file_options={"content-type": "text/html", "x-upsert": "true"}
                )
        return True
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return False

# [NEW] 클라우드에서 결과 가져오기 함수
def get_cloud_results(folder_name: str):
    sb = _get_supabase()
    if sb is None:
        return None, None
    
    # Video는 Public URL을 바로 사용
    video_url = sb.storage.from_(BUCKET_NAME).get_public_url(f"{folder_name}/input.mp4")
    
    # HTML은 내용을 다운로드해서 읽음 (iframe 이슈 방지 및 깔끔한 렌더링)
    html_content = None
    try:
        data = sb.storage.from_(BUCKET_NAME).download(f"{folder_name}/trajectory_plot.html")
        html_content = data.decode("utf-8")
    except Exception:
        html_content = None
        
    return video_url, html_content

# DB 히스토리 관련 함수
def load_history():
    sb = _get_supabase()
    if sb is None:
        return []
    try:
        resp = (
            sb.table("history")
            .select("folder_name,analysis_name,created_at,points")
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        )
        data = resp.data if hasattr(resp, "data") else []
        if not isinstance(data, list):
            return []
        rows = []
        for it in data:
            if not it.get("analysis_name"):
                continue
            rows.append({
                "folder_name": it.get("folder_name", ""),
                "analysis_name": it.get("analysis_name", ""),
                "created_at": str(it.get("created_at", "")),
                "points": int(it.get("points", 0) or 0),
            })
        return rows
    except Exception:
        return []

def add_history(record: dict):
    sb = _get_supabase()
    if sb is None:
        st.toast("Supabase not configured (History skipped)", icon="⚠️")
        return
    try:
        created_at = record.get("created_at")
        if not created_at:
            created_at = datetime.datetime.utcnow()
        if isinstance(created_at, str):
            try:
                created_at = datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            except Exception:
                created_at = datetime.datetime.utcnow()
        
        created_at_iso = created_at.replace(microsecond=0).isoformat() + "Z"
        payload = {
            "folder_name": str(record.get("folder_name", "")),
            "analysis_name": str(record.get("analysis_name", "")),
            "created_at": created_at_iso,
            "points": int(record.get("points", 0) or 0),
        }
        sb.table("history").insert(payload).execute()
        st.toast("Saved to DB & Cloud", icon="")
    except Exception as e:
        print(f"History insert failed: {e}")

# --- UI 유틸리티 ---
def show_analysis_results(video_source, html_content):
    """
    결과를 보여주는 공통 함수 (로컬 파일 경로 or URL 모두 처리 가능하도록 수정)
    video_source: 파일 경로(Path) 또는 URL(str)
    html_content: HTML 문자열
    """
    col_left, col_center, col_right = st.columns([0.15, 0.7, 0.15])
    
    with col_center:
        st.divider()
        st.markdown("### Video")
        
        # 입력값이 문자열(URL)인지 경로(Path)인지 확인하여 처리
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


# --- UI 스타일링 ---
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

# --- Pages ---

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
        st.caption("Runs are automatically uploaded to Supabase so you never lose data.")
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

        # 1. 로컬 임시 폴더 생성
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

        # 궤적 추출
        trajectory_points = []
        for frame_result in results:
            if frame_result.boxes is None: continue
            boxes = frame_result.boxes.xyxy
            if boxes is None: continue
            arr = boxes.cpu().numpy()
            for box in arr:
                x1, y1, x2, y2 = box[:4]
                trajectory_points.append(((x1 + x2) / 2, (y1 + y2) / 2))

        # 그래프 생성 및 저장
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

        # 2. Supabase Storage로 업로드
        with st.spinner("Uploading results to Cloud Storage..."):
            upload_success = upload_to_supabase(output_folder, folder_name)
        
        # 3. DB 기록 저장
        if upload_success:
            add_history({
                "folder_name": folder_name,
                "analysis_name": analysis_name,
                "created_at": datetime.datetime.now(),
                "points": pt_count,
            })
            st.success("Analysis & Upload Complete!")
        else:
            st.error("Analysis done, but Cloud Upload failed.")

        # 4. 결과 보기 (방금 만든 로컬 파일 사용해서 즉시 표시)
        with result_container:
            show_analysis_results(str(temp_video), html_content)


def history_page():
    st.title("History")
    
    items = load_history()
    if not items:
        st.info("No history found.")
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
        f"{it['analysis_name']} ({it['created_at'][:10]}) - {it['points']} pts": it 
        for it in filtered_items
    }
    
    selected_label = st.selectbox("Select a launch", list(options.keys()), label_visibility="collapsed")
    selected_data = options[selected_label]
    folder_name = selected_data["folder_name"]
    
    # [수정됨] 로컬 파일 대신 Supabase Cloud에서 URL 및 HTML 데이터 가져오기
    with st.spinner("Loading from Cloud..."):
        vid_url, html_data = get_cloud_results(folder_name)
    
    if vid_url:
        show_analysis_results(vid_url, html_data)
    else:
        st.error("Could not retrieve files from Cloud Storage.")

# --- Main Execution ---
apply_ui()
topbar()

if st.session_state["page"] == "Home":
    home_page()
elif st.session_state["page"] == "Analyze":
    analyze_page()
elif st.session_state["page"] == "History":
    history_page()