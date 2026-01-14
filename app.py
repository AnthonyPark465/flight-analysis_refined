import os
import base64
import datetime
from pathlib import Path
import shutil

import streamlit as st
# Vercel 등 클라우드 환경에서 에러 방지를 위해 matplotlib 백엔드 설정
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
import plotly.graph_objects as go
from supabase import create_client

# --- 기본 설정 ---
BASE_DIR = Path(__file__).resolve().parent

# 로컬 저장소 설정 (Vercel에서는 임시 폴더인 /tmp 사용 권장)
def _pick_persist_dir() -> Path:
    # Vercel 환경인지 확인
    if os.environ.get("VERCEL"):
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
# 모델 경로: 환경변수 혹은 기본 파일
WEIGHTS_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "250921_best.pt")))

# 페이지 정의 (Results 통합됨)
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
            # 포인트가 없거나 데이터가 불완전한 행은 스킵 (빈 리스트 제거 요청 반영)
            if not it.get("analysis_name") or not it.get("points"):
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
        st.toast("Saved to history ✅", icon="✅")
    except Exception as e:
        print(f"History insert failed: {e}")

# --- 유틸리티 ---
def pick_video_file(folder: Path):
    if not folder.exists():
        return None
    mp4s = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        return None
    non_input = [p for p in mp4s if p.name.lower() not in ("input.mp4", "original.mp4")]
    return non_input[0] if non_input else mp4s[0]

def load_saved_plot_html(folder: Path):
    html_path = folder / "trajectory_plot.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return None

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
        /* 기본 헤더 숨김 */
        [data-testid="stHeader"] { display: none !important; }
        
        /* 폰트 설정 */
        html, body, [class*="css"] {
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        
        /* --- 네비게이션(Radio) 스타일링 --- */
        
        /* 1. 라디오 그룹 컨테이너 */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: flex;
            justify-content: flex-end;
            gap: 24px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 10px;
            background: transparent;
        }

        /* 2. 동그란 라디오 버튼(원) 숨기기 */
        div[data-testid="stRadio"] label > div:first-child {
            display: none !important;
        }

        /* 3. 텍스트(p태그) 스타일 - 여기가 핵심입니다 */
        /* label 바로 아래가 아니라, 내부 텍스트 요소(p)를 직접 타겟팅합니다 */
        div[data-testid="stRadio"] label p {
            font-size: 1rem !important;
            font-weight: 500 !important;
            color: #94a3b8 !important; /* 기본: 연한 회색 */
            
            /* 여기에 transition을 걸어야 글자 색이 부드럽게 바뀝니다 */
            transition: color 0.25s ease-in-out !important; 
        }

        /* 4. 마우스 호버 효과 */
        div[data-testid="stRadio"] label:hover p {
            color: #334155 !important; /* 마우스 올리면: 중간 회색 */
        }

        /* 5. 선택된 항목 스타일 (Active) */
        div[data-testid="stRadio"] label:has(input:checked) p {
            color: #0f172a !important; /* 선택됨: 진한 검정 */
            font-weight: 700 !important;
        }

        /* 기타 스타일 */
        button[title="View fullscreen"] { display: none !important; }
        [data-testid="stImage"] button { display: none !important; }
        [data-testid="stFileUploader"] { margin-top: 0px; }
        .topbar-divider {
            height: 1px;
            background: #e2e8f0;
            margin: 0 0 24px 0;
        }
        .big-label {
            font-size: 1.2rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        </style>
    """, unsafe_allow_html=True)

def render_logo_svg(width_px: int = 120):
    # 로고 파일이 없으면 텍스트만 표시
    if not LOGO_PATH.exists():
        st.markdown("### FlightData")
        return

    # 이미지를 Base64로 변환
    b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    
    # [수정됨] 버튼 해킹(hidden button) 코드를 제거하고 순수 이미지만 출력합니다.
    # 클릭 시 이동 기능은 우측 네비게이션 메뉴를 사용하세요.
    st.markdown(
        f"""
        <div style="display:flex; align-items:center;">
          <img src="data:image/svg+xml;base64,{b64}" style="width:{width_px}px; height:auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

def topbar():
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    
    # URL 쿼리 파라미터나 세션 상태 안전장치
    if st.session_state["page"] not in PAGES:
        st.session_state["page"] = "Home"

    c1, c2 = st.columns([0.2, 0.8], vertical_alignment="bottom")

    with c1:
        # 로고 표시
        render_logo_svg(width_px=140)

    with c2:
        current_page = st.session_state["page"]
        
        # 화면에 보여줄 텍스트 (스크린샷처럼 About으로 변경)
        # PAGES = ["Home", "Analyze", "History"] 라고 가정
        display_map = {
            "Home": "About",
            "Analyze": "Analyze",
            "History": "History"
        }
        
        # 현재 페이지의 인덱스 찾기
        try:
            idx = PAGES.index(current_page)
        except ValueError:
            idx = 0
            
        # 선택용 리스트 생성 (About, Analyze, History)
        display_options = [display_map[p] for p in PAGES]

        selected_display = st.radio(
            "nav_radio",
            display_options,
            index=idx,
            horizontal=True,
            label_visibility="collapsed",
            key="nav_radio_key"
        )
        
        # 선택된 라벨("About")을 다시 내부 페이지명("Home")으로 변환
        reverse_map = {v: k for k, v in display_map.items()}
        selected_page = reverse_map[selected_display]

        if selected_page != current_page:
            st.session_state["page"] = selected_page
            st.rerun()

    # 구분선 (이미 스타일에서 border-bottom을 줬으므로 여기서는 여백만 조정하거나 제거 가능)
    # st.markdown("<div class='topbar-divider'></div>", unsafe_allow_html=True)

# --- Pages ---

def home_page():
    # 아이콘 변경 요청: AI 느낌 제거 -> 직관적인 이모지 사용
    
    left, right = st.columns([1.2, 0.8], gap="large", vertical_alignment="center")

    with left:
        st.title("Flight Analysis")
        st.markdown(
            """
            <div style='color:#64748b; font-size:1.1rem; margin-bottom:20px;'>
            Upload a launch video and turn it into clean trajectory + performance signals, instantly.
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # 태그들
        st.markdown(
            """
            <div style="display:flex; gap:10px; margin-bottom:20px;">
              <span style="background:#f1f5f9; padding:6px 12px; border-radius:20px; font-size:0.9rem; color:#334155;">📈 Trajectory</span>
              <span style="background:#f1f5f9; padding:6px 12px; border-radius:20px; font-size:0.9rem; color:#334155;">📏 Distance</span>
              <span style="background:#f1f5f9; padding:6px 12px; border-radius:20px; font-size:0.9rem; color:#334155;">🧭 Angle & Speed</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        if MISSION_PATH.exists():
            st.image(str(MISSION_PATH), use_column_width=True)

    st.divider()

    # 아이콘/텍스트 변경
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.subheader("Quick Start")
        st.caption("Drop a video, name the run, and generate a trajectory in one flow.")
    with c2:
        st.subheader("Precise Tracking") 
        st.caption("Detection-based tracking to collect points and visualize the path accurately.")
    with c3:
        st.subheader("Auto-Save") 
        st.caption("Every launch is stored securely. Re-open past runs and compare quickly.")

    st.divider()
    st.markdown("Feedback welcome: **palkiayp@gmail.com**")

@st.cache_resource
def get_model():
    # 모델 파일 없으면 에러 방지
    if not WEIGHTS_PATH.exists():
        return None
    return YOLO(str(WEIGHTS_PATH))

def analyze_page():
    st.title("Analyze")
    
    # 텍스트 축약 및 정리
    st.markdown(
        """
        <div style="background:#f8fafc; padding:15px; border-radius:8px; border:1px solid #e2e8f0; margin-bottom:25px; color:#475569;">
        Please upload an MP4 or MOV file, enter a name, and press the Start button to begin analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1, 1], gap="large")
    
    with c1:
        st.markdown('<div class="big-label">1. Video file</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov"], label_visibility="collapsed")

    with c2:
        st.markdown('<div class="big-label">2. Analysis name</div>', unsafe_allow_html=True)
        analysis_name = st.text_input("Name", placeholder="e.g. Test_Flight_01", label_visibility="collapsed")
        
        st.write("") # Spacer
        # Start 버튼 추가
        start_btn = st.button("Start Analysis", type="primary", use_container_width=True)

    # 로직: Start 버튼을 눌러야 실행
    if start_btn:
        if not uploaded_file:
            st.warning("Please upload a video file first.")
            return
        if not analysis_name:
            st.warning("Please enter an analysis name.")
            return

        RES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장 및 폴더 생성
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
            st.error(f"Model weights not found at {WEIGHTS_PATH}. Please check configuration.")
            return

        with st.spinner("Running detection & tracking..."):
            try:
                # YOLO Inference
                results = model(
                    str(temp_video),
                    save=True,
                    show=False,
                    project=str(RES_DIR),
                    name=folder_name,
                    exist_ok=True # 폴더 이미 생성했으므로
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        # 결과 비디오 찾기
        # Ultralytics는 project/name 폴더 안에 결과를 저장함. 
        # 위에서 output_folder를 미리 만들었지만, model.predict가 내부에 또 생성할 수 있으므로 경로 확인 필요
        # 보통 project/name/input.mp4 (avi) 등으로 저장됨.
        
        # 궤적 추출
        trajectory_points = []
        for frame_result in results:
            if frame_result.boxes is None:
                continue
            boxes = frame_result.boxes.xyxy
            if boxes is None:
                continue
            arr = boxes.cpu().numpy()
            for box in arr:
                x1, y1, x2, y2 = box[:4]
                trajectory_points.append(((x1 + x2) / 2, (y1 + y2) / 2))

        # 데이터 저장
        pt_count = len(trajectory_points)
        add_history({
            "folder_name": folder_name,
            "analysis_name": analysis_name,
            "created_at": datetime.datetime.now(),
            "points": pt_count,
        })
        
        # 그래프 생성
        if pt_count > 1:
            xs, ys = zip(*trajectory_points)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Path", 
                                     line=dict(color='#0f172a', width=2),
                                     marker=dict(size=4)))
            fig.update_yaxes(autorange="reversed")
            
            # 그래프 여백 및 ModeBar 제거, 레이블 겹침 방지
            fig.update_layout(
                margin=dict(l=60, r=20, t=40, b=40),
                title="Trajectory",
                xaxis_title="X (px)",
                yaxis_title="Y (px)",
                height=500,
                hovermode="closest",
                dragmode=False # 줌/팬 비활성화 (요청사항: 메뉴 삭제 효과)
            )
            
            html_path = output_folder / "trajectory_plot.html"
            fig.write_html(str(html_path), config={'displayModeBar': False})
            
        st.success("Analysis Complete! Go to History menu to view details if not shown below.")
        
        # 바로 결과 보여주기 (세션 상태를 이용해 리로드 없이 보여주거나, History 페이지로 유도)
        st.session_state["page"] = "History"
        st.rerun()


def history_page():
    # History & Result 통합
    st.title("History")
    
    items = load_history()
    if not items:
        st.info("No history found. Run an analysis first.")
        return

    # 검색 및 필터
    search_q = st.text_input("Search", placeholder="Type analysis name...", label_visibility="collapsed")
    
    filtered_items = items
    if search_q:
        q_lower = search_q.lower()
        filtered_items = [it for it in items if q_lower in it["analysis_name"].lower()]
    
    if not filtered_items:
        st.warning("No matching records.")
        return

    # 드롭다운 라벨 생성
    options = {
        f"{it['analysis_name']} ({it['created_at'][:10]}) - {it['points']} pts": it 
        for it in filtered_items
    }
    
    selected_label = st.selectbox("Select a launch", list(options.keys()), label_visibility="collapsed")
    selected_data = options[selected_label]
    
    folder_name = selected_data["folder_name"]
    target_dir = RES_DIR / folder_name
    
    st.divider()
    
    # 결과 화면: 좌우 배치 (비율 조정으로 영상/그래프 작게)
    c_vid, c_plot = st.columns([1, 1], gap="medium")
    
    with c_vid:
        st.markdown("### Video")
        vid_file = pick_video_file(target_dir)
        
        # Vercel 등에서는 파일시스템이 초기화되므로 파일이 없을 수 있음
        if vid_file and vid_file.exists():
            st.video(str(vid_file))
        else:
            st.error("Video file not found on server (Files are temporary on this demo).")
            st.caption(f"Looking in: {target_dir}")

    with c_plot:
        st.markdown("### Trajectory")
        html_content = load_saved_plot_html(target_dir)
        
        if html_content:
            st.components.v1.html(html_content, height=520, scrolling=False)
        else:
            # 파일이 없으면 포인트 데이터라도 있으면 다시 그릴 수 있으나, 
            # 현재 구조상 파일시스템 의존적이므로 메시지 출력
            st.warning("Trajectory plot not found.")

    # Vercel 환경 안내
    if os.environ.get("VERCEL"):
        st.info("Note: On Vercel (Serverless), analyzed files are deleted after the session ends. Only the database record persists.")

# --- Main Execution ---
apply_ui()
topbar()

if st.session_state["page"] == "Home":
    home_page()
elif st.session_state["page"] == "Analyze":
    analyze_page()
elif st.session_state["page"] == "History":
    history_page()