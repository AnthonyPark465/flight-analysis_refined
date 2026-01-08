import os
import base64
import datetime
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
import plotly.graph_objects as go

from supabase import create_client


BASE_DIR = Path(__file__).resolve().parent


def _pick_persist_dir() -> Path:
    candidates = []
    env_dir = os.getenv("PERSIST_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path("/data"))
    candidates.append(BASE_DIR / "res")

    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return p
        except Exception:
            continue
    return BASE_DIR / "res"


RES_DIR = _pick_persist_dir()

LOGO_PATH = BASE_DIR / "flightdata-logo.svg"
MISSION_PATH = BASE_DIR / "mission.png"
WEIGHTS_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "250921_best.pt")))

PAGES = ["About", "Analyze", "History", "Results"]


def _get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets.get(name, default))
    except Exception:
        pass
    return str(os.getenv(name, default))


@st.cache_resource
def _get_supabase():
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def init_db():
    pass


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
            rows.append(
                {
                    "folder_name": it.get("folder_name", ""),
                    "analysis_name": it.get("analysis_name", ""),
                    "created_at": str(it.get("created_at", "")),
                    "points": int(it.get("points", 0) or 0),
                }
            )
        return rows
    except Exception:
        return []


def add_history(record: dict):
    sb = _get_supabase()
    if sb is None:
        return

    try:
        created_at = record.get("created_at")
        if not created_at:
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _to_iso(s: str) -> str:
            s = str(s).strip()
            if "T" in s and s.endswith("Z"):
                return s
            try:
                dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except Exception:
                try:
                    dt = datetime.datetime.strptime(s, "%Y-%m-%d")
                except Exception:
                    dt = datetime.datetime.utcnow()
            return dt.replace(microsecond=0).isoformat() + "Z"

        payload = {
            "folder_name": str(record.get("folder_name", "")),
            "analysis_name": str(record.get("analysis_name", "")),
            "created_at": _to_iso(created_at),
            "points": int(record.get("points", 0) or 0),
        }
        sb.table("history").insert(payload).execute()
    except Exception:
        return


def pick_video_file(folder: Path):
    if not folder.exists():
        return None
    mp4s = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        return None
    non_input = [
        p
        for p in mp4s
        if p.name.lower()
        not in ("input.mp4", "original.mp4", "temp_upload.mp4")
    ]
    return non_input[0] if non_input else mp4s[0]


def render_saved_plot(folder: Path):
    html_path = folder / "trajectory_plot.html"
    if html_path.exists():
        st.components.v1.html(
            html_path.read_text(encoding="utf-8"),
            height=720,
            scrolling=True,
        )
        return True
    return False


def apply_ui():
    st.set_page_config(
        page_title="Flight Analysis",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        [data-testid="stHeader"] { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        section[data-testid="stSidebar"] { display: none; }

        html, body, [class*="css"] {
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans";
        }

        .block-container {
            padding-top: 0.10rem;
            padding-bottom: 2.0rem;
            max-width: 1180px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stBlockBorderWrapper"]{
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }

        div[data-testid="stImage"] { margin-bottom: 0 !important; }

        .topbar-divider{
            height: 1px;
            background: rgba(15,23,42,0.10);
            margin: 2px 0 14px 0;
        }

        div[data-testid="stRadio"] div[role="radiogroup"]{
            flex-direction: row !important;
            justify-content: flex-end !important;
            align-items: flex-end !important;
            gap: 14px !important;
            border-bottom: 1px solid rgba(15,23,42,0.10) !important;
            padding-bottom: 2px !important;
            margin: 0 !important;
        }

        div[data-testid="stRadio"] label{
            margin: 0 !important;
            padding: 6px 2px 10px 2px !important;
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            cursor: pointer !important;
        }

        div[data-testid="stRadio"] label > div:first-child{
            display:none !important;
        }

        div[data-testid="stRadio"] label *{
            color: rgba(15,23,42,0.62) !important;
            font-weight: 760 !important;
            font-size: 0.98rem !important;
            line-height: 1 !important;
        }

        div[data-testid="stRadio"] label:hover *{
            color: rgba(15,23,42,0.92) !important;
        }

        div[data-testid="stRadio"] label:has(input:checked){
            border-bottom-color: #fd8c73 !important;
        }
        div[data-testid="stRadio"] label:has(input:checked) *{
            color: rgba(15,23,42,0.92) !important;
        }

        div[data-testid="stRadio"] input:checked ~ div *{
            color: rgba(15,23,42,0.92) !important;
        }

        .h1 { font-size: 2.5rem; font-weight: 900; letter-spacing: -0.05em; margin: 0 0 12px 0; line-height: 1.04; color:#0f172a; }
        .sub { color: rgba(15,23,42,0.62); font-size: 1.06rem; margin: 0 0 16px 0; }

        .chips { display:flex; flex-wrap:wrap; gap:10px; margin-top: 12px; }
        .chip {
            display:inline-flex;
            align-items:center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(15,23,42,0.10);
            background: rgba(15,23,42,0.02);
            font-size: 0.92rem;
            color: rgba(15,23,42,0.78);
        }

        .soft {
            background: rgba(15,23,42,0.02);
            border: 1px solid rgba(15,23,42,0.06);
            border-radius: 16px;
            padding: 14px 14px;
        }
        .featT { font-weight: 840; letter-spacing: -0.01em; color:#0f172a; margin-bottom: 6px; }
        .featD { color: rgba(15,23,42,0.62); font-size: 0.96rem; line-height: 1.55; }
        .muted { color: rgba(15,23,42,0.62); }

        .stTextInput input { border-radius: 14px !important; }
        div[data-testid="stFileUploader"] section { border-radius: 16px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_logo_svg(width_px: int = 72):
    if not LOGO_PATH.exists():
        return
    b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center;">
          <img src="data:image/svg+xml;base64,{b64}" style="width:{width_px}px; height:auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def topbar():
    if "page" not in st.session_state or st.session_state["page"] not in PAGES:
        st.session_state["page"] = "About"

    c1, c2 = st.columns([0.70, 0.30], vertical_alignment="bottom")

    with c1:
        render_logo_svg(width_px=150)

    with c2:
        current_idx = PAGES.index(st.session_state["page"])
        st.radio(
            "nav",
            PAGES,
            index=current_idx,
            horizontal=True,
            key="page",
            label_visibility="collapsed",
        )

    st.markdown("<div class='topbar-divider'></div>", unsafe_allow_html=True)


def about_page():
    left, right = st.columns([1.25, 0.75], gap="large", vertical_alignment="center")

    with left:
        st.markdown("<div class='h1'>Flight Analysis</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sub'>Upload a launch video and turn it into clean trajectory + performance signals, instantly.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="chips">
              <span class="chip">📈 Trajectory</span>
              <span class="chip">📏 Distance estimate</span>
              <span class="chip">🧭 Angle &amp; speed intuition</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin-top:14px;' class='muted'>No telemetry needed. Built for students, hobbyists, and quick experiments.</div>",
            unsafe_allow_html=True,
        )

    with right:
        if MISSION_PATH.exists():
            st.image(str(MISSION_PATH), width=420)

    st.divider()

    f1, f2, f3 = st.columns(3, gap="large")
    with f1:
        st.markdown(
            "<div class='soft'><div class='featT'>⚡ Fast workflow</div><div class='featD'>Drop a video, name the run, and generate a trajectory in one flow.</div></div>",
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            "<div class='soft'><div class='featT'>🧠 Motion extraction</div><div class='featD'>Detection-based tracking to collect points and visualize the path.</div></div>",
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            "<div class='soft'><div class='featT'>🗂️ Saved history</div><div class='featD'>Every launch is stored (Supabase Postgres). Re-open past runs and compare quickly.</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("<div class='featT'>Get in touch</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='muted'>Feedback welcome: <a href='mailto:palkiayp@gmail.com'><b>palkiayp@gmail.com</b></a></div>",
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    if not WEIGHTS_PATH.exists():
        return None
    return YOLO(str(WEIGHTS_PATH))


def analysis_page():
    st.markdown(
        "<div class='h1' style='font-size:1.95rem;'>Analyze</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sub'>Upload a .mp4 or .mov, name this run, then generate trajectory.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        uploaded_file = st.file_uploader("Video file", type=["mp4", "mov"], key="upload_video")
    with c2:
        analysis_name = st.text_input(
            "Analysis name",
            placeholder="e.g., cardboard_plane_test_01",
            key="analysis_name",
        )

    if not (uploaded_file and analysis_name):
        st.info("Upload a video and enter a name to start.")
        return

    RES_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in analysis_name if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
    folder_name = f"{timestamp}_{safe_name}" if safe_name else f"{timestamp}_analysis"
    output_folder = RES_DIR / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    temp_video = output_folder / "input.mp4"
    temp_video.write_bytes(uploaded_file.read())

    st.divider()
    st.markdown("**Preview**")
    st.video(str(temp_video))

    model = get_model()
    if model is None:
        st.error(f"Model weights not found: {WEIGHTS_PATH.as_posix()}")
        return

    st.divider()
    st.markdown("**Run detection**")

    with st.spinner("Running model..."):
        try:
            results = model(
                str(temp_video),
                save=True,
                show=False,
                project=str(RES_DIR),
                name=folder_name,
            )
        except Exception as e:
            st.error("Inference failed.")
            st.exception(e)
            return

    video_to_show = pick_video_file(output_folder)
    st.success(f"Saved: {folder_name}")
    if video_to_show:
        st.video(str(video_to_show))

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

    st.divider()
    st.markdown("**Trajectory**")

    if len(trajectory_points) < 2:
        st.info("Not enough detections to draw a trajectory.")
        add_history(
            {
                "folder_name": folder_name,
                "analysis_name": analysis_name,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "points": len(trajectory_points),
            }
        )
        return

    xs, ys = zip(*trajectory_points)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Object"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        title="Trajectory",
        xaxis_title="X (pixels)",
        yaxis_title="Y (pixels)",
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    try:
        fig.write_html(str(output_folder / "trajectory_plot.html"))
    except Exception:
        pass

    add_history(
        {
            "folder_name": folder_name,
            "analysis_name": analysis_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "points": len(trajectory_points),
        }
    )


def history_page():
    sb = _get_supabase()
    if sb is None:
        st.warning("History storage is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Secrets.")
    items = load_history()

    st.markdown("<div class='h1' style='font-size:1.95rem;'>History</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Your past launches are saved automatically after each analysis.</div>", unsafe_allow_html=True)

    if not items:
        st.info("No history yet. Run an analysis first.")
        return

    q = st.text_input("Search", placeholder="Type to filter…", key="history_search")
    filtered = items
    if q.strip():
        qq = q.strip().lower()
        filtered = [
            it
            for it in items
            if qq in str(it.get("analysis_name", "")).lower()
            or qq in str(it.get("folder_name", "")).lower()
        ]

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        labels = [
            f"{it.get('analysis_name','(no name)')} · {it.get('created_at','')} · pts:{it.get('points',0)}"
            for it in filtered
        ]
        idx = st.selectbox(
            "Select a launch",
            list(range(len(labels))),
            format_func=lambda i: labels[i],
            key="history_pick",
        )

    chosen = filtered[idx]
    folder = RES_DIR / chosen.get("folder_name", "")

    with right:
        st.markdown(f"**{chosen.get('analysis_name','(no name)')}**")
        st.caption(f"{chosen.get('created_at','')} · folder: {chosen.get('folder_name','')}")
        vid = pick_video_file(folder)
        if vid:
            st.video(str(vid))
        if not render_saved_plot(folder):
            st.info("No saved trajectory plot. Generate it from Analyze page first.")


def results_page():
    RES_DIR.mkdir(parents=True, exist_ok=True)

    st.markdown("<div class='h1' style='font-size:1.95rem;'>Results</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>Browse result folders saved on disk.</div>", unsafe_allow_html=True)

    subdirs = [d for d in RES_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        st.info("No result folders yet.")
        return

    subdirs_sorted = sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True)
    names = [p.name for p in subdirs_sorted]

    selected = st.selectbox("Select a folder", names, key="results_pick")
    selected_dir = RES_DIR / selected

    vid = pick_video_file(selected_dir)
    if vid:
        st.video(str(vid))
    if not render_saved_plot(selected_dir):
        st.info("No saved trajectory plot in this folder.")


apply_ui()
topbar()

page = st.session_state.get("page", "About")
if page == "About":
    about_page()
elif page == "Analyze":
    analysis_page()
elif page == "History":
    history_page()
else:
    results_page()
