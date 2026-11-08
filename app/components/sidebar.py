"""Mission-control navigation shared by all operational pages."""
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _local_system_state() -> tuple[bool, int]:
    """Read only local artifact presence; this is not a live mission connection."""
    data_dir = PROJECT_ROOT / "data"
    ready = (data_dir / "labeled_anomalies.csv").exists() and any((data_dir / "test").glob("*.npy"))
    checkpoints = len(list((PROJECT_ROOT / "models" / "checkpoints").glob("*_best.pt")))
    return ready, checkpoints


def render_sidebar() -> None:
    """Render the compact Orbital Intelligence mission navigation."""
    dataset_ready, checkpoint_count = _local_system_state()
    dataset_label = "DATASET READY" if dataset_ready else "DATASET UNAVAILABLE"
    dataset_color = "#22C55E" if dataset_ready else "#F59E0B"
    model_label = f"{checkpoint_count} MODEL ARTIFACT{'S' if checkpoint_count != 1 else ''}" if checkpoint_count else "EVALUATION PENDING"
    model_color = "#38BDF8" if checkpoint_count else "#F59E0B"
    with st.sidebar:
        st.markdown("""
        <div style="padding:.45rem .35rem 1.1rem;border-bottom:1px solid #243247;">
          <div style="display:flex;align-items:center;gap:.7rem;"><div style="width:32px;height:32px;border:1px solid #38BDF8;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,rgba(56,189,248,.2),rgba(139,92,246,.18));color:#E0F2FE;font-weight:800;font-size:.82rem;">OI</div><div><div style="font-family:'Space Grotesk',sans-serif;color:#F8FAFC;font-weight:700;font-size:.92rem;letter-spacing:.07em;">ORBITAL</div><div style="color:#38BDF8;font-family:'JetBrains Mono',monospace;font-size:.57rem;letter-spacing:.17em;">INTELLIGENCE</div></div></div>
          <div style="font-family:'JetBrains Mono',monospace;color:#94A3B8;font-size:.61rem;letter-spacing:.08em;margin-top:.75rem;">SPACECRAFT ANOMALY DETECTION</div>
        </div><div style="font-family:'JetBrains Mono',monospace;color:#64748B;font-size:.62rem;font-weight:600;letter-spacing:.14em;margin:1.1rem .2rem .35rem;">MISSION CONTROL</div>""", unsafe_allow_html=True)
        st.page_link("Home.py", label="Entry Portal")
        st.page_link("pages/01_Mission_Overview.py", label="01  Mission Overview")
        st.page_link("pages/02_Telemetry_Explorer.py", label="02  Telemetry Explorer")
        st.page_link("pages/03_Anomaly_Monitor.py", label="03  Anomaly Monitor")
        st.page_link("pages/04_Model_Analytics.py", label="04  Model Analytics")
        st.page_link("pages/05_Model_Information.py", label="05  Model Information")
        st.page_link("pages/06_Dataset_Status.py", label="06  Dataset Status")
        st.markdown(f"""
        <div style="position:fixed;bottom:1.3rem;left:1rem;right:1rem;background:rgba(17,28,46,.72);border:1px solid #243247;border-radius:10px;padding:.75rem .8rem;">
          <div style="display:flex;align-items:center;gap:.42rem;font-family:'JetBrains Mono',monospace;font-size:.6rem;color:{dataset_color};letter-spacing:.06em;"><span style="width:6px;height:6px;border-radius:50%;background:{dataset_color};box-shadow:0 0 8px {dataset_color};"></span>{dataset_label}</div>
          <div style="margin-top:.48rem;color:{model_color};font-family:'JetBrains Mono',monospace;font-size:.58rem;letter-spacing:.05em;">{model_label}</div><div style="margin-top:.6rem;color:#64748B;font-family:'JetBrains Mono',monospace;font-size:.57rem;letter-spacing:.08em;">ORBITAL INTELLIGENCE · v1.0.0</div>
        </div>""", unsafe_allow_html=True)
