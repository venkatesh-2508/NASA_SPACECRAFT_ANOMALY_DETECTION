"""Model-information deck: architecture and local detection methodology."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Model Information | Orbital Intelligence", page_icon="ℹ️", layout="wide")

from app.components.charts import PALETTE, inject_global_css
from app.components.sidebar import render_sidebar
from src.utils.config import load_config


inject_global_css()
render_sidebar()
cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
model_cfg = cfg["model"]
preprocess_cfg = cfg["preprocessing"]
detection_cfg = cfg["detection"]

left, crumb = st.columns([2.2, 4])
with left:
    if st.button("← Return to Home Entry Portal", key="return_home"):
        st.switch_page("Home.py")
with crumb:
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:.76rem;color:#94A3B8;padding-top:.7rem;'>ORBITAL INTELLIGENCE &nbsp;›&nbsp; <b style='color:#F8FAFC;'>05 · MODEL INFORMATION</b></div>", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
  <div class="eyebrow">Detection methodology · local model specification</div>
  <h1 style="font-size:2.25rem;margin:.35rem 0 .45rem;">Model Architecture &amp; Methodology</h1>
  <div style="max-width:760px;color:#CBD5E1;font-size:.93rem;line-height:1.65;">A concise reference for the recurrent forecasting model, residual-based event detection, and event-level evaluation used throughout Mission Control.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("### Neural forecasting architecture")
architecture, specification = st.columns([1.3, 1])
with architecture:
    st.markdown("""
    <div class="hud-card" style="min-height:245px;">
      <div class="eyebrow">Sequence flow</div>
      <div style="font-family:'Space Grotesk';font-size:1.2rem;color:#F8FAFC;font-weight:700;margin:.55rem 0;">Telemetry window → LSTM → next-value forecast</div>
      <div style="font-family:JetBrains Mono,monospace;color:#CBD5E1;font-size:.78rem;line-height:2.05;">INPUT WINDOW&nbsp;&nbsp; [batch, historical steps, command features]<br><span style="color:#64748B;">↓</span><br><span style="color:#38BDF8;">RECURRENT LAYERS&nbsp;&nbsp; temporal state encoding</span><br><span style="color:#64748B;">↓</span><br><span style="color:#A78BFA;">LINEAR OUTPUT&nbsp;&nbsp; next telemetry estimate ŷₜ</span><br><span style="color:#64748B;">↓</span><br><span style="color:#22C55E;">RESIDUAL&nbsp;&nbsp; |yₜ − ŷₜ|</span></div>
    </div>
    """, unsafe_allow_html=True)
with specification:
    st.markdown(f"""
    <div class="hud-card" style="min-height:245px;border-left:3px solid {PALETTE['accent2']};">
      <div class="eyebrow">Configured local model</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.85rem;margin-top:1rem;">
        <div><div style="color:#94A3B8;font-size:.68rem;font-family:JetBrains Mono,monospace;">HIDDEN UNITS</div><div style="color:#F8FAFC;font-size:1.25rem;font-weight:700;">{model_cfg['hidden_size']}</div></div>
        <div><div style="color:#94A3B8;font-size:.68rem;font-family:JetBrains Mono,monospace;">LSTM LAYERS</div><div style="color:#F8FAFC;font-size:1.25rem;font-weight:700;">{model_cfg['num_layers']}</div></div>
        <div><div style="color:#94A3B8;font-size:.68rem;font-family:JetBrains Mono,monospace;">WINDOW LENGTH</div><div style="color:#F8FAFC;font-size:1.25rem;font-weight:700;">{preprocess_cfg['sequence_length']}</div></div>
        <div><div style="color:#94A3B8;font-size:.68rem;font-family:JetBrains Mono,monospace;">DROPOUT</div><div style="color:#F8FAFC;font-size:1.25rem;font-weight:700;">{model_cfg['dropout']:.1f}</div></div>
      </div>
      <div style="margin-top:1.15rem;color:#CBD5E1;font-size:.8rem;">Loss: mean-squared forecast error · Optimizer settings are recorded in <code>config.yaml</code>.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Detection pipeline")
pipeline = [
    ("01", "Normalize", "Scale local training telemetry before window generation.", PALETTE["accent"]),
    ("02", "Forecast", "Use the LSTM to predict the next telemetry measurement.", PALETTE["accent2"]),
    ("03", "Calculate residual", "Compute absolute forecast error for every available output step.", PALETTE["warning"]),
    ("04", "Smooth", f"Apply EWMA smoothing with span {detection_cfg['smoothing_span']} to suppress transient noise.", "#38BDF8"),
    ("05", "Calibrate threshold", f"Search μ + zσ candidates from z={detection_cfg['z_score_min']} to {detection_cfg['z_score_max']}.", PALETTE["success"]),
    ("06", "Prune sequences", f"Discard low-impact event candidates using the configured {detection_cfg['min_percent_change']:.0%} error-drop rule.", PALETTE["critical"]),
]
for number, title, detail, color in pipeline:
    st.markdown(f"""
    <div class="hud-card" style="border-left:3px solid {color};margin:.45rem 0;padding:.85rem 1rem;">
      <div style="display:flex;gap:.8rem;align-items:baseline;"><span style="font-family:JetBrains Mono,monospace;color:{color};font-size:.69rem;font-weight:700;">{number}</span><span style="font-family:'Space Grotesk';font-size:1rem;font-weight:700;color:#F8FAFC;">{title}</span></div>
      <div style="color:#CBD5E1;font-size:.84rem;margin:.32rem 0 0 1.9rem;">{detail}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Evaluation convention")
st.markdown("""
<div class="hud-card" style="border-left:3px solid #38BDF8;"><div style="font-family:'Space Grotesk';font-size:1.08rem;font-weight:700;color:#F8FAFC;">Event-level overlap scoring</div><p style="margin:.45rem 0 0;max-width:950px;font-size:.87rem;line-height:1.65;">Detected and expert-labeled anomaly intervals are evaluated by temporal overlap. Precision, recall, and F1 are computed from true-positive, false-positive, and false-negative event counts. The dashboard leaves true negatives unavailable because they are not persisted by the local event evaluator.</p></div>
""", unsafe_allow_html=True)
st.caption("Method reference: Hundman et al., “Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding,” KDD 2018.")
