"""Orbital Intelligence cinematic entry portal for the local telemetry application."""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.charts import inject_global_css
from src.data.data_loader import DataLoader
from src.utils.config import get_path, load_config


st.set_page_config(page_title="Orbital Intelligence", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")
inject_global_css()


def inject_landing_css() -> None:
    st.markdown("""
    <style>
    [data-testid='stSidebar'] {display:none;}
    .landing-shell {position:relative;overflow:hidden;border:1px solid rgba(56,189,248,.19);border-radius:20px;min-height:650px;padding:1.1rem 3.4rem 3.2rem;background:linear-gradient(125deg,rgba(8,13,24,.97),rgba(10,20,38,.84) 48%,rgba(20,15,43,.82));}
    .landing-shell:before {content:'';position:absolute;inset:0;pointer-events:none;opacity:.55;background-image:radial-gradient(circle at 12% 16%,#fff 0 1px,transparent 1.5px),radial-gradient(circle at 24% 40%,#bae6fd 0 1px,transparent 1.5px),radial-gradient(circle at 42% 22%,#fff 0 .8px,transparent 1.4px),radial-gradient(circle at 55% 75%,#c4b5fd 0 1px,transparent 1.5px),radial-gradient(circle at 76% 15%,#fff 0 .8px,transparent 1.4px),radial-gradient(circle at 88% 48%,#bae6fd 0 1px,transparent 1.5px),radial-gradient(circle at 71% 68%,#fff 0 1px,transparent 1.5px);background-size:230px 190px,290px 250px,195px 250px,310px 210px,270px 300px,210px 180px,350px 250px;animation:drift-stars 24s linear infinite;}
    .landing-shell:after {content:'';position:absolute;width:720px;height:720px;border:1px solid rgba(56,189,248,.15);border-radius:50%;right:-290px;bottom:-390px;box-shadow:0 0 0 76px rgba(56,189,248,.025),0 0 0 150px rgba(139,92,246,.018);pointer-events:none;}
    .landing-nav,.hero-grid,.landing-footer {position:relative;z-index:1;}.landing-nav {display:flex;justify-content:space-between;align-items:center;padding:.15rem 0 1.6rem;}
    .brand-lockup {display:flex;gap:.65rem;align-items:center}.brand-mark {width:32px;height:32px;border-radius:10px;display:grid;place-items:center;border:1px solid #38BDF8;background:linear-gradient(135deg,rgba(56,189,248,.22),rgba(139,92,246,.22));font-family:'Space Grotesk';font-size:.77rem;font-weight:800;color:#e0f2fe}.brand-name {font-family:'Space Grotesk';font-weight:700;color:#f8fafc;letter-spacing:.08em;font-size:.88rem}.brand-sub {font-family:'JetBrains Mono';font-size:.55rem;color:#38BDF8;letter-spacing:.17em}.local-badge {font-family:'JetBrains Mono';font-size:.61rem;letter-spacing:.08em;color:#bbf7d0;border:1px solid rgba(34,197,94,.3);padding:.35rem .55rem;border-radius:999px;background:rgba(34,197,94,.08)}
    .hero-grid {display:grid;grid-template-columns:1.13fr .87fr;gap:2rem;align-items:center;min-height:495px}.hero-copy {padding:2rem 0 1.4rem}.hero-kicker {font-family:'JetBrains Mono';font-size:.68rem;letter-spacing:.15em;color:#7dd3fc;display:flex;align-items:center;gap:.5rem}.hero-kicker:before {content:'';width:25px;height:1px;background:#38BDF8}.hero-title {font-family:'Space Grotesk';font-size:clamp(3rem,6vw,5.8rem);line-height:.92;letter-spacing:-.07em;color:#f8fafc;font-weight:700;margin:.9rem 0 1rem}.hero-title span {display:block;background:linear-gradient(100deg,#e0f2fe 0%,#38bdf8 56%,#a78bfa 100%);-webkit-background-clip:text;background-clip:text;color:transparent}.hero-product {font-family:'JetBrains Mono';font-size:.69rem;color:#cbd5e1;letter-spacing:.13em;margin-bottom:1.25rem}.hero-lede {max-width:580px;color:#cbd5e1;font-size:1rem;line-height:1.75;margin:0 0 1.4rem}.hero-quote {color:#f8fafc;font-family:'Space Grotesk';font-size:1.25rem;line-height:1.35;max-width:520px;margin-bottom:1.6rem}.hero-quote em {color:#7dd3fc;font-style:normal}.hero-actions {display:flex;gap:.75rem;align-items:center;flex-wrap:wrap}
    .orbital-visual {position:relative;height:420px;min-width:320px;display:grid;place-items:center}.planet {position:absolute;width:245px;height:245px;border-radius:50%;background:radial-gradient(circle at 34% 29%,#dbeafe 0%,#38bdf8 8%,#176a98 26%,#0b2944 52%,#071827 73%,#030914 100%);box-shadow:-24px 21px 40px rgba(0,0,0,.65),0 0 70px rgba(56,189,248,.22),inset -25px -15px 35px rgba(0,0,0,.42);overflow:hidden}.planet:before {content:'';position:absolute;inset:16px 4px 40px 28px;background:linear-gradient(136deg,transparent 23%,rgba(186,230,253,.2) 24% 26%,transparent 27% 41%,rgba(186,230,253,.13) 42% 45%,transparent 46%);opacity:.85;transform:rotate(-20deg)}.planet:after {content:'';position:absolute;inset:-50px;background:repeating-linear-gradient(13deg,transparent 0 14px,rgba(255,255,255,.035) 15px 16px);transform:rotate(18deg)}.orbit {position:absolute;width:420px;height:160px;border:1px solid rgba(125,211,252,.3);border-radius:50%;transform:rotate(-24deg);box-shadow:0 0 26px rgba(56,189,248,.06);animation:orbit-breathe 7s ease-in-out infinite}.orbit.two {width:300px;height:380px;transform:rotate(58deg);border-color:rgba(167,139,250,.23)}.satellite {position:absolute;width:24px;height:13px;border:1px solid #bae6fd;background:#1e3a5f;transform:translate(167px,-44px) rotate(-16deg);box-shadow:0 0 15px rgba(56,189,248,.7)}.satellite:before,.satellite:after {content:'';position:absolute;top:2px;width:16px;height:8px;background:repeating-linear-gradient(90deg,#38bdf8 0 2px,#1e3a5f 2px 4px);border:1px solid rgba(186,230,253,.6)}.satellite:before {right:23px}.satellite:after {left:23px}.data-tag {position:absolute;right:5%;top:14%;border:1px solid rgba(56,189,248,.24);background:rgba(8,13,24,.7);backdrop-filter:blur(10px);border-radius:9px;padding:.7rem .8rem;font-family:'JetBrains Mono';font-size:.62rem;color:#cbd5e1;line-height:1.7}.data-tag b {display:block;color:#7dd3fc;letter-spacing:.1em;font-size:.57rem}.data-tag.bottom {right:auto;left:5%;top:auto;bottom:10%;border-color:rgba(167,139,250,.24)}.data-tag.bottom b {color:#c4b5fd}
    .landing-footer {border-top:1px solid rgba(36,50,71,.72);padding-top:1.2rem;display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}.signal-card {font-family:'JetBrains Mono';font-size:.61rem;color:#94a3b8;letter-spacing:.055em}.signal-card b {display:block;color:#e2e8f0;font-family:'Space Grotesk';font-size:.9rem;letter-spacing:0;margin-bottom:.2rem}.signal-dot {display:inline-block;width:6px;height:6px;background:#22c55e;border-radius:50%;margin-right:.4rem;box-shadow:0 0 9px #22c55e}
    @keyframes drift-stars {from {background-position:0 0,0 0,0 0,0 0,0 0,0 0,0 0}to {background-position:34px 90px,-46px 120px,55px 80px,-55px 105px,40px 100px,-50px 95px,75px 80px}}@keyframes orbit-breathe {50% {opacity:.7;transform:rotate(-24deg) scale(1.025)}}
    @media(max-width:860px) {.landing-shell {padding:1.1rem 1.3rem 2rem;border-radius:13px}.hero-grid {grid-template-columns:1fr;min-height:auto}.hero-copy {padding:1.3rem 0}.orbital-visual {height:285px;order:-1;transform:scale(.75);margin:-45px 0 -65px}.landing-footer {grid-template-columns:1fr;gap:.75rem}.local-badge {display:none}.hero-title {font-size:3.25rem}.data-tag{display:none}}@media(prefers-reduced-motion:reduce){.landing-shell:before,.orbit{animation:none!important}}
    </style>""", unsafe_allow_html=True)


def start_mission(loader: DataLoader) -> None:
    """Show initialization information derived from local files, then enter mission control."""
    status = st.status("MISSION INITIALIZATION", expanded=True)
    status.write("Checking the local telemetry analysis engine…")
    time.sleep(0.25)
    dataset = loader.get_dataset_status()
    if dataset["dataset_available"]:
        status.write(f"Loaded {dataset['num_test_files']} local telemetry channels and {dataset['num_labels']} dataset label records.")
    else:
        status.write("Local telemetry dataset is not complete; mission control will show availability details.")
    time.sleep(0.25)
    prediction_count = len(list((PROJECT_ROOT / "results" / "predictions").glob("*_detection.json")))
    status.write(f"Prepared anomaly-detection interface with {prediction_count} available local prediction result set(s).")
    time.sleep(0.25)
    status.update(label="MISSION CONTROL READY", state="complete", expanded=False)
    st.session_state["orbital_launched"] = True
    time.sleep(0.3)
    st.switch_page("pages/01_Mission_Overview.py")


cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
loader = DataLoader(str(get_path(cfg, "data_dir")))
inject_landing_css()
if st.session_state.pop("launch_mission", False):
    start_mission(loader)

st.markdown("""
<div class="landing-shell"><div class="landing-nav"><div class="brand-lockup"><div class="brand-mark">OI</div><div><div class="brand-name">ORBITAL</div><div class="brand-sub">INTELLIGENCE</div></div></div><div class="local-badge">● LOCAL DATASET MODE</div></div>
<div class="hero-grid"><div class="hero-copy"><div class="hero-kicker">DEEP-SPACE TELEMETRY INTELLIGENCE</div><div class="hero-title">ORBITAL <span>INTELLIGENCE</span></div><div class="hero-product">NASA SPACECRAFT ANOMALY DETECTION</div><p class="hero-quote">Monitoring the Unknown.<br><em>Predicting the Unexpected.</em></p><p class="hero-lede">An intelligent spacecraft health monitoring platform powered by locally available NASA telemetry data and LSTM-based anomaly detection.</p></div>
<div class="orbital-visual" aria-label="Stylised Earth orbital telemetry visual"><div class="orbit"></div><div class="orbit two"></div><div class="planet"></div><div class="satellite"></div><div class="data-tag"><b>MISSION LINK</b>SMAP / MSL<br><span style="color:#86efac">LOCAL • READY</span></div><div class="data-tag bottom"><b>ANALYSIS ENGINE</b>LSTM + DYNAMIC<br>THRESHOLDING</div></div></div></div>""", unsafe_allow_html=True)
cta_left, cta_right, _ = st.columns([1.85, 1.8, 3.35])
with cta_left:
    if st.button("LAUNCH MISSION CONTROL  →", key="launch_control", use_container_width=True):
        st.session_state["launch_mission"] = True
        st.rerun()
with cta_right:
    if st.button("EXPLORE THE TECHNOLOGY", key="explore_technology", use_container_width=True):
        st.switch_page("pages/05_Model_Information.py")
st.markdown("""
<div class="landing-footer"><div class="signal-card"><b><span class="signal-dot"></span>Historical telemetry</b>NASA SMAP and MSL dataset archive</div><div class="signal-card"><b>Sequence intelligence</b>LSTM forecasting with dynamic thresholds</div><div class="signal-card"><b>Mission-grade analysis</b>Evidence-backed events, not simulated live contact</div></div>""", unsafe_allow_html=True)
