"""
Page 1: Mission Overview — Deep Space Fleet Monitor
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
st.set_page_config(page_title="Mission Overview | Orbital Intelligence", page_icon="📊", layout="wide")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.components.charts import inject_global_css, apply_plotly_theme, PALETTE
from app.components.sidebar import render_sidebar
from app.components.metric_cards import metric_card
from src.utils.config import load_config, get_path
from src.data.data_loader import DataLoader
from src.utils.helpers import load_json, infer_mission

inject_global_css()
render_sidebar()

cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
loader = DataLoader(str(get_path(cfg, "data_dir")))
metrics_dir = get_path(cfg, "metrics_dir")

# --- Breadcrumb & Return to Entry Portal ---
col_back, col_crumbs = st.columns([2.2, 4])
with col_back:
    if st.button("🏠 ← Return to Home Entry Portal", key="btn_return_home"):
        st.switch_page("Home.py")
with col_crumbs:
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94A3B8; padding-top: 8px;">
        <span style="color: #38BDF8;">Orbital Intelligence</span> &nbsp;›&nbsp; 
        <b style="color: #FFFFFF;">01 · Mission Overview</b>
    </div>
    """, unsafe_allow_html=True)

# --- Page Header Banner ---
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(6, 9, 19, 0.95) 0%, rgba(13, 24, 46, 0.85) 50%, rgba(10, 16, 32, 0.95) 100%);
            border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.6rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00F0FF; letter-spacing: 0.12em; text-transform: uppercase;">
        ORBITAL INTELLIGENCE · LOCAL TELEMETRY ANALYSIS
    </div>
    <h1 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0 0.4rem 0; letter-spacing: -0.01em;">
        Mission Control Center
    </h1>
    <div style="font-size: 0.92rem; color: #CBD5E1; max-width: 820px; line-height: 1.5;">
        Intelligent spacecraft health monitoring across the locally available SMAP and MSL historical telemetry archive.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Load evaluation report if available ---
eval_report = None
eval_path = metrics_dir / "evaluation_report.json"
if eval_path.exists():
    try:
        eval_report = load_json(str(eval_path))
    except Exception:
        pass

anomaly_table = pd.DataFrame()
anomaly_table_path = metrics_dir / "anomaly_table.csv"
if anomaly_table_path.exists():
    try:
        anomaly_table = pd.read_csv(anomaly_table_path)
    except Exception:
        pass

status = loader.get_dataset_status()
channels = loader.get_channel_ids()
smap_chs = loader.get_smap_channels()
msl_chs = loader.get_msl_channels()

# The mission selector changes the operational KPIs and waveform inventory using
# the same local telemetry and model-evaluation artifacts used by the other decks.
selector_label, selector_control, _ = st.columns([1.2, 1.5, 3.3])
with selector_label:
    st.markdown("<div class='eyebrow' style='padding-top:2rem;'>Mission scope</div>", unsafe_allow_html=True)
with selector_control:
    mission_scope = st.selectbox("Mission selector", ["All Missions", "SMAP", "MSL"], label_visibility="collapsed")

scoped_channels = channels
scoped_events = anomaly_table
if mission_scope == "SMAP":
    scoped_channels = smap_chs
    scoped_events = anomaly_table[anomaly_table["mission"] == "SMAP"] if not anomaly_table.empty else anomaly_table
elif mission_scope == "MSL":
    scoped_channels = msl_chs
    scoped_events = anomaly_table[anomaly_table["mission"] == "MSL"] if not anomaly_table.empty else anomaly_table
scoped_metrics = eval_report.get(mission_scope, {}) if eval_report and mission_scope != "All Missions" else (eval_report or {}).get("overall", {})

# ---- KPI CARDS ----
st.markdown("### 🔑 Key Mission Indicators")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Telemetry Channels", str(len(scoped_channels)), color=PALETTE["accent"])
with c2:
    detected = len(scoped_events) if not scoped_events.empty else "0"
    metric_card("Detected Events", str(detected), color=PALETTE["critical"])
with c3:
    metric_card("SMAP Channels", str(len(smap_chs)), color=PALETTE["accent2"])
with c4:
    metric_card("MSL Channels", str(len(msl_chs)), color="#38BDF8")
with c5:
    f1 = f"{scoped_metrics.get('f1', 0):.3f}" if scoped_metrics else "Evaluation Pending"
    metric_card("Model F1-Score", f1, color=PALETTE["success"])
with c6:
    prec = f"{scoped_metrics.get('precision', 0):.3f}" if scoped_metrics else "Evaluation Pending"
    metric_card("Precision", prec, color=PALETTE["warning"])

st.markdown("<hr>", unsafe_allow_html=True)

# ---- MISSION CARDS ----
st.markdown("### 🚀 Mission Profile & Surveillance")
col1, col2 = st.columns(2)

with col1:
    smap_f1 = eval_report.get("SMAP", {}).get("f1", None) if eval_report else None
    f1_display = f"<div style='margin-top:0.6rem; color:{PALETTE['success']}; font-weight:700; font-family:\"JetBrains Mono\", monospace;'>LSTM F1: {smap_f1:.3f}</div>" if smap_f1 is not None else ""
    st.markdown(f"""
    <div class="hud-card" style="border-left: 4px solid {PALETTE['accent']};">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <span style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:{PALETTE['accent']}; font-weight:700;">
                    EARTH ORBITAL SATELLITE
                </span>
                <div style="font-size:1.25rem; font-weight:700; color:#FFFFFF; margin-top:2px;">
                    🌍 SMAP — Soil Moisture Active Passive
                </div>
            </div>
        </div>
        <div style="color:{PALETTE['text2']}; font-size:0.85rem; margin-top:0.5rem; font-family:'JetBrains Mono', monospace;">
            LAUNCH: 2015-01-31 &nbsp;|&nbsp; ORBIT: 685 km LEO &nbsp;|&nbsp; ACTIVE CHANNELS: {len(smap_chs)}
        </div>
        <div style="margin-top:0.6rem; font-size:0.88rem; color:#CBD5E1; line-height:1.5;">
            Monitors Earth's water cycle, crop productivity, and weather forecasting through high-precision L-band microwave telemetry.
        </div>
        {f1_display}
    </div>
    """, unsafe_allow_html=True)

with col2:
    msl_f1 = eval_report.get("MSL", {}).get("f1", None) if eval_report else None
    f1_display_msl = f"<div style='margin-top:0.6rem; color:{PALETTE['success']}; font-weight:700; font-family:\"JetBrains Mono\", monospace;'>LSTM F1: {msl_f1:.3f}</div>" if msl_f1 is not None else ""
    st.markdown(f"""
    <div class="hud-card" style="border-left: 4px solid {PALETTE['accent2']};">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <span style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:{PALETTE['accent2']}; font-weight:700;">
                    MARTIAN SURFACE ROVER
                </span>
                <div style="font-size:1.25rem; font-weight:700; color:#FFFFFF; margin-top:2px;">
                    🔴 MSL — Mars Science Laboratory (Curiosity)
                </div>
            </div>
        </div>
        <div style="color:{PALETTE['text2']}; font-size:0.85rem; margin-top:0.5rem; font-family:'JetBrains Mono', monospace;">
            LAUNCH: 2011-11-26 &nbsp;|&nbsp; LANDING: 2012-08-06 &nbsp;|&nbsp; ACTIVE CHANNELS: {len(msl_chs)}
        </div>
        <div style="margin-top:0.6rem; font-size:0.88rem; color:#CBD5E1; line-height:1.5;">
            Autonomous rover exploring Gale Crater on Mars to evaluate ancient habitability and Martian geologic evolution.
        </div>
        {f1_display_msl}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---- CHANNEL DISTRIBUTION CHART ----
if status["dataset_available"]:
    st.markdown("### 📡 Telemetry Channel Distribution")
    col1, col2 = st.columns([2, 1])

    with col1:
        # Channel file sizes (number of test time steps per channel)
        channel_steps = {}
        for cid in scoped_channels[:30]:  # limit for fast initial rendering
            try:
                data = loader.load_test(cid)
                channel_steps[cid] = data.shape[0]
            except Exception:
                pass

        if channel_steps:
            df_steps = pd.DataFrame(list(channel_steps.items()), columns=["Channel", "Test Steps"])
            df_steps["Mission"] = df_steps["Channel"].apply(infer_mission)
            df_steps = df_steps.sort_values("Test Steps", ascending=True)

            fig = go.Figure()
            for mission, color in [("SMAP", PALETTE["accent"]), ("MSL", PALETTE["accent2"])]:
                sub = df_steps[df_steps["Mission"] == mission]
                if not sub.empty:
                    fig.add_trace(go.Bar(
                        y=sub["Channel"], x=sub["Test Steps"],
                        name=mission, orientation="h",
                        marker=dict(color=color, line=dict(color=color, width=1)),
                        opacity=0.85,
                    ))
            fig = apply_plotly_theme(fig, "Test Telemetry Sequence Length per Channel", height=max(320, len(channel_steps)*19))
            fig.update_layout(barmode="stack", xaxis_title="Time Steps", yaxis_title="Channel")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie: SMAP vs MSL
        fig2 = go.Figure(go.Pie(
            labels=["SMAP", "MSL"],
            values=[len(smap_chs), len(msl_chs)],
            marker=dict(colors=[PALETTE["accent"], PALETTE["accent2"]], line=dict(color="#060913", width=2)),
            hole=0.45,
            textinfo="label+percent",
        ))
        fig2 = apply_plotly_theme(fig2, "Mission Fleet Composition", height=320)
        st.plotly_chart(fig2, use_container_width=True)

# ---- RECENT ANOMALIES ----
if not anomaly_table.empty:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🚨 Detected Spacecraft Anomaly Events")
    display_df = anomaly_table.sort_values("max_error", ascending=False).head(10).reset_index(drop=True)
    st.dataframe(
        display_df.style.format({
            "max_error": "{:.4f}",
            "mean_error": "{:.4f}",
            "threshold": "{:.4f}",
        }),
        use_container_width=True,
    )

# ---- EVALUATION SUMMARY ----
if eval_report:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📊 Detection Performance Overview")
    col1, col2, col3 = st.columns(3)
    for label, key, col in [("SMAP Mission", "SMAP", col1), ("MSL Mission", "MSL", col2), ("Overall Fleet", "overall", col3)]:
        sub = eval_report.get(key, {})
        with col:
            st.markdown(f"**{label}**")
            if sub:
                st.metric("Precision", f"{sub.get('precision', 0):.3f}")
                st.metric("Recall", f"{sub.get('recall', 0):.3f}")
                st.metric("F1-Score", f"{sub.get('f1', 0):.3f}")
                st.metric("TP / FP / FN", f"{sub.get('TP',0)} / {sub.get('FP',0)} / {sub.get('FN',0)}")
            else:
                st.info("No data yet")
