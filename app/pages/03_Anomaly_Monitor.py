"""
Page 3: Anomaly Monitor — Deep Space Threat Surveillance
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
st.set_page_config(page_title="Anomaly Monitor | Orbital Intelligence", page_icon="🚨", layout="wide")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.components.charts import inject_global_css, apply_plotly_theme, PALETTE
from app.components.sidebar import render_sidebar
from src.utils.config import load_config, get_path
from src.data.data_loader import DataLoader
from src.utils.helpers import load_json

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
        <b style="color: #FFFFFF;">03 · Tactical Anomaly Monitor</b>
    </div>
    """, unsafe_allow_html=True)

# --- Page Header Banner ---
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(6, 9, 19, 0.95) 0%, rgba(13, 24, 46, 0.85) 50%, rgba(10, 16, 32, 0.95) 100%);
            border: 1px solid rgba(255, 51, 102, 0.35); border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.6rem;
            box-shadow: 0 8px 32px rgba(255, 51, 102, 0.15);">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #FF3366; letter-spacing: 0.12em; text-transform: uppercase;">
        LOCAL MODEL EVENTS · ANOMALY SURVEILLANCE LOG
    </div>
    <h1 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0 0.4rem 0; letter-spacing: -0.01em;">
        Spacecraft Anomaly Monitor
    </h1>
    <div style="font-size: 0.92rem; color: #CBD5E1; max-width: 820px; line-height: 1.5;">
        Automated anomaly event registry generated via LSTM forecasting, dynamic nonparametric thresholding, 
        and false-positive sequence pruning.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Load anomaly table ---
table_path = metrics_dir / "anomaly_table.csv"
if not table_path.exists():
    st.warning("""
    **No anomaly detection results found.**  
    Please run the detection pipeline:
    ```bash
    python run_detection.py --channel P-1
    ```
    """)
    st.stop()

anomaly_df = pd.read_csv(table_path)

if anomaly_df.empty:
    st.info("No anomaly events detected by the current model threshold.")
    st.stop()

# ---- FILTER CONTROLS ----
st.markdown("### 🔍 Telemetry Incident Filters")
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        missions = ["All"] + sorted(anomaly_df["mission"].unique().tolist())
        sel_mission = st.selectbox("Spacecraft Mission", missions)
    with col2:
        channels = ["All"] + sorted(anomaly_df["channel"].unique().tolist())
        sel_channel = st.selectbox("Telemetry Channel", channels)
    with col3:
        sort_by = st.selectbox("Sort Priority", ["max_error", "duration", "start", "mean_error"])
    with col4:
        statuses = ["All"] + sorted(anomaly_df["status"].dropna().unique().tolist()) if "status" in anomaly_df.columns else ["All"]
        sel_status = st.selectbox("Detection Status", statuses)

# Apply filters
df = anomaly_df.copy()
if sel_mission != "All":
    df = df[df["mission"] == sel_mission]
if sel_channel != "All":
    df = df[df["channel"] == sel_channel]
if sel_status != "All" and "status" in df.columns:
    df = df[df["status"] == sel_status]
df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

# ---- KPI STRIP ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Flagged Events", len(df))
with c2:
    st.metric("SMAP Events", len(df[df["mission"] == "SMAP"]))
with c3:
    st.metric("MSL Events", len(df[df["mission"] == "MSL"]))
with c4:
    avg_dur = f"{df['duration'].mean():.0f} timesteps" if not df.empty else "—"
    st.metric("Average Anomaly Duration", avg_dur)

st.markdown("<hr>", unsafe_allow_html=True)

# ---- ANOMALY TABLE ----
st.markdown(f"### 📋 Incident Registry ({len(df)} Sequences Flagged)")

# Style table
styled = df.style.format({
    "max_error": "{:.4f}",
    "mean_error": "{:.4f}",
    "threshold": "{:.4f}",
}).background_gradient(subset=["max_error"], cmap="Reds")

st.dataframe(styled, use_container_width=True, height=280)

# Export CSV
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Export Incident Telemetry Log (CSV)",
    data=csv_data,
    file_name="nasa_detected_anomalies.csv",
    mime="text/csv",
)

# Selecting a logged sequence reveals only fields emitted by the local detector;
# no severity value is inferred or fabricated in this interface.
if not df.empty:
    st.markdown("#### Selected event detail")
    event_options = {
        f"{row.mission} · {row.channel} · steps {int(row.start):,}–{int(row.end):,}": index
        for index, row in df.iterrows()
    }
    selected_event = st.selectbox("Incident", list(event_options), label_visibility="collapsed")
    event = df.loc[event_options[selected_event]]
    detail_cols = st.columns(6)
    for column, label, value in [
        (detail_cols[0], "Mission", event["mission"]),
        (detail_cols[1], "Channel", event["channel"]),
        (detail_cols[2], "Duration", f"{int(event['duration']):,} steps"),
        (detail_cols[3], "Max error", f"{event['max_error']:.5f}"),
        (detail_cols[4], "Threshold", f"{event['threshold']:.5f}"),
        (detail_cols[5], "Status", event.get("status", "—")),
    ]:
        with column:
            st.metric(label, value)

st.markdown("<hr>", unsafe_allow_html=True)

# ---- DISTRIBUTIONS ----
col1, col2 = st.columns(2)
with col1:
    st.markdown("### ⚡ Prediction Error Severity Distribution")
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["max_error"],
            nbinsx=20,
            name="Max Error",
            marker=dict(color=PALETTE["critical"], line=dict(color="#FF3366", width=1)),
            opacity=0.85,
        ))
        fig = apply_plotly_theme(fig, "Histogram of Peak Telemetry Residual Error", height=340)
        fig.update_layout(xaxis_title="Max Error Deviation", yaxis_title="Number of Incidents")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### ⏱️ Anomaly Duration Distribution")
    if not df.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df["duration"],
            nbinsx=20,
            name="Duration",
            marker=dict(color=PALETTE["accent2"], line=dict(color="#8B5CF6", width=1)),
            opacity=0.85,
        ))
        fig2 = apply_plotly_theme(fig2, "Histogram of Anomaly Duration (Time Steps)", height=340)
        fig2.update_layout(xaxis_title="Incident Duration (Steps)", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

# ---- CHANNEL BREAKDOWN ----
if not df.empty:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📡 Anomaly Frequency per Spacecraft Channel")
    channel_counts = df.groupby(["channel", "mission"]).size().reset_index(name="count")
    channel_counts = channel_counts.sort_values("count", ascending=True)
    fig3 = go.Figure()
    for mission, color in [("SMAP", PALETTE["accent"]), ("MSL", PALETTE["accent2"])]:
        sub = channel_counts[channel_counts["mission"] == mission]
        if not sub.empty:
            fig3.add_trace(go.Bar(
                y=sub["channel"], x=sub["count"], name=mission,
                orientation="h", marker_color=color, opacity=0.85
            ))
    fig3 = apply_plotly_theme(fig3, "Incident Frequency by Sensor Channel", height=max(280, len(channel_counts) * 25))
    fig3.update_layout(barmode="stack", xaxis_title="Detected Incidents", yaxis_title="Channel ID")
    st.plotly_chart(fig3, use_container_width=True)
