"""
Page 2: Telemetry Explorer — Real-Time Waveform & Error Analysis
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
st.set_page_config(page_title="Telemetry Explorer | Orbital Intelligence", page_icon="📡", layout="wide")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.charts import inject_global_css, apply_plotly_theme, PALETTE
from app.components.sidebar import render_sidebar
from src.utils.config import load_config, get_path
from src.data.data_loader import DataLoader
from src.utils.helpers import load_json, infer_mission

inject_global_css()
render_sidebar()

cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
loader = DataLoader(str(get_path(cfg, "data_dir")))
pred_dir = get_path(cfg, "predictions_dir")

# --- Breadcrumb & Return to Entry Portal ---
col_back, col_crumbs = st.columns([2.2, 4])
with col_back:
    if st.button("🏠 ← Return to Home Entry Portal", key="btn_return_home"):
        st.switch_page("Home.py")
with col_crumbs:
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94A3B8; padding-top: 8px;">
        <span style="color: #38BDF8;">Orbital Intelligence</span> &nbsp;›&nbsp; 
        <b style="color: #FFFFFF;">02 · Telemetry Waveform Explorer</b>
    </div>
    """, unsafe_allow_html=True)

# --- Page Header Banner ---
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(6, 9, 19, 0.95) 0%, rgba(13, 24, 46, 0.85) 50%, rgba(10, 16, 32, 0.95) 100%);
            border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.6rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00F0FF; letter-spacing: 0.12em; text-transform: uppercase;">
        LOCAL TELEMETRY SENSORS · WAVEFORM FORECASTING
    </div>
    <h1 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0 0.4rem 0; letter-spacing: -0.01em;">
        Telemetry Stream Explorer
    </h1>
    <div style="font-size: 0.92rem; color: #CBD5E1; max-width: 820px; line-height: 1.5;">
        Multi-channel time-series visualizer. Compare ground-truth observations against LSTM predictions, 
        EWMA-smoothed error signals, and dynamically calibrated detection thresholds.
    </div>
</div>
""", unsafe_allow_html=True)

if not loader.dataset_available():
    st.error("Dataset not available. Please check the Dataset Status page.")
    st.stop()

# ---- CONTROLS ----
channels = loader.get_channel_ids()

with st.container():
    col1, col2, col3 = st.columns([1.5, 1.5, 3])
    with col1:
        mission_filter = st.selectbox("Mission Filter", ["All", "SMAP", "MSL"])
        if mission_filter == "SMAP":
            filtered_channels = loader.get_smap_channels()
        elif mission_filter == "MSL":
            filtered_channels = loader.get_msl_channels()
        else:
            filtered_channels = channels

        if not filtered_channels:
            st.warning("No channels found.")
            st.stop()

        channel_id = st.selectbox("Telemetry Channel", filtered_channels)

    with col2:
        split = st.radio("Telemetry Split", ["Test", "Train"], horizontal=True)
        # Load raw data
        try:
            if split == "Train":
                raw = loader.load_train(channel_id)
            else:
                raw = loader.load_test(channel_id)
            n_steps, n_features = raw.shape
            st.caption(f"Sensor shape: **{n_steps:,}** time steps × **{n_features}** channels")
        except Exception as e:
            st.error(f"Could not load data: {e}")
            st.stop()

    with col3:
        start_pct, end_pct = st.slider("Time Window Slice (%)", 0, 100, (0, 100))

telemetry = raw[:, 0]
n = len(telemetry)
start_idx = int(start_pct / 100 * n)
end_idx = int(end_pct / 100 * n)
time_idx = np.arange(start_idx, end_idx)
tel_slice = telemetry[start_idx:end_idx]

# ---- LOAD DETECTION RESULTS ----
result = None
arrays = None
detection_path = pred_dir / f"{channel_id}_detection.json"
arrays_path = pred_dir / f"{channel_id}_arrays.npz"

if detection_path.exists():
    try:
        result = load_json(str(detection_path))
    except Exception:
        pass

if arrays_path.exists():
    try:
        arrays = np.load(str(arrays_path))
    except Exception:
        pass

st.markdown("<hr>", unsafe_allow_html=True)

# ---- PLOTS ----
if result and split == "Test":
    seq_offset = result.get("sequence_offset", 250)
    if arrays is not None:
        actual = arrays["actual"]
        predicted = arrays["predicted"]
        smoothed_error = arrays["smoothed_error"]
    else:
        actual = np.array(result.get("actual", []))
        predicted = np.array(result.get("predicted", []))
        smoothed_error = np.array(result.get("smoothed_error", []))
    threshold = result.get("threshold", None)
    gt_intervals = result.get("ground_truth", [])
    pred_intervals = result.get("pruned_sequences", [])

    # Adjust indices to full test sequence
    full_time = np.arange(seq_offset, seq_offset + len(actual))

    # Clip to selected range
    mask = (full_time >= start_idx) & (full_time < end_idx)
    t_slice = full_time[mask]
    a_slice = actual[mask]
    p_slice = predicted[mask] if len(predicted) == len(actual) else a_slice
    e_slice = smoothed_error[mask] if len(smoothed_error) == len(actual) else None

    # Filter intervals
    gt_filtered = [(s, e) for s, e in gt_intervals if e >= start_idx and s <= end_idx]
    pred_filtered = [(s, e) for s, e in pred_intervals if e >= start_idx and s <= end_idx]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
        subplot_titles=(f"Sensor Telemetry Waveform: Actual vs LSTM Forecast", "EWMA Residual Error vs Dynamic Threshold")
    )

    # Row 1: Actual vs Predicted
    fig.add_trace(go.Scatter(
        x=t_slice, y=a_slice, name="Actual Telemetry",
        line=dict(color=PALETTE["accent"], width=1.4)
    ), row=1, col=1)

    if len(p_slice) > 0:
        fig.add_trace(go.Scatter(
            x=t_slice, y=p_slice, name="LSTM Forecast (ŷ)",
            line=dict(color=PALETTE["accent2"], width=1.4, dash="dot")
        ), row=1, col=1)

    # Ground-truth anomalies
    for gi, (gs, ge) in enumerate(gt_filtered):
        fig.add_vrect(
            x0=gs, x1=ge, fillcolor="rgba(255, 51, 102, 0.22)", line_width=1,
            line_color=PALETTE["critical"],
            annotation_text="Ground Truth Anomaly" if gi == 0 else "",
            annotation_font_color=PALETTE["critical"], row=1, col=1
        )

    # Predicted anomalies
    for pi, (ps, pe) in enumerate(pred_filtered):
        fig.add_vrect(
            x0=ps, x1=pe, fillcolor="rgba(245, 158, 11, 0.25)",
            line_color=PALETTE["warning"], line_width=1.5,
            annotation_text="Detected Anomaly" if pi == 0 else "",
            annotation_font_color=PALETTE["warning"], row=1, col=1
        )

    # Row 2: Error & Threshold
    if e_slice is not None:
        fig.add_trace(go.Scatter(
            x=t_slice, y=e_slice, name="EWMA Smoothed Error (e_s)",
            line=dict(color=PALETTE["warning"], width=1.2)
        ), row=2, col=1)
        if threshold:
            fig.add_hline(
                y=threshold, line_color=PALETTE["critical"], line_dash="dash", line_width=1.5,
                annotation_text=f"Dynamic Threshold = {threshold:.4f}",
                annotation_font_color=PALETTE["critical"], row=2, col=1
            )

    fig.update_layout(
        **{k: v for k, v in apply_plotly_theme(go.Figure(), height=620).layout.to_plotly_json().items()},
        height=620,
        title=f"Channel {channel_id} ({infer_mission(channel_id)}) — Telemetry Analysis Deck",
        legend=dict(orientation="h", y=-0.08, x=0.1),
        paper_bgcolor=PALETTE["card"],
        plot_bgcolor=PALETTE["bg2"],
    )
    fig.update_yaxes(title_text="Standardized Value", row=1, col=1, gridcolor=PALETTE["grid"])
    fig.update_yaxes(title_text="Error Deviation", row=2, col=1, gridcolor=PALETTE["grid"])
    fig.update_xaxes(title_text="Telemetry Timestep", gridcolor=PALETTE["grid"])

    st.plotly_chart(fig, use_container_width=True)

    # ---- METRICS ----
    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dynamic Threshold", f"{threshold:.5f}" if threshold else "—")
    with c2:
        st.metric("Expert Ground-Truth Events", len(gt_intervals))
    with c3:
        st.metric("Model Detected Sequences", len(pred_intervals))
    with c4:
        raw_err = np.array(result.get("raw_error", []))
        st.metric("Mean Residual Error", f"{np.mean(raw_err):.5f}" if len(raw_err) > 0 else "—")

else:
    # Simple telemetry plot
    gt_intervals = loader.get_channel_labels(channel_id) if split == "Test" else []
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_idx, y=tel_slice,
        name=f"{channel_id} Telemetry", line=dict(color=PALETTE["accent"], width=1.4)
    ))
    for gi, (gs, ge) in enumerate(gt_intervals):
        if ge >= start_idx and gs <= end_idx:
            fig.add_vrect(
                x0=max(gs, start_idx), x1=min(ge, end_idx),
                fillcolor="rgba(255, 51, 102, 0.22)", line_color=PALETTE["critical"], line_width=1,
                annotation_text="Ground Truth Anomaly" if gi == 0 else "",
                annotation_font_color=PALETTE["critical"],
            )
    fig = apply_plotly_theme(fig, f"Channel {channel_id} — {split} Telemetry Stream", height=420)
    fig.update_layout(xaxis_title="Time Step", yaxis_title="Telemetry Measurement (Col 0)")
    st.plotly_chart(fig, use_container_width=True)

    if split == "Test" and not detection_path.exists():
        st.info(f"ℹ️ Model has not yet run inference for **{channel_id}**. Checkpoints available: P-1, M-1.")

# ---- RAW DATA TABLE ----
with st.expander("📋 Inspect Raw Sensor Telemetry Array (First 100 observations)"):
    df_sample = pd.DataFrame(raw[:100], columns=[f"Feature_{i}" for i in range(raw.shape[1])])
    df_sample.index.name = "Timestep"
    st.dataframe(df_sample, use_container_width=True)
