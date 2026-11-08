"""
Page 6: Dataset Status — Telemetry Archives & Validation Audit
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
st.set_page_config(page_title="Dataset Status | Orbital Intelligence", page_icon="💾", layout="wide")

import pandas as pd

from app.components.charts import inject_global_css, PALETTE
from app.components.sidebar import render_sidebar
from src.utils.config import load_config, get_path
from src.data.data_loader import DataLoader
from src.data.data_validation import DataValidator

inject_global_css()
render_sidebar()

cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
loader = DataLoader(str(get_path(cfg, "data_dir")))
validator = DataValidator(loader)

# --- Breadcrumb & Return to Entry Portal ---
col_back, col_crumbs = st.columns([2.2, 4])
with col_back:
    if st.button("🏠 ← Return to Home Entry Portal", key="btn_return_home"):
        st.switch_page("Home.py")
with col_crumbs:
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #94A3B8; padding-top: 8px;">
        <span style="color: #38BDF8;">Orbital Intelligence</span> &nbsp;›&nbsp; 
        <b style="color: #FFFFFF;">06 · Telemetry Dataset Archive</b>
    </div>
    """, unsafe_allow_html=True)

# --- Page Header Banner ---
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(6, 9, 19, 0.95) 0%, rgba(13, 24, 46, 0.85) 50%, rgba(10, 16, 32, 0.95) 100%);
            border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.6rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00F0FF; letter-spacing: 0.12em; text-transform: uppercase;">
        DATA INTEGRITY AUDIT · TELEMETRY ARCHIVE VALIDATION
    </div>
    <h1 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0 0.4rem 0; letter-spacing: -0.01em;">
        Telemetry Dataset Status
    </h1>
    <div style="font-size: 0.92rem; color: #CBD5E1; max-width: 820px; line-height: 1.5;">
        Verification audit for SMAP and MSL binary telemetry arrays, expert anomaly intervals, 
        and feature dimensions across the mission archive.
    </div>
</div>
""", unsafe_allow_html=True)

# ---- VALIDATION ----
with st.spinner("Validating telemetry repository..."):
    report = validator.validate_all()

# ---- STATUS BANNER ----
if report["passed"]:
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 10px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 1.4rem;">✅</span>
        <div>
            <div style="color: #10B981; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">
                DATASET VALIDATION AUDIT PASSED
            </div>
            <div style="color: #CBD5E1; font-size: 0.82rem;">
                All 81 telemetry binary channels verified with zero missing values or dimension mismatches.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("❌ Dataset validation encountered errors — see log details below.")

# ---- TOP METRICS ----
col1, col2, col3, col4 = st.columns(4)
ts = report.get("train_summary", {})
tst = report.get("test_summary", {})
ls = report.get("label_summary", {})
with col1:
    st.metric("Training Telemetry Files", ts.get("num_files", 0))
with col2:
    st.metric("Test Telemetry Files", tst.get("num_files", 0))
with col3:
    st.metric("Ground-Truth Labels", ls.get("num_rows", 0))
with col4:
    st.metric("Validation Errors", len(report.get("errors", [])))

st.markdown("<hr>", unsafe_allow_html=True)

# ---- PATHS ----
st.markdown("### 📁 Telemetry Repository Architecture")
data_dir = get_path(cfg, "data_dir")
st.code(f"""
Data Directory   : {data_dir}
├── train/       : {loader.train_dir} (81 .npy channel arrays)
├── test/        : {loader.test_dir} (81 .npy channel arrays)
└── labels       : {loader.labels_file} (105 expert-labeled anomaly intervals)
""", language="text")

# ---- LABEL SUMMARY ----
if ls:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🏷️ Mission Anomaly Label Distribution")
    col1, col2 = st.columns(2)
    with col1:
        if "spacecraft_counts" in ls:
            st.markdown("**Incidents by Spacecraft Mission**")
            df_sc = pd.DataFrame(ls["spacecraft_counts"].items(), columns=["Mission", "Incident Count"])
            st.dataframe(df_sc, use_container_width=True)
    with col2:
        if "anomaly_class_counts" in ls:
            st.markdown("**Incidents by Anomaly Classification**")
            df_cls = pd.DataFrame(ls["anomaly_class_counts"].items(), columns=["Class", "Incident Count"])
            st.dataframe(df_cls, use_container_width=True)

# ---- CHANNEL TABLE ----
if report.get("channel_details"):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📡 Complete Channel Dimensions & Integrity Audit")
    details = report["channel_details"]
    df = pd.DataFrame(details)
    if not df.empty:
        for col in ["train_shape", "test_shape"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x) if x else "—")
        df["nan_train"] = df["nan_train"].apply(lambda x: "⚠️ Yes" if x else "✅ No")
        df["nan_test"] = df["nan_test"].apply(lambda x: "⚠️ Yes" if x else "✅ No")
        st.dataframe(df, use_container_width=True, height=380)

# ---- ERRORS / WARNINGS ----
if report.get("errors"):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ❌ Validation Anomalies")
    for e in report["errors"]:
        st.error(e)
