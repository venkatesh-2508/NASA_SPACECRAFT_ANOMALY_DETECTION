"""
NASA Spacecraft Anomaly Detection
App Component: Metric Cards — Aerospace HUD Card Widgets with High Contrast
"""
import streamlit as st


def metric_card(label: str, value: str, delta: str = None, color: str = "#38BDF8"):
    delta_html = f'<div style="font-size: .72rem; color: #94A3B8; margin-top: 5px; font-family: \'JetBrains Mono\', monospace; font-weight: 500;">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div style="background: rgba(17, 28, 46, .88); border: 1px solid #243247; border-top: 2px solid {color};
                border-radius: 12px; padding: 1.05rem .9rem; text-align: left; box-shadow: 0 8px 22px rgba(0,0,0,.16);">
        <div style="font-size: .64rem; color: #94A3B8; text-transform: uppercase; letter-spacing: .08em;
                    font-family: 'JetBrains Mono', monospace; font-weight: 600; margin-bottom: .45rem;">
            {label}
        </div>
        <div style="font-size: 1.55rem; font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">
            {value}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def status_indicator(label: str, ok: bool):
    icon = "🟢" if ok else "🔴"
    color = "#10B981" if ok else "#FF3B69"
    st.markdown(f'<span style="color: {color}; font-weight: 700; font-family: \'JetBrains Mono\', monospace; font-size: 0.88rem;">{icon} {label}</span>', unsafe_allow_html=True)
