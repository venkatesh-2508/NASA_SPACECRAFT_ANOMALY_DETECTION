"""Shared visual system and Plotly helpers for Orbital Intelligence."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st


PALETTE = {
    "bg": "#080D18", "bg2": "#0B1220", "card": "#111C2E", "card_border": "#243247",
    "accent": "#38BDF8", "accent2": "#8B5CF6", "success": "#22C55E", "warning": "#F59E0B",
    "critical": "#EF4444", "text": "#F8FAFC", "text2": "#CBD5E1", "text_muted": "#94A3B8",
    "grid": "rgba(148, 163, 184, .16)",
}

PLOTLY_TEMPLATE = dict(layout=go.Layout(
    paper_bgcolor="rgba(17, 28, 46, .96)", plot_bgcolor="rgba(8, 13, 24, .7)",
    font=dict(color=PALETTE["text"], family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=PALETTE["grid"], linecolor=PALETTE["card_border"], zerolinecolor=PALETTE["grid"], tickfont=dict(color=PALETTE["text_muted"]), title_font=dict(color=PALETTE["text2"])),
    yaxis=dict(gridcolor=PALETTE["grid"], linecolor=PALETTE["card_border"], zerolinecolor=PALETTE["grid"], tickfont=dict(color=PALETTE["text_muted"]), title_font=dict(color=PALETTE["text2"])),
    legend=dict(bgcolor="rgba(8, 13, 24, .78)", bordercolor=PALETTE["card_border"], borderwidth=1, font=dict(color=PALETTE["text2"], size=11)),
    margin=dict(l=52, r=24, t=54, b=46),
))


def inject_global_css():
    """Install the responsive aerospace design system on each Streamlit page."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    :root {{--void:{PALETTE['bg']};--sidebar:{PALETTE['bg2']};--panel:{PALETTE['card']};--line:{PALETTE['card_border']};--cyan:{PALETTE['accent']};--violet:{PALETTE['accent2']};--green:{PALETTE['success']};--amber:{PALETTE['warning']};--red:{PALETTE['critical']};--ink:{PALETTE['text']};--muted:{PALETTE['text_muted']};}}
    .stApp {{color:var(--ink);font-family:Inter,sans-serif;background-color:var(--void);background-image:radial-gradient(circle at 72% -12%,rgba(56,189,248,.12),transparent 31rem),radial-gradient(circle at 12% 92%,rgba(139,92,246,.10),transparent 25rem);}}
    .main .block-container {{max-width:1520px;padding-top:1.7rem;padding-bottom:3.5rem;}} #MainMenu,footer {{visibility:hidden;}}
    h1,h2,h3,h4 {{font-family:'Space Grotesk',Inter,sans-serif!important;color:var(--ink)!important;}} h2 {{font-size:1.28rem!important;margin-top:1.6rem!important;}} p,li {{color:{PALETTE['text2']};}}
    [data-testid='stSidebar'] {{background:linear-gradient(180deg,#0B1220,#080D18 72%)!important;border-right:1px solid var(--line);}} [data-testid='stSidebar']>div:first-child {{padding-top:.75rem;}}
    [data-testid='stSidebar'] [data-testid='stPageLink-NavLink'] {{border:1px solid transparent;border-radius:9px;margin:3px 0;padding:.57rem .68rem;transition:background .18s ease,border-color .18s ease,transform .18s ease;}}
    [data-testid='stSidebar'] [data-testid='stPageLink-NavLink']:hover {{background:rgba(56,189,248,.09);border-color:rgba(56,189,248,.27);transform:translateX(2px);}}
    [data-testid='stSidebar'] [data-testid='stPageLink-NavLink'] p {{color:#CBD5E1!important;font-size:.83rem!important;font-weight:600!important;}} [data-testid='stSidebar'] [aria-current='page'] {{background:rgba(56,189,248,.11)!important;border-left:2px solid var(--cyan)!important;}} [data-testid='stSidebar'] [aria-current='page'] p {{color:#F8FAFC!important;}} [data-testid='stSidebarNav'] {{display:none;}}
    .dashboard-header {{background:linear-gradient(118deg,rgba(17,28,46,.96),rgba(17,28,46,.65));border:1px solid var(--line);border-radius:16px;padding:1.45rem 1.6rem;margin-bottom:1.25rem;box-shadow:0 16px 42px rgba(0,0,0,.2);position:relative;overflow:hidden;}}
    .dashboard-header:after {{content:'';position:absolute;right:-60px;top:-80px;width:210px;height:210px;border-radius:50%;border:1px solid rgba(56,189,248,.17);box-shadow:0 0 0 34px rgba(56,189,248,.035),0 0 0 68px rgba(56,189,248,.022);}} .eyebrow {{font-family:'JetBrains Mono',monospace;color:var(--cyan);font-size:.68rem;font-weight:600;letter-spacing:.13em;text-transform:uppercase;}}
    .hud-card,.info-card {{background:rgba(17,28,46,.88);border:1px solid var(--line);border-radius:13px;padding:1.2rem;box-shadow:0 10px 28px rgba(0,0,0,.16);transition:border-color .2s ease,transform .2s ease,box-shadow .2s ease;}} .hud-card:hover,.info-card:hover {{border-color:rgba(56,189,248,.48);transform:translateY(-2px);box-shadow:0 14px 34px rgba(0,0,0,.24);}}
    [data-testid='metric-container'] {{background:rgba(17,28,46,.88);border:1px solid var(--line);border-radius:12px;padding:1rem 1.05rem;box-shadow:0 8px 22px rgba(0,0,0,.16);transition:transform .2s ease,border-color .2s ease;}} [data-testid='metric-container']:hover {{transform:translateY(-2px);border-color:rgba(56,189,248,.48);}}
    [data-testid='stMetricLabel'] {{font-family:'JetBrains Mono',monospace;font-size:.67rem!important;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)!important;}} [data-testid='stMetricValue'] {{font-family:'Space Grotesk',sans-serif;color:var(--ink)!important;font-size:1.55rem!important;}} [data-testid='stMetricDelta'] {{font-size:.72rem!important;}}
    hr {{border:0;height:1px;background:linear-gradient(90deg,transparent,var(--line),transparent);margin:1.7rem 0;}}
    .stButton>button {{min-height:2.65rem;border:1px solid rgba(125,211,252,.45);border-radius:9px;background:linear-gradient(135deg,#38BDF8,#0EA5E9);color:#06111f;font-weight:800;letter-spacing:.025em;box-shadow:0 8px 22px rgba(14,165,233,.2);transition:transform .18s ease,box-shadow .18s ease,filter .18s ease;}} .stButton>button:hover {{transform:translateY(-1px);filter:brightness(1.08);box-shadow:0 12px 27px rgba(14,165,233,.3);}} .stButton>button:active {{transform:translateY(0) scale(.985);}}
    [data-testid='stSelectbox']>div>div,[data-testid='stMultiSelect']>div>div {{background:#0B1220!important;border-color:var(--line)!important;color:var(--ink)!important;border-radius:8px!important;}} [data-testid='stDataFrame'] {{border:1px solid var(--line);border-radius:11px;overflow:hidden;}} [data-testid='stPlotlyChart'] {{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:rgba(17,28,46,.6);}} [data-testid='stExpander'] {{border:1px solid var(--line);border-radius:10px;background:rgba(17,28,46,.6);}} .stDownloadButton>button {{border-radius:9px;border-color:var(--line);color:var(--text);}}
    @keyframes rise-in {{from {{opacity:0;transform:translateY(10px)}}to {{opacity:1;transform:translateY(0)}}}} .dashboard-header,[data-testid='metric-container'] {{animation:rise-in .45s ease both;}}
    @media(max-width:780px) {{.main .block-container {{padding:1rem .8rem 2.2rem;}}.dashboard-header {{padding:1.15rem;border-radius:12px;}}[data-testid='metric-container'] {{padding:.8rem;}}}} @media(prefers-reduced-motion:reduce) {{*,*:before,*:after {{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important;}}}}
    </style>""", unsafe_allow_html=True)


def apply_plotly_theme(fig: go.Figure, title: str | None = None, height: int = 400) -> go.Figure:
    """Apply the consistent dark, high-contrast analytical chart theme."""
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=height, title=dict(text=f"<b>{title}</b>" if title else None, font=dict(size=14, color=PALETTE["text"])))
    return fig


def telemetry_chart(actual: np.ndarray, predicted: np.ndarray | None = None, smoothed_error: np.ndarray | None = None, threshold: float | None = None, gt_intervals: list | None = None, pred_intervals: list | None = None, title: str = "Spacecraft telemetry waveform", height: int = 500) -> go.Figure:
    """Create a real telemetry chart with optional model and event overlays."""
    fig = go.Figure(); steps = np.arange(len(actual))
    fig.add_trace(go.Scatter(x=steps, y=actual, name="Actual telemetry", line=dict(color=PALETTE["accent"], width=1.7)))
    if predicted is not None: fig.add_trace(go.Scatter(x=steps, y=predicted, name="LSTM forecast", line=dict(color=PALETTE["accent2"], width=1.5, dash="dot")))
    for index, (start, end) in enumerate(gt_intervals or []): fig.add_vrect(x0=start, x1=end, fillcolor="rgba(239,68,68,.17)", line_color=PALETTE["critical"], line_width=1, annotation_text="Ground truth" if index == 0 else "", annotation_font_color=PALETTE["critical"])
    for index, (start, end) in enumerate(pred_intervals or []): fig.add_vrect(x0=start, x1=end, fillcolor="rgba(245,158,11,.16)", line_color=PALETTE["warning"], line_width=1, annotation_text="Detected" if index == 0 else "", annotation_font_color=PALETTE["warning"])
    return apply_plotly_theme(fig, title=title, height=height)
