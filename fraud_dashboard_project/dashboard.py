import json
import os
import sys
import html as html_lib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# DECISION_OVERRIDE_V1

# make the bundled fraudcore package importable regardless of the working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

# PROJECTOR_FIX_V36
# ============================================================
# Hybrid (AE + CatBoost) + RL Fraud Dashboard
# Output mode: reads finished dashboard output files only.
# ============================================================

st.set_page_config(
    page_title="Hybrid + RL Fraud Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling -----------------------------
st.markdown(
    """
<style>
:root {
    --bg: #ffffff;
    --panel: #ffffff;
    --panel-soft: #f8fafc;
    --sidebar: #f3f4f6;
    --text: #0f172a;
    --muted: #334155;
    --muted-2: #475569;
    --border: #cbd5e1;
    --grid: #cbd5e1;
    --blue: #2563eb;
    --blue-hover: #1d4ed8;
    --blue-soft: #eff6ff;
    --orange-soft: #fff7ed;
}

/* Keep Streamlit header visible enough so the sidebar open/close button is usable */
[data-testid="stHeader"] {
    background: #ffffff !important;
    height: 3.2rem !important;
    visibility: visible !important;
    border-bottom: 1px solid #e5e7eb !important;
    z-index: 999998 !important;
}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    visibility: hidden !important;
    height: 0rem !important;
}

/* Global light base */
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1580px !important;
}
html, body, p, li, label, span, div, h1, h2, h3, h4, h5, h6, .stMarkdown, .stCaption {
    color: var(--text) !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
}
html, body, [class*="css"] { font-size: 18px !important; }
h1 { font-size: 2.45rem !important; font-weight: 850 !important; line-height: 1.15 !important; }
h2 { font-size: 1.85rem !important; font-weight: 850 !important; line-height: 1.2 !important; }
h3 { font-size: 1.45rem !important; font-weight: 800 !important; line-height: 1.25 !important; }
p, li, label, .stMarkdown, .stCaption { font-size: 1.03rem !important; line-height: 1.55 !important; }

/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: var(--sidebar) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text) !important;
}

/* Metric cards */
.metric-card {
    background-color: var(--panel-soft) !important;
    border: 2px solid var(--border) !important;
    border-radius: 1rem !important;
    padding: 1rem 1.1rem !important;
    min-height: 118px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.08) !important;
    overflow: hidden !important;
}
.metric-title {
    color: var(--muted) !important;
    font-size: 0.9rem !important;
    text-transform: uppercase !important;
    letter-spacing: .035em !important;
    font-weight: 850 !important;
    line-height: 1.35 !important;
}
.metric-value {
    color: var(--text) !important;
    font-size: clamp(1.35rem, 2.2vw, 1.85rem) !important;
    font-weight: 900 !important;
    margin-top: .35rem !important;
    line-height: 1.15 !important;
    white-space: nowrap !important;
}
.metric-sub {
    color: var(--muted-2) !important;
    font-size: 0.9rem !important;
    margin-top: .3rem !important;
    line-height: 1.4 !important;
}

/* Notes */
.section-note {
    background-color: var(--blue-soft) !important;
    border: 2px solid #bfdbfe !important;
    border-radius: .9rem !important;
    padding: 1rem 1.2rem !important;
    color: var(--text) !important;
    font-size: 1.04rem !important;
    line-height: 1.55 !important;
}
.section-note * { color: var(--text) !important; }
.warning-note {
    background-color: var(--orange-soft) !important;
    border: 2px solid #fed7aa !important;
    border-radius: .9rem !important;
    padding: 1rem 1.2rem !important;
    color: #7c2d12 !important;
    font-size: 1.04rem !important;
    line-height: 1.55 !important;
}
.warning-note * { color: #7c2d12 !important; }
.small-muted { color: var(--muted-2) !important; font-size: 0.95rem !important; }

/* Very visible tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background: #e5e7eb !important;
    padding: 0.5rem !important;
    border-radius: 1rem !important;
    margin: 0.6rem 0 1.2rem 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: #ffffff !important;
    border: 2px solid var(--border) !important;
    border-radius: 0.8rem !important;
    color: var(--text) !important;
    font-size: 1.1rem !important;
    font-weight: 850 !important;
    padding: 0.65rem 1.1rem !important;
    white-space: nowrap !important;
}
.stTabs [aria-selected="true"] {
    background: var(--blue) !important;
    color: #ffffff !important;
    border-color: var(--blue) !important;
}
.stTabs [aria-selected="true"] * { color: #ffffff !important; }

/* Buttons */
div.stButton > button, div[data-testid="stDownloadButton"] button, .stDownloadButton button, button[kind="primary"], button[kind="secondary"] {
    background-color: var(--blue) !important;
    color: #ffffff !important;
    border: 2px solid var(--blue-hover) !important;
    border-radius: 0.7rem !important;
    font-size: 1rem !important;
    font-weight: 850 !important;
    min-height: 3rem !important;
    white-space: nowrap !important;
}
div.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover, .stDownloadButton button:hover {
    background-color: var(--blue-hover) !important;
    color: #ffffff !important;
}
div.stButton > button *, div[data-testid="stDownloadButton"] button *, .stDownloadButton button *, button p, button span {
    color: #ffffff !important;
}
button:disabled, button[disabled] {
    background-color: #94a3b8 !important;
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Inputs/selects/textarea/number input */
input, textarea {
    background-color: #ffffff !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    caret-color: var(--text) !important;
}
input::placeholder, textarea::placeholder { color: #64748b !important; opacity: 1 !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"], [data-baseweb="textarea"] {
    background-color: #ffffff !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-baseweb="input"] *, [data-baseweb="base-input"] *, [data-baseweb="select"] *, [data-baseweb="textarea"] * {
    background-color: #ffffff !important;
    color: var(--text) !important;
    opacity: 1 !important;
}
[data-testid="stNumberInput"] button {
    background-color: #e2e8f0 !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-testid="stNumberInput"] button * { color: var(--text) !important; }
.stRadio *, .stSelectbox *, .stTextInput *, .stNumberInput *, .stTextArea *, .stFileUploader * {
    color: var(--text) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 0.75rem !important;
}
[data-testid="stExpander"] * { color: var(--text) !important; }

/* Custom HTML tables: light and readable */
.table-wrap {
    width: 100%;
    max-height: 520px;
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: 0.8rem;
    background: #ffffff;
    margin: 0.4rem 0 1rem 0;
}
table.clean-table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff !important;
    color: var(--text) !important;
    font-size: 0.98rem !important;
}
table.clean-table th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: #e2e8f0 !important;
    color: var(--text) !important;
    font-weight: 850 !important;
    border-bottom: 2px solid var(--border) !important;
    padding: 0.7rem 0.75rem !important;
    text-align: left !important;
    white-space: nowrap !important;
}
table.clean-table td {
    background: #ffffff !important;
    color: var(--text) !important;
    border-bottom: 1px solid #e5e7eb !important;
    padding: 0.62rem 0.75rem !important;
    white-space: nowrap !important;
}
table.clean-table tr:nth-child(even) td { background: #f8fafc !important; }
table.clean-table tr:hover td { background: #eef2ff !important; }

/* Risk badges for bank-operations queues */
.risk-tag {
    display: inline-block !important;
    padding: 0.25rem 0.65rem !important;
    border-radius: 999px !important;
    font-size: 0.86rem !important;
    font-weight: 900 !important;
    white-space: nowrap !important;
    border: 1px solid transparent !important;
}
.risk-low { background: #dcfce7 !important; color: #14532d !important; border-color: #86efac !important; }
.risk-medium { background: #fef3c7 !important; color: #78350f !important; border-color: #fcd34d !important; }
.risk-high { background: #fee2e2 !important; color: #7f1d1d !important; border-color: #fca5a5 !important; }

/* Make any remaining Streamlit dataframes less jarring */
[data-testid="stDataFrame"], [data-testid="stTable"], .stDataFrame {
    background-color: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0.65rem !important;
}

/* Alerts and code */
[data-testid="stAlert"] {
    background-color: #fff7ed !important;
    color: var(--text) !important;
    border: 1px solid #fed7aa !important;
}
[data-testid="stAlert"] * { color: var(--text) !important; }
code, pre {
    background: #e2e8f0 !important;
    color: var(--text) !important;
    border-radius: 0.35rem !important;
}

/* Plot containers */
.js-plotly-plot, .plotly, .plot-container {
    background: #ffffff !important;
}
.modebar, .modebar-group { background: rgba(255,255,255,0.92) !important; }


/* PROJECTOR_FIX_V22: make collapsed sidebar button and Plotly controls readable/hidden */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[title="Open sidebar"],
button[title="Close sidebar"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 2px solid #1d4ed8 !important;
    border-radius: 999px !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="stSidebarCollapseButton"] *,
button[title="Open sidebar"] *,
button[title="Close sidebar"] * {
    color: #ffffff !important;
    stroke: #ffffff !important;
    fill: #ffffff !important;
    opacity: 1 !important;
}
.modebar, .modebar-container {
    display: none !important;
    visibility: hidden !important;
}
.js-plotly-plot .plotly .modebar {
    display: none !important;
}


/* PROJECTOR_FIX_V22: force sidebar open/close control to be visible on projector */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[title="Open sidebar"],
button[title="Close sidebar"] {
    position: fixed !important;
    top: 0.45rem !important;
    left: 0.55rem !important;
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border: 3px solid #ffffff !important;
    border-radius: 999px !important;
    width: 48px !important;
    height: 48px !important;
    min-width: 48px !important;
    min-height: 48px !important;
    opacity: 1 !important;
    visibility: visible !important;
    z-index: 2147483647 !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.35) !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
button[title="Open sidebar"] svg,
button[title="Close sidebar"] svg {
    stroke: #ffffff !important;
    fill: #ffffff !important;
    color: #ffffff !important;
    opacity: 1 !important;
}


/* PROJECTOR_FIX_V22: make Streamlit sidebar reopen control visible after sidebar is collapsed */
[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.98) !important;
    border-bottom: 1px solid #cbd5e1 !important;
    height: 56px !important;
    min-height: 56px !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 2147483600 !important;
    pointer-events: none !important;
}

/* Streamlit names this control differently across versions, so target all known variants */
[data-testid="stHeader"] button,
[data-testid="stHeader"] [role="button"],
[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stBaseButton-header"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label*="sidebar" i],
button[title*="sidebar" i],
button[aria-label*="menu" i],
button[title*="menu" i] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    background: #2563eb !important;
    color: #ffffff !important;
    border: 3px solid #ffffff !important;
    border-radius: 999px !important;
    width: 52px !important;
    height: 52px !important;
    min-width: 52px !important;
    min-height: 52px !important;
    padding: 0 !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.28) !important;
    z-index: 2147483647 !important;
}

/* Force the icon itself to be visible */
[data-testid="stHeader"] button svg,
[data-testid="stHeader"] [role="button"] svg,
[data-testid="stBaseButton-headerNoPadding"] svg,
[data-testid="stBaseButton-header"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
button[aria-label*="sidebar" i] svg,
button[title*="sidebar" i] svg,
button[aria-label*="menu" i] svg,
button[title*="menu" i] svg {
    color: #ffffff !important;
    stroke: #ffffff !important;
    fill: #ffffff !important;
    opacity: 1 !important;
    width: 30px !important;
    height: 30px !important;
}

/* Put the collapsed-open control in an obvious place. This affects the header button only. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label*="Open sidebar" i],
button[title*="Open sidebar" i],
button[aria-label*="Show sidebar" i],
button[title*="Show sidebar" i] {
    position: fixed !important;
    top: 12px !important;
    left: 14px !important;
}

/* Add a visible hamburger mark if Streamlit's SVG becomes hidden by the browser/theme */
[data-testid="collapsedControl"]::after,
[data-testid="stSidebarCollapsedControl"]::after,
button[aria-label*="Open sidebar" i]::after,
button[title*="Open sidebar" i]::after,
button[aria-label*="Show sidebar" i]::after,
button[title*="Show sidebar" i]::after {
    content: "☰" !important;
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------- Last-pass projector/readability overrides -----------------------------
st.markdown(
    """
<style>
/* FINAL VISIBILITY OVERRIDE: keeps controls readable on white projector theme */
:root {
  --final-text: #0f172a;
  --final-muted: #334155;
  --final-border: #94a3b8;
  --final-blue: #1d4ed8;
  --final-blue-dark: #1e40af;
}

/* Prevent awkward single-word breaks in cards/buttons/tabs */
.metric-card, .metric-card *, .stButton button, .stDownloadButton button,
.stTabs [data-baseweb="tab"], [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
  word-break: keep-all !important;
  overflow-wrap: normal !important;
  hyphens: none !important;
}
.metric-title { white-space: nowrap !important; font-size: 0.84rem !important; }
.metric-value { white-space: nowrap !important; font-size: clamp(1.35rem, 1.9vw, 1.8rem) !important; }
.metric-sub { word-break: normal !important; overflow-wrap: normal !important; }

/* Buttons: blue background + white text, including nested Streamlit markdown paragraphs */
div.stButton > button, div[data-testid="stDownloadButton"] button, .stDownloadButton button, button {
  background: var(--final-blue) !important;
  color: #ffffff !important;
  border: 2px solid var(--final-blue-dark) !important;
  border-radius: 0.7rem !important;
  min-height: 3.1rem !important;
  font-weight: 850 !important;
  opacity: 1 !important;
}
div.stButton > button *, div[data-testid="stDownloadButton"] button *, .stDownloadButton button *,
button *, button p, button span, button div {
  color: #ffffff !important;
  opacity: 1 !important;
}
div.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover, .stDownloadButton button:hover {
  background: var(--final-blue-dark) !important;
  color: #ffffff !important;
}

/* Inputs/selects/textareas/number inputs: white box + dark readable value */
input, textarea, select,
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"],
[data-baseweb="input"] > div, [data-baseweb="base-input"] > div, [data-baseweb="textarea"] > div, [data-baseweb="select"] > div {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: var(--final-text) !important;
  border-color: var(--final-border) !important;
  opacity: 1 !important;
}
input *, textarea *, select *,
[data-testid="stTextInput"] *, [data-testid="stNumberInput"] *, [data-testid="stTextArea"] *, [data-testid="stSelectbox"] *,
[data-baseweb="input"] *, [data-baseweb="base-input"] *, [data-baseweb="textarea"] *, [data-baseweb="select"] * {
  color: var(--final-text) !important;
  background-color: transparent !important;
  opacity: 1 !important;
}
input::placeholder, textarea::placeholder { color: #64748b !important; opacity: 1 !important; }
[data-testid="stNumberInput"] button, [data-testid="stNumberInput"] button * {
  background: #e2e8f0 !important;
  color: var(--final-text) !important;
  border-color: var(--final-border) !important;
}

/* Sidebar text and controls */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background: #f3f4f6 !important;
  color: var(--final-text) !important;
}
[data-testid="stSidebar"] * { color: var(--final-text) !important; opacity: 1 !important; }
[data-testid="stSidebar"] input { background: #ffffff !important; color: var(--final-text) !important; }

/* Light readable custom tables */
.table-wrap, table.clean-table, table.clean-table th, table.clean-table td {
  color: var(--final-text) !important;
}
table.clean-table th { background: #dbeafe !important; color: var(--final-text) !important; }
table.clean-table td { background: #ffffff !important; color: var(--final-text) !important; }
table.clean-table tr:nth-child(even) td { background: #f8fafc !important; }

/* Plotly text should never be pale on white */
.js-plotly-plot text, .plotly text, .svg-container text {
  fill: var(--final-text) !important;
  color: var(--final-text) !important;
  opacity: 1 !important;
}

/* Clear view selector */
.stTabs [data-baseweb="tab-list"] {
  background: #e2e8f0 !important;
  border: 2px solid #cbd5e1 !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--final-text) !important;
  font-size: 1.05rem !important;
  min-width: 260px !important;
}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * {
  background: var(--final-blue) !important;
  color: #ffffff !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------- Emergency projector CSS patch v2 -----------------------------
st.markdown(
    """
<style>
/* PROJECTOR_FIX_V18 — high contrast controls and tabs */
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] div {
  color: #0f172a !important;
  opacity: 1 !important;
  visibility: visible !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div {
  color: #ffffff !important;
}
.stTabs [data-baseweb="tab"] {
  min-width: 300px !important;
  height: auto !important;
}

/* Make all download/submit buttons readable */
div[data-testid="stDownloadButton"] button,
div.stButton > button {
  background: #2563eb !important;
  border: 2px solid #1d4ed8 !important;
  color: #ffffff !important;
  opacity: 1 !important;
}
div[data-testid="stDownloadButton"] button p,
div[data-testid="stDownloadButton"] button span,
div.stButton > button p,
div.stButton > button span {
  color: #ffffff !important;
  opacity: 1 !important;
}

/* Keep sidebar controls readable */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="textarea"] {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 2px solid #94a3b8 !important;
  opacity: 1 !important;
}
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
  color: #475569 !important;
  opacity: 1 !important;
}

/* Strong chart text contrast */
.js-plotly-plot text,
.plotly text,
.svg-container text,
.gtitle, .xtitle, .ytitle, .legend text {
  fill: #0f172a !important;
  opacity: 1 !important;
}

/* Better table readability */
.table-wrap, table.clean-table, table.clean-table th, table.clean-table td {
  background: #ffffff !important;
  color: #0f172a !important;
}
table.clean-table th { background: #dbeafe !important; }
table.clean-table tr:nth-child(even) td { background: #f8fafc !important; }

/* Summary tables: no horizontal slider, all content visible */
.summary-table-wrap {
  width: 100%;
  overflow: visible !important;
  border: 1px solid var(--final-border);
  border-radius: 0.8rem;
  background: #ffffff;
  margin: 0.4rem 0 1rem 0;
}
table.summary-table {
  width: 100% !important;
  table-layout: fixed !important;
}
table.summary-table th,
table.summary-table td {
  white-space: normal !important;
  overflow-wrap: normal !important;
  word-break: normal !important;
  hyphens: none !important;
  vertical-align: top !important;
}
table.summary-table th:first-child,
table.summary-table td:first-child {
  width: 28% !important;
}
table.summary-table th:nth-child(2),
table.summary-table td:nth-child(2) {
  width: 72% !important;
}

/* No awkward word splitting */
.metric-card, .metric-card *, h1, h2, h3, button, button *, .stTabs * {
  word-break: keep-all !important;
  overflow-wrap: normal !important;
  hyphens: none !important;
}
.metric-title { white-space: nowrap !important; }
.metric-value { white-space: nowrap !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------- Hide Streamlit deploy/menu buttons -----------------------------
st.markdown(
    """
<style>
/* Hide only the Streamlit top-right Deploy and three-dot Main menu buttons.
   IMPORTANT: do NOT hide stToolbar/stToolbarActions, because Streamlit also puts
   the sidebar open button there when the sidebar is collapsed. */
[data-testid="stHeader"] [data-testid="stDeployButton"],
[data-testid="stHeader"] [data-testid="stMainMenu"],
[data-testid="stHeader"] button[title="Deploy"],
[data-testid="stHeader"] button[aria-label="Deploy"],
[data-testid="stHeader"] button[title="Main menu"],
[data-testid="stHeader"] button[aria-label="Main menu"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------- Sidebar visibility fix -----------------------------
st.markdown(
    """
<style>
/* PROJECTOR_FIX_V35: keep sidebar readable WITHOUT forcing it permanently open.
   The previous fix forced display/transform/width, which stopped the sidebar from closing. */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: #f3f4f6 !important;
    color: #0f172a !important;
    border-right: 2px solid #cbd5e1 !important;
}

[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"] * {
    color: #0f172a !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="textarea"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 2px solid #94a3b8 !important;
    border-radius: 0.55rem !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #0f172a !important;
}

/* Keep the native sidebar open/close button visible and clickable. */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[aria-label*="sidebar" i],
button[title*="sidebar" i] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    background: #2563eb !important;
    color: #ffffff !important;
    border: 3px solid #ffffff !important;
    border-radius: 999px !important;
    min-width: 48px !important;
    min-height: 48px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.28) !important;
    z-index: 2147483647 !important;
}

[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
button[aria-label*="sidebar" i] svg,
button[title*="sidebar" i] svg {
    color: #ffffff !important;
    stroke: #ffffff !important;
    fill: #ffffff !important;
    opacity: 1 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------- Sidebar button restore patch -----------------------------
st.markdown(
    """
<style>
/* PROJECTOR_FIX_V36: restore the native Streamlit sidebar open/close button.
   This keeps the top-right Deploy/Menu hidden but leaves the toolbar container alive. */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stToolbarActions"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stHeader"] {
    pointer-events: none !important;
}

[data-testid="stHeader"] button,
[data-testid="stHeader"] [role="button"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label*="sidebar" i],
button[title*="sidebar" i] {
    pointer-events: auto !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label*="Open sidebar" i],
button[title*="Open sidebar" i],
button[aria-label*="Close sidebar" i],
button[title*="Close sidebar" i],
button[aria-label*="sidebar" i],
button[title*="sidebar" i] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: #2563eb !important;
    color: #ffffff !important;
    border: 3px solid #ffffff !important;
    border-radius: 999px !important;
    width: 50px !important;
    height: 50px !important;
    min-width: 50px !important;
    min-height: 50px !important;
    padding: 0 !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.28) !important;
    z-index: 2147483647 !important;
}

[data-testid="collapsedControl"]::after,
[data-testid="stSidebarCollapsedControl"]::after,
button[aria-label*="Open sidebar" i]::after,
button[title*="Open sidebar" i]::after {
    content: "☰" !important;
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
}

[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
button[aria-label*="sidebar" i] svg,
button[title*="sidebar" i] svg {
    color: #ffffff !important;
    stroke: #ffffff !important;
    fill: #ffffff !important;
    opacity: 1 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOT_BG = "#ffffff"
PAPER_BG = "#ffffff"
FONT_COL = "#0f172a"
GRID_COL = "#cbd5e1"

CHART_COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]
ACTION_ORDER = ["accept", "review", "reject"]
STATUS_ORDER = ["approved", "pending_review", "blocked"]
ACTION_COLORS = {
    "accept": "#22c55e",
    "review": "#f59e0b",
    "reject": "#ef4444",
    "approved": "#22c55e",
    "pending_review": "#f59e0b",
    "blocked": "#ef4444",
}
COST_COLORS = {
    "Missed fraud": "#dc2626",
    "False rejection": "#f59e0b",
    "Manual review": "#2563eb",
    "Overload": "#7c3aed",
}
FRAUD_RESULT_COLORS = {
    "Detected fraud": "#16a34a",
    "Missed fraud": "#dc2626",
}
GROUND_TRUTH_COLUMNS = [
    "isFraud",
    "missed_fraud_cost",
    "false_rejection_cost",
    "manual_review_cost",
    "overload_cost",
    "transaction_cost",
    "protected_exposure",
]

# ----------------------------- Helpers -----------------------------
def fmt_money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "N/A"


def fmt_money_compact(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "N/A"
    sign = "-" if x < 0 else ""
    x_abs = abs(x)
    if x_abs >= 1_000_000:
        return f"{sign}${x_abs/1_000_000:.2f}M"
    if x_abs >= 1_000:
        return f"{sign}${x_abs/1_000:.1f}K"
    return f"{sign}${x_abs:,.0f}"


def fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "N/A"


def fmt_num(x: Any) -> str:
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "N/A"


def metric_card(title: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">{title}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COL, size=18),
        title=dict(font=dict(size=24, color=FONT_COL), x=0.02, xanchor="left", y=0.98, yanchor="top"),
        margin=dict(l=78, r=45, t=150, b=105),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=17, color=FONT_COL),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(255,255,255,0)",
            borderwidth=0,
            title_text="",
        ),
    )
    fig.update_xaxes(
        color=FONT_COL,
        gridcolor=GRID_COL,
        zerolinecolor=GRID_COL,
        linecolor=GRID_COL,
        tickfont=dict(size=16, color=FONT_COL),
        title_font=dict(size=18, color=FONT_COL),
        automargin=True,
    )
    fig.update_yaxes(
        color=FONT_COL,
        gridcolor=GRID_COL,
        zerolinecolor=GRID_COL,
        linecolor=GRID_COL,
        tickfont=dict(size=16, color=FONT_COL),
        title_font=dict(size=18, color=FONT_COL),
        automargin=True,
    )
    fig.update_traces(textfont=dict(color=FONT_COL, size=16))

    # Keep outside bar labels visible. Do this only for bar traces because
    # heatmaps do not support cliponaxis.
    bar_values = []
    for trace in fig.data:
        if getattr(trace, "type", "") == "bar" and getattr(trace, "orientation", None) != "h":
            try:
                vals = pd.to_numeric(pd.Series(trace.y), errors="coerce").dropna().astype(float).tolist()
                bar_values.extend(vals)
                trace.update(cliponaxis=False)
            except Exception:
                pass

    if bar_values:
        ymin = min(bar_values + [0.0])
        ymax = max(bar_values + [0.0])
        span = max(ymax - ymin, abs(ymax), 1.0)
        pad_top = span * 0.22
        pad_bottom = span * 0.08 if ymin < 0 else 0.0
        fig.update_yaxes(range=[ymin - pad_bottom, ymax + pad_top])

    fig.update_layout(legend_title_text="", legend_title=dict(text=""))
    for trace in fig.data:
        try:
            trace.update(legendgrouptitle_text="")
        except Exception:
            pass
    return fig


def show_fig(fig: go.Figure) -> None:
    """Render Plotly charts without modebar overlays and with consistent projector styling."""
    st.plotly_chart(
        style_fig(fig),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def render_table(df: pd.DataFrame, max_rows: Optional[int] = 500, height_px: int = 520) -> None:
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    shown = df.copy()
    note = ""
    if max_rows is not None and len(shown) > max_rows:
        shown = shown.head(max_rows)
        note = f"<div class='small-muted'>Showing first {max_rows:,} rows out of {len(df):,}. Use export for the full file.</div>"
    html = shown.to_html(index=False, classes="clean-table", border=0, escape=True)
    st.markdown(f"{note}<div class='table-wrap' style='max-height:{height_px}px'>{html}</div>", unsafe_allow_html=True)


def render_table_no_scroll(df: pd.DataFrame) -> None:
    """Render a small table fully visible without horizontal or vertical sliders."""
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    html = df.to_html(index=False, classes="clean-table summary-table", border=0, escape=True)
    st.markdown(f"<div class='summary-table-wrap'>{html}</div>", unsafe_allow_html=True)


def render_table_with_html(df: pd.DataFrame, html_cols: Optional[list] = None, max_rows: Optional[int] = 500, height_px: int = 520) -> None:
    """Render a table where selected columns contain safe HTML badges."""
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    shown = df.copy()
    note = ""
    if max_rows is not None and len(shown) > max_rows:
        shown = shown.head(max_rows)
        note = f"<div class='small-muted'>Showing first {max_rows:,} rows out of {len(df):,}. Use export for the full file.</div>"

    html_cols = set(html_cols or [])
    for col in shown.columns:
        if col not in html_cols:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else html_lib.escape(str(v)))
    html = shown.to_html(index=False, classes="clean-table", border=0, escape=False)
    st.markdown(f"{note}<div class='table-wrap' style='max-height:{height_px}px'>{html}</div>", unsafe_allow_html=True)


def classify_transaction_risk(row: pd.Series) -> str:
    """Classify operational risk using the same per-transaction RL thresholds shown in the dashboard."""
    try:
        score = float(row.get("catboost_risk_score", np.nan))
        low = float(row.get("effective_T_low", np.nan))
        high = float(row.get("effective_T_high", np.nan))
    except Exception:
        return "Unknown"
    if np.isfinite(high) and score >= high:
        return "High risk"
    if np.isfinite(low) and score >= low:
        return "Medium risk"
    if np.isfinite(score):
        return "Low risk"
    return "Unknown"


def risk_badge(level: str) -> str:
    clean = str(level or "Unknown")
    key = clean.lower()
    css = "risk-medium"
    if "low" in key:
        css = "risk-low"
    elif "high" in key:
        css = "risk-high"
    return f"<span class='risk-tag {css}'>{html_lib.escape(clean)}</span>"


def add_risk_badge_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_level_text"] = out.apply(classify_transaction_risk, axis=1)
    out["Risk level"] = out["risk_level_text"].map(risk_badge)
    return out


def render_key_value_table(series_or_dict: Any, field_name: str = "Field", value_name: str = "Value", max_rows: Optional[int] = None) -> None:
    if isinstance(series_or_dict, pd.Series):
        df = series_or_dict.reset_index()
        df.columns = [field_name, value_name]
    elif isinstance(series_or_dict, dict):
        df = pd.DataFrame(list(series_or_dict.items()), columns=[field_name, value_name])
    else:
        df = pd.DataFrame(series_or_dict)
    render_table(df, max_rows=max_rows, height_px=420)


def safe_read_json(uploaded_file) -> Dict[str, Any]:
    return json.load(uploaded_file)


def load_from_folder(folder: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Optional[pd.DataFrame]]:
    tx_path = folder / "dashboard_transactions.csv"
    metrics_path = folder / "dashboard_metrics.json"
    policy_path = folder / "rl_policy.json"
    comparison_path = folder / "policy_comparison.csv"

    missing = [str(p) for p in [tx_path, metrics_path, policy_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required output files:\n" + "\n".join(missing))

    tx = pd.read_csv(tx_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else None
    return tx, metrics, policy, comparison


def normalize_outputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for c in out.columns:
        lc = c.lower()
        if lc in ["transactionid", "transaction_id", "id"]:
            rename_map[c] = "TransactionID"
        elif lc in ["transactiondt", "transaction_dt", "dt"]:
            rename_map[c] = "TransactionDT"
        elif lc in ["transactionamt", "transaction_amt", "amount"]:
            rename_map[c] = "TransactionAmt"
    out = out.rename(columns=rename_map)

    if "rl_action" in out.columns:
        out["rl_action"] = out["rl_action"].astype(str).str.lower().str.strip()
    if "final_status" in out.columns:
        out["final_status"] = out["final_status"].astype(str).str.lower().str.strip()

    numeric_cols = [
        "TransactionAmt", "TransactionDT", "isFraud", "catboost_risk_score",
        "effective_T_low", "effective_T_high", "transaction_cost",
        "missed_fraud_cost", "false_rejection_cost", "manual_review_cost",
        "overload_cost", "protected_exposure",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "final_status" not in out.columns and "rl_action" in out.columns:
        out["final_status"] = out["rl_action"].map({
            "accept": "approved",
            "review": "pending_review",
            "reject": "blocked",
        }).fillna("unknown")

    if "human_review_decision" not in out.columns:
        out["human_review_decision"] = np.where(out.get("final_status", "") == "pending_review", "not_reviewed_yet", "")

    if "TransactionDT" in out.columns:
        out["day_index"] = (out["TransactionDT"].fillna(0).astype(float) // 86400).astype(int)
    else:
        out["day_index"] = 0
    return out


def validate_required_columns(df: pd.DataFrame) -> list:
    required = [
        "TransactionID", "TransactionAmt", "catboost_risk_score",
        "effective_T_low", "effective_T_high", "rl_action", "final_status",
    ]
    return [c for c in required if c not in df.columns]


def filter_policy_comparison(comparison: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if comparison is None:
        return None
    pc = comparison.copy()
    if "policy_name" in pc.columns:
        mask = ~pc["policy_name"].astype(str).str.lower().str.contains(
            "amount-bucketed|amount bucketed|bucketed grid", na=False
        )
        pc = pc.loc[mask].reset_index(drop=True)
    return pc


def operational_view(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in GROUND_TRUTH_COLUMNS if c in df.columns], errors="ignore")


def _status_from_action(action: str) -> str:
    return {
        "accept": "approved",
        "review": "pending_review",
        "reject": "blocked",
    }.get(str(action).lower().strip(), "unknown")


def apply_decision_overrides(df: pd.DataFrame, overrides: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Apply session-only employee overrides to the operational dataframe.

    This does not change the offline evaluation data or ground-truth metrics.
    It only updates the bank-operations view and records overrides in the audit trail.
    """
    out = df.copy()
    out["override_status"] = ""
    out["override_reason"] = ""
    out["override_timestamp"] = ""

    if not overrides or "TransactionID" not in out.columns:
        return out

    tx_ids = out["TransactionID"].astype(str)
    for tx_id, rec in overrides.items():
        mask = tx_ids == str(tx_id)
        if not mask.any():
            continue
        new_action = str(rec.get("override_action", "")).lower().strip()
        if new_action not in ["accept", "review", "reject"]:
            continue
        out.loc[mask, "rl_action"] = new_action
        out.loc[mask, "final_status"] = rec.get("override_final_status", _status_from_action(new_action))
        out.loc[mask, "human_review_decision"] = rec.get("override_reason", "employee_override")
        out.loc[mask, "override_status"] = "overridden"
        out.loc[mask, "override_reason"] = rec.get("override_reason", "")
        out.loc[mask, "override_timestamp"] = rec.get("timestamp", "")
    return out


def build_operational_report(df: pd.DataFrame, metrics: Dict[str, Any], policy: Dict[str, Any]) -> str:
    lines = []
    lines.append("Hybrid (AE + CatBoost) + RL Fraud Operations Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Operational metrics")
    for key in ["total_transactions", "accepted_count", "review_count", "rejected_count", "review_rate"]:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    lines.append(f"- reviewers: {policy.get('reviewers', 'N/A')}")
    lines.append(f"- reviews_per_reviewer_per_day: {policy.get('reviews_per_reviewer_per_day', 'N/A')}")
    lines.append("")
    lines.append("Note: ground-truth labels and offline evaluation metrics are hidden from the operational view.")
    return "\n".join(lines)


def build_evaluation_report(df: pd.DataFrame, metrics: Dict[str, Any], policy: Dict[str, Any], comparison: Optional[pd.DataFrame]) -> str:
    lines = []
    lines.append("Hybrid (AE + CatBoost) + RL Offline Evaluation Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Evaluation metrics")
    for key in [
        "total_transactions", "accepted_count", "review_count", "rejected_count",
        "auc_roc_test", "precision", "recall", "false_positive_rate",
        "review_rate", "total_cost", "total_profit", "profit_gain", "protected_exposure",
        "manual_review_cost_total", "missed_fraud_cost_total", "false_rejection_cost_total",
        "overload_cost_total", "tp", "fp", "tn", "fn",
    ]:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    lines.append("")
    lines.append("Policy")
    for key, value in policy.items():
        lines.append(f"- {key}: {value}")
    if comparison is not None and not comparison.empty:
        lines.append("")
        lines.append("Policy comparison")
        lines.append(comparison.to_string(index=False))
    return "\n".join(lines)


# ----------------------------- Sidebar load -----------------------------
st.sidebar.title("Dashboard inputs")
st.sidebar.caption("PROJECTOR_FIX_V35 loaded")
st.sidebar.caption("Output mode: loads finished hybrid (AE + CatBoost) + RL outputs. It does not train models or run inference.")

load_mode = st.sidebar.radio(
    "Load method",
    ["Use local dashboard_outputs folder", "Upload output files manually"],
    index=0,
)

transactions_df = None
metrics = None
policy = None
policy_comparison = None
load_error = None

try:
    if load_mode == "Use local dashboard_outputs folder":
        folder_text = "dashboard_outputs"
        st.sidebar.markdown("**Output folder:** `dashboard_outputs`")
        folder = Path(folder_text)
        if not folder.is_absolute():
            folder = Path(__file__).resolve().parent / folder
        transactions_df, metrics, policy, policy_comparison = load_from_folder(folder)
    else:
        tx_file = st.sidebar.file_uploader("Upload dashboard_transactions.csv", type=["csv"])
        metrics_file = st.sidebar.file_uploader("Upload dashboard_metrics.json", type=["json"])
        policy_file = st.sidebar.file_uploader("Upload rl_policy.json", type=["json"])
        comparison_file = st.sidebar.file_uploader("Optional: upload policy_comparison.csv", type=["csv"])
        if tx_file and metrics_file and policy_file:
            transactions_df = pd.read_csv(tx_file)
            metrics = safe_read_json(metrics_file)
            policy = safe_read_json(policy_file)
            policy_comparison = pd.read_csv(comparison_file) if comparison_file is not None else None
        else:
            st.info("Upload the three required files to load the dashboard.")
            st.stop()
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(load_error)
    st.stop()

transactions_df = normalize_outputs(transactions_df)
missing_cols = validate_required_columns(transactions_df)
if missing_cols:
    st.error("The transactions file is missing required columns: " + ", ".join(missing_cols))
    st.stop()

policy_comparison = filter_policy_comparison(policy_comparison)

# ---------- Live staffing control ----------
live_deployment = None
if load_mode == "Use local dashboard_outputs folder":
    _bundle_path = folder / "deployment_bundle.npz"
    if _bundle_path.exists():
        try:
            from fraudcore.dashboard_export import load_bundle as _load_bundle, live_dashboard_payload as _live
            _bk = _load_bundle(str(_bundle_path))
        except Exception as _e:
            _bk = None
            st.sidebar.caption(f"Live re-tune unavailable: {_e}")
        if _bk is not None and _bk.get("mode") == "capacity" and "val_scores" in _bk and "headline_policy_a4" in _bk and _bk.get("txns_per_day"):
            _file_rev = int(policy.get("reviewers", 6) or 6)
            _rpd = int(_bk.get("reviews_per_reviewer_per_day", 100) or 100)
            st.sidebar.markdown("---")
            st.sidebar.subheader("Live staffing")
            st.sidebar.caption("Re-tunes the budget-fit and triage on validation. PPO stays frozen — no retraining.")
            n_reviewers = int(st.sidebar.number_input(
                "Reviewers",
                min_value=1,
                value=max(int(_file_rev), 1),
                step=1,
                format="%d",
                help="Type a positive whole number. Daily capacity = reviewers × reviews/day. Changing this re-derives the review budget and re-tunes the static fit + triage live."
            ))

            @st.cache_data(show_spinner=False)
            def _retune(bundle_path, mtime, reviewers, rpd):
                bk = _load_bundle(bundle_path)
                pl = _live(
                    int(reviewers),
                    val_scores=bk["val_scores"], val_labels=bk["val_labels"], val_amounts=bk["val_amounts"],
                    test_scores=bk["scores"], test_labels=bk["labels"], test_amounts=bk["amounts"],
                    meta_df=bk["meta_df"], headline_policy_a4=bk["headline_policy_a4"],
                    z_mean=bk["z_mean"], z_std=bk["z_std"], min_t_low=bk["min_t_low"], max_t_high=bk["max_t_high"],
                    reviews_per_reviewer_per_day=int(rpd), txns_per_day=bk["txns_per_day"],
                    grid_thresholds=bk.get("grid_thresholds"), single_thresholds=bk.get("single_thresholds"),
                    bucketed_decisions=bk.get("bucketed_decisions"),
                    auc_val=bk.get("auc_val"), auc_test=bk.get("auc_test"))
                return pl["transactions"], pl["metrics"], pl["policy"], pl["comparison"], pl["deployment"]

            if n_reviewers != _file_rev:
                with st.spinner(f"Re-tuning for {n_reviewers} reviewers (PPO frozen)…"):
                    tx2, m2, p2, c2, dep2 = _retune(str(_bundle_path), os.path.getmtime(_bundle_path), n_reviewers, _rpd)
                transactions_df = normalize_outputs(tx2)
                metrics, policy, policy_comparison, live_deployment = m2, p2, filter_policy_comparison(c2), dep2
                st.sidebar.success(f"Live · {n_reviewers} reviewers · budget {100*p2['review_budget_frac']:.1f}% · triage k={dep2['triage_k']:g}")
            else:
                st.sidebar.caption(f"Showing the exported {_file_rev}-reviewer result. Change the number to re-tune.")

if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []
if "decision_overrides" not in st.session_state:
    st.session_state.decision_overrides = {}

ops_transactions_df = apply_decision_overrides(transactions_df, st.session_state.decision_overrides)

# ----------------------------- Header -----------------------------
_hybrid = bool(policy.get("score_is_hybrid", True))
_model_short = "Hybrid (AE + CatBoost)" if _hybrid else "CatBoost"
_score_label = "hybrid denoising-autoencoder + CatBoost" if _hybrid else "CatBoost"

st.title(f"{_model_short} + RL Fraud Review Dashboard")
st.caption(f"This dashboard reads completed {_model_short.lower()} + RL outputs. It does not train models or rerun the RL pipeline.")

if metrics.get("dummy_data") or policy.get("dummy_data"):
    st.warning("You are viewing dummy placeholder outputs. Replace the files inside dashboard_outputs/ with real outputs before presenting final results.")

_ptype = str(policy.get("policy_type", ""))
if _ptype.startswith("capacity"):
    _triage_txt = " plus a budget-pressure triage rule" if (policy.get("triage_k") and policy.get("triage_gate", "off") != "off") else ""
    _policy_desc = (
        "The <b>deployed policy</b> applies amount-conditioned accept / review / reject thresholds "
        "under a fixed daily review budget"
        f"{_triage_txt}. If reviewer capacity is exhausted, extra review cases fall back to a profit-optimal accept/reject cutoff."
    )
else:
    _policy_desc = "The <b>PPO policy</b> uses the score and amount-conditioned thresholds to assign accept, review, or reject."

st.markdown(
    f"""
<div class="section-note">
<b>System flow:</b> a {_score_label} model produces <code>catboost_risk_score</code> for each transaction. {_policy_desc}
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------- Core counts -----------------------------
total_transactions = int(metrics.get("total_transactions", len(transactions_df)))
accepted_count = int(metrics.get("accepted_count", (transactions_df["rl_action"] == "accept").sum()))
review_count = int(metrics.get("review_count", (transactions_df["rl_action"] == "review").sum()))
rejected_count = int(metrics.get("rejected_count", (transactions_df["rl_action"] == "reject").sum()))
predicted_blocked_amount = float(transactions_df.loc[transactions_df["rl_action"] == "reject", "TransactionAmt"].sum())
protected_exposure = float(metrics.get("protected_exposure", transactions_df.get("protected_exposure", pd.Series([0])).sum()))
total_cost = float(metrics.get("total_cost", transactions_df.get("transaction_cost", pd.Series([0])).sum()))
profit_gain = float(metrics.get("profit_gain", 0.0))
review_rate = float(metrics.get("review_rate", review_count / max(total_transactions, 1)))
reviewers = int(policy.get("reviewers", 0) or 0)
reviews_per_reviewer = int(policy.get("reviews_per_reviewer_per_day", 0) or 0)
daily_capacity = reviewers * reviews_per_reviewer

if "TransactionDT" in transactions_df.columns and transactions_df["TransactionDT"].notna().any():
    _dt = pd.to_numeric(transactions_df["TransactionDT"], errors="coerce")
    _span_days = max((float(_dt.max()) - float(_dt.min())) / 86400.0, 1e-9)
elif "day_index" in transactions_df.columns and transactions_df["day_index"].notna().any():
    _span_days = max(float(transactions_df["day_index"].max() - transactions_df["day_index"].min()) + 1.0, 1.0)
else:
    _span_days = 1.0

daily_reviews = review_count / _span_days
capacity_util = daily_reviews / daily_capacity if daily_capacity > 0 else np.nan
avg_risk = float(transactions_df["catboost_risk_score"].mean())
total_dataset_amount = float(transactions_df["TransactionAmt"].sum())

ops_total_transactions = len(ops_transactions_df)
ops_accepted_count = int((ops_transactions_df["rl_action"] == "accept").sum())
ops_review_count = int((ops_transactions_df["rl_action"] == "review").sum())
ops_rejected_count = int((ops_transactions_df["rl_action"] == "reject").sum())
ops_review_rate = ops_review_count / max(ops_total_transactions, 1)
ops_predicted_blocked_amount = float(ops_transactions_df.loc[ops_transactions_df["rl_action"] == "reject", "TransactionAmt"].sum())
ops_total_dataset_amount = float(ops_transactions_df["TransactionAmt"].sum())
ops_daily_reviews = ops_review_count / _span_days
ops_capacity_util = ops_daily_reviews / daily_capacity if daily_capacity > 0 else np.nan

# ----------------------------- Tabs -----------------------------
st.markdown("### Choose dashboard view")
ops_tab, eval_tab = st.tabs(["🏦 Bank operations", "📊 Offline evaluation"])

with ops_tab:
    opdf = ops_transactions_df
    st.markdown(
        """
<div class="section-note">
<b>Operational view:</b> this is the bank-employee view. It hides ground-truth labels and evaluation-only fields such as <code>isFraud</code>, confusion matrix, detected/missed fraud, and true cost buckets.
</div>
""",
        unsafe_allow_html=True,
    )

    ops_overview_tab, ops_review_tab, ops_risk_tab = st.tabs(["Overview", "Review & Override", "Risk Insights"])

    with ops_overview_tab:
        st.subheader("Executive summary")
        row1 = st.columns(4)
        with row1[0]:
            metric_card("Transactions", fmt_num(ops_total_transactions), "Loaded output rows")
        with row1[1]:
            metric_card("Accepted transactions", fmt_num(ops_accepted_count), f"Transactions accepted by system: {fmt_pct(ops_accepted_count / max(ops_total_transactions, 1))}")
        with row1[2]:
            metric_card("Blocked transactions", fmt_num(ops_rejected_count), f"Transactions rejected by system: {fmt_pct(ops_rejected_count / max(ops_total_transactions, 1))}")
        with row1[3]:
            metric_card("Manual reviews", fmt_num(ops_review_count), f"Review rate {fmt_pct(ops_review_rate)}")

        row2 = st.columns(4)
        with row2[0]:
            metric_card("Blocked amount", fmt_money_compact(ops_predicted_blocked_amount), "Rejected transaction amount by system")
        with row2[1]:
            metric_card("Total amount", fmt_money_compact(ops_total_dataset_amount), "Sum of transaction amounts")
        with row2[2]:
            metric_card("Daily review capacity", fmt_num(daily_capacity), f"{reviewers} reviewers × {reviews_per_reviewer}/day")
        with row2[3]:
            metric_card("Capacity utilization", fmt_pct(ops_capacity_util) if np.isfinite(ops_capacity_util) else "N/A", f"~{ops_daily_reviews:,.0f} reviews/day vs {daily_capacity:,.0f}/day capacity")
        with st.expander("Metric definitions and formula sheet", expanded=False):
            st.markdown(
                """
### Risk score
`catboost_risk_score` is the hybrid classifier output for each transaction. The score is not the transaction amount; amount only shifts the RL thresholds.

### RL decision rule
`accept` if `risk_score < effective_T_low`  
`review` if `effective_T_low <= risk_score < effective_T_high`  
`reject` if `risk_score >= effective_T_high`

### Deployment note
Bank employees do not see the true `isFraud` label at decision time. Ground-truth metrics are kept only in the Offline Evaluation tab for historical testing.
    """
            )
            st.markdown("#### RL policy values loaded from `rl_policy.json`")
            render_key_value_table(policy, field_name="Policy parameter", value_name="Value", max_rows=None)

        st.subheader("Operational exports")
        op_report_txt = build_operational_report(opdf, metrics, policy)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download operational report", data=op_report_txt, file_name="operational_report.txt", mime="text/plain")
        with d2:
            st.download_button("Download operational transactions CSV", data=operational_view(opdf).to_csv(index=False), file_name="operational_transactions_export.csv", mime="text/csv")

        st.subheader("Operational decision analytics")
        left, right = st.columns(2)
        with left:
            action_counts = opdf["rl_action"].value_counts().reindex(ACTION_ORDER).fillna(0).reset_index()
            action_counts.columns = ["Action", "Count"]
            action_counts["Action label"] = action_counts["Action"].map({
                "accept": "Accept",
                "review": "Review",
                "reject": "Reject",
            }).fillna(action_counts["Action"])
            action_label_colors = {
                "Accept": ACTION_COLORS.get("accept", "#22c55e"),
                "Review": ACTION_COLORS.get("review", "#f59e0b"),
                "Reject": ACTION_COLORS.get("reject", "#ef4444"),
            }
            fig = px.pie(
                action_counts,
                names="Action label",
                values="Count",
                hole=0.45,
                title="Action Mix: Accept / Review / Reject",
                color="Action label",
                color_discrete_map=action_label_colors,
            )
            fig.update_traces(
                texttemplate="%{label}<br>%{value:,}<br>%{percent}",
                textposition="outside",
                textfont=dict(size=20, color="#111827"),
                outsidetextfont=dict(size=20, color="#111827"),
                marker=dict(line=dict(color="#ffffff", width=3)),
                pull=[0.02] * len(action_counts),
                sort=False,
                automargin=True,
                hovertemplate="%{label}<br>Transactions: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.36, xanchor="center", x=0.5, font=dict(size=18)),
                uniformtext=dict(minsize=18, mode="show"),
                margin=dict(t=230, b=60, l=60, r=140),
                height=660,
            )
            show_fig(fig)
        with right:
            used_reviews = float(ops_daily_reviews) if np.isfinite(ops_daily_reviews) else 0.0
            capacity = float(daily_capacity) if daily_capacity > 0 else 0.0
            used_inside_capacity = min(used_reviews, capacity) if capacity > 0 else used_reviews
            remaining_capacity = max(capacity - used_reviews, 0.0)
            overload_reviews = max(used_reviews - capacity, 0.0)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=["Daily reviews"],
                x=[used_inside_capacity],
                name="Used capacity",
                orientation="h",
                text=[f"{used_reviews:,.0f} reviews/day"],
                textposition="inside",
                marker=dict(color="#2563eb"),
            ))
            if remaining_capacity > 0:
                fig.add_trace(go.Bar(
                    y=["Daily reviews"],
                    x=[remaining_capacity],
                    name="Remaining capacity",
                    orientation="h",
                    text=[f"{remaining_capacity:,.0f} remaining"],
                    textposition="inside",
                    marker=dict(color="#bfdbfe"),
                ))
            if overload_reviews > 0:
                fig.add_trace(go.Bar(
                    y=["Daily reviews"],
                    x=[overload_reviews],
                    name="Over capacity",
                    orientation="h",
                    text=[f"{overload_reviews:,.0f} over"],
                    textposition="inside",
                    marker=dict(color="#ef4444"),
                ))
            if capacity > 0:
                fig.add_shape(
                    type="line",
                    x0=capacity,
                    x1=capacity,
                    y0=-0.5,
                    y1=0.5,
                    xref="x",
                    yref="y",
                    line=dict(color="#0f172a", width=3, dash="dash"),
                )
                fig.add_annotation(
                    x=capacity,
                    y=1.12,
                    xref="x",
                    yref="paper",
                    text="Capacity limit",
                    showarrow=False,
                    font=dict(size=18, color="#0f172a"),
                    xanchor="center",
                    yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#cbd5e1",
                    borderwidth=1,
                    borderpad=4,
                )
            fig.update_layout(
                title="Review Capacity Usage",
                barmode="stack",
                xaxis_title="Reviews per day",
                yaxis_title="",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.28,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=17, color="#0f172a"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#cbd5e1",
                    borderwidth=1,
                ),
                margin=dict(t=155, b=150, l=90, r=90),
                height=540,
            )
            fig.update_traces(textfont=dict(size=18, color="#0f172a"), textposition="inside")
            fig.update_xaxes(range=[0, max(used_reviews, capacity, 1.0) * 1.22])

            # Custom render for this chart: keep the legend above the capacity label
            # so they never overlap on projector-sized screens.
            fig = style_fig(fig)
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.34,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=17, color="#0f172a"),
                    bgcolor="rgba(255,255,255,0.97)",
                    bordercolor="rgba(255,255,255,0)",
                    borderwidth=0,
                    title_text="",
                ),
                margin=dict(t=215, b=115, l=90, r=90),
                height=560,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                opdf,
                x="catboost_risk_score",
                color="rl_action",
                nbins=40,
                barmode="overlay",
                title="Transaction Risk Distribution",
                color_discrete_map=ACTION_COLORS,
            )
            # Amount-conditioned thresholds are functions, not fixed values,
            # so this histogram only shows the distribution of risk scores by decision.
            # The threshold behavior is shown separately in the RL Threshold Behavior chart.
            fig.update_layout(bargap=0.05, xaxis_title="Risk score", yaxis_title="Transaction count")
            show_fig(fig)
        with right:
            daily = opdf.groupby("day_index").agg(
                reviews=("rl_action", lambda s: (s == "review").sum()),
                rejects=("rl_action", lambda s: (s == "reject").sum()),
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["day_index"], y=daily["reviews"], mode="lines+markers", name="Reviews", line=dict(color="#2563eb", width=4)))
            fig.add_trace(go.Scatter(x=daily["day_index"], y=daily["rejects"], mode="lines+markers", name="Rejects", line=dict(color="#ef4444", width=3)))
            if daily_capacity > 0 and not daily.empty:
                fig.add_trace(go.Scatter(
                    x=daily["day_index"],
                    y=[daily_capacity] * len(daily),
                    mode="lines",
                    name="Reviewer capacity",
                    line=dict(color="#0f172a", width=3, dash="dash"),
                ))
            fig.update_layout(title="Daily Review Load", xaxis_title="Day index", yaxis_title="Count")
            show_fig(fig)


    with ops_risk_tab:
        st.subheader("High-risk transactions to monitor")
        high_risk_df = add_risk_badge_columns(opdf)
        high_risk_df = high_risk_df[high_risk_df["risk_level_text"] == "High risk"].copy()
        high_risk_df = high_risk_df.sort_values("catboost_risk_score", ascending=False)

        high_risk_cols = [c for c in [
            "Risk level", "TransactionID", "TransactionAmt", "card1", "catboost_risk_score",
            "effective_T_low", "effective_T_high", "rl_action", "final_status"
        ] if c in high_risk_df.columns]

        if high_risk_df.empty:
            st.success("No high-risk transactions are currently present in this batch.")
        else:
            render_table_with_html(high_risk_df[high_risk_cols], html_cols=["Risk level"], max_rows=None, height_px=620)
            st.caption(f"Showing all {len(high_risk_df):,} high-risk transactions, sorted by risk score.")

        st.subheader("Amount-aware risk analysis")
        threshold_panel, card_group_panel = st.columns(2)

        with threshold_panel:
            threshold_cols = ["TransactionAmt", "catboost_risk_score", "rl_action"]
            sample = opdf[threshold_cols].dropna().copy()
            sample = sample[sample["TransactionAmt"] > 0].sort_values("TransactionAmt")
            if len(sample) > 1500:
                sample = sample.iloc[np.linspace(0, len(sample) - 1, 1500).astype(int)]
            if sample.empty:
                st.info("No amount-versus-risk data available for this batch.")
            else:
                fig = px.scatter(
                    sample,
                    x="TransactionAmt",
                    y="catboost_risk_score",
                    color="rl_action",
                    title="Risk Score vs Transaction Amount by Final Decision",
                    labels={
                        "TransactionAmt": "Transaction amount",
                        "catboost_risk_score": "Risk score",
                        "rl_action": "Final decision",
                    },
                    color_discrete_map=ACTION_COLORS,
                    category_orders={"rl_action": ["accept", "review", "reject"]},
                    opacity=0.55,
                )
                fig.update_traces(marker=dict(size=8))
                candidate_ticks = [1, 10, 100, 1000, 5000, 10000]
                max_amt = float(sample["TransactionAmt"].max())
                min_amt = max(float(sample["TransactionAmt"].min()), 1.0)
                tickvals = [v for v in candidate_ticks if min_amt <= v <= max(max_amt * 1.05, 1)]
                if 5000 <= max_amt < 10000 and 5000 not in tickvals:
                    tickvals.append(5000)
                tickvals = sorted(set(tickvals))
                ticktext_map = {1: "$1", 10: "$10", 100: "$100", 1000: "$1k", 5000: "$5k", 10000: "$10k"}
                fig.update_layout(
                    height=560,
                    margin=dict(l=70, r=35, t=145, b=95),
                    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=14)),
                )
                fig.update_xaxes(
                    type="log",
                    title_text="Transaction amount ($)",
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=[ticktext_map[v] for v in tickvals],
                )
                fig.update_yaxes(title_text="Risk score", range=[0, 1.02])
                show_fig(fig)
                st.caption("This chart shows how accepted, reviewed, and rejected transactions are distributed across transaction amounts and risk scores. It is intended for operational interpretation rather than internal threshold inspection.")

        with card_group_panel:
            if "card1" in opdf.columns:
                group = opdf.groupby("card1").agg(
                    amount=("TransactionAmt", "sum"),
                    avg_risk=("catboost_risk_score", "mean"),
                    txns=("TransactionID", "count") if "TransactionID" in transactions_df.columns else ("rl_action", "count"),
                ).reset_index()
                top_group = group.sort_values("amount", ascending=False).head(25)
                fig = px.scatter(
                    top_group,
                    x="amount",
                    y="avg_risk",
                    size="txns",
                    color="avg_risk",
                    hover_name="card1",
                    title="Card Group Amount and Risk",
                    labels={"amount": "Total transaction amount", "avg_risk": "Average risk score", "txns": "Transactions"},
                    color_continuous_scale="Turbo",
                )
                fig.update_layout(
                    height=560,
                    margin=dict(l=70, r=35, t=145, b=95),
                    coloraxis_colorbar=dict(title=dict(text="Avg risk", font=dict(size=14)), tickfont=dict(size=13)),
                )
                show_fig(fig)
            else:
                st.info("No card-group field is available for this batch.")


    with ops_review_tab:
        st.subheader("Prioritized review queue")
        review_df = opdf[opdf["rl_action"] == "review"].copy().sort_values("catboost_risk_score", ascending=False)
        if review_df.empty:
            st.success("No transactions are currently pending manual review.")
        else:
            review_display_df = add_risk_badge_columns(review_df)
            queue_cols = [c for c in [
                "Risk level", "TransactionID", "TransactionAmt", "card1", "catboost_risk_score",
                "effective_T_low", "effective_T_high", "rl_action", "human_review_decision", "final_status"
            ] if c in review_display_df.columns]
            render_table_with_html(review_display_df[queue_cols], html_cols=["Risk level"], max_rows=200, height_px=520)

            st.markdown("#### Take action on transaction")
            q1, q2 = st.columns([1, 2])
            with q1:
                selected_id = st.selectbox("Select Transaction ID", review_display_df["TransactionID"].astype(str).tolist())
                selected_row = review_display_df[review_display_df["TransactionID"].astype(str) == selected_id].iloc[0]
                decision = st.radio("Human reviewer decision", ["Approve", "Reject", "Hold for investigation"], horizontal=False)
                notes = st.text_area("Notes", placeholder="Optional review notes")
                submit = st.button("Submit decision")
            with q2:
                st.markdown("##### Selected transaction details")
                detail_cols = [c for c in [
                    "TransactionID", "TransactionAmt", "card1", "catboost_risk_score",
                    "effective_T_low", "effective_T_high", "risk_level_text", "rl_action",
                    "human_review_decision", "final_status"
                ] if c in selected_row.index]
                details = selected_row[detail_cols].rename({"risk_level_text": "Risk level"})
                render_key_value_table(details, field_name="Field", value_name="Value", max_rows=None)

            if submit:
                final_status_map = {
                    "Approve": "approved",
                    "Reject": "blocked",
                    "Hold for investigation": "under investigation",
                }
                st.session_state.audit_trail.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "manual_review_decision",
                    "TransactionID": selected_id,
                    "previous_action": selected_row.get("rl_action", "review"),
                    "new_action": decision,
                    "final_status": final_status_map[decision],
                    "reason": "Human reviewer decision",
                    "catboost_risk_score": float(selected_row["catboost_risk_score"]),
                    "TransactionAmt": float(selected_row["TransactionAmt"]),
                    "notes": notes,
                })
                st.success("Decision added to the session audit trail.")

        st.subheader("Decision override")
        st.markdown(
            """
    <div class="section-note">
    Use this when a customer or employee confirms that the system decision should be changed. Overrides are session-based and recorded in the audit trail.
    </div>
    """,
            unsafe_allow_html=True,
        )

        o1, o2 = st.columns([1, 2])
        with o1:
            override_id = st.text_input("Transaction ID", placeholder="Enter TransactionID")
            override_choice = st.selectbox(
                "Override system decision",
                ["Approve transaction", "Block transaction", "Send to manual review"],
            )
            override_reason = st.selectbox(
                "Reason",
                ["Customer confirmed legitimate", "Customer reported fraud", "Analyst manual decision", "Other"],
            )
            override_notes = st.text_area("Override notes", placeholder="Optional notes")
            override_submit = st.button("Apply override")

        with o2:
            st.markdown("##### Transaction selected for override")
            if override_id.strip():
                selected_override = opdf[opdf["TransactionID"].astype(str) == override_id.strip()]
                if selected_override.empty:
                    st.warning("No transaction found with this ID.")
                else:
                    override_cols = [c for c in [
                        "TransactionID", "TransactionAmt", "card1", "catboost_risk_score",
                        "effective_T_low", "effective_T_high", "rl_action", "final_status",
                        "override_status", "override_reason", "override_timestamp",
                    ] if c in selected_override.columns]
                    render_key_value_table(selected_override.iloc[0][override_cols], field_name="Field", value_name="Current value", max_rows=None)
            else:
                st.info("Enter a transaction ID to review and override its current system decision.")

        if override_submit:
            tx_id = override_id.strip()
            original_match = transactions_df[transactions_df["TransactionID"].astype(str) == tx_id]
            current_match = opdf[opdf["TransactionID"].astype(str) == tx_id]
            if not tx_id or original_match.empty or current_match.empty:
                st.error("Please enter a valid Transaction ID before applying an override.")
            else:
                action_map = {
                    "Approve transaction": ("accept", "approved"),
                    "Block transaction": ("reject", "blocked"),
                    "Send to manual review": ("review", "pending_review"),
                }
                new_action, new_status = action_map[override_choice]
                original_row = original_match.iloc[0]
                current_row = current_match.iloc[0]
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # If this transaction already had a manual review or previous override
                # in the audit trail, use that latest audit action as the previous action.
                # Otherwise, fall back to the current system/dashboard action.
                previous_action_for_audit = current_row.get("rl_action", "")
                previous_status_for_audit = current_row.get("final_status", "")
                for audit_record in reversed(st.session_state.audit_trail):
                    if str(audit_record.get("TransactionID", "")) == str(tx_id):
                        previous_action_for_audit = audit_record.get("new_action", previous_action_for_audit)
                        previous_status_for_audit = audit_record.get("final_status", previous_status_for_audit)
                        break

                st.session_state.decision_overrides[tx_id] = {
                    "timestamp": ts,
                    "override_action": new_action,
                    "override_final_status": new_status,
                    "override_reason": override_reason,
                    "override_notes": override_notes,
                }
                st.session_state.audit_trail.append({
                    "timestamp": ts,
                    "event_type": "decision_override",
                    "TransactionID": tx_id,
                    "previous_action": previous_action_for_audit,
                    "previous_status": previous_status_for_audit,
                    "new_action": new_action,
                    "final_status": new_status,
                    "reason": override_reason,
                    "catboost_risk_score": float(original_row.get("catboost_risk_score", np.nan)),
                    "TransactionAmt": float(original_row.get("TransactionAmt", np.nan)),
                    "notes": override_notes,
                })
                st.success("Override applied and recorded in the audit trail.")
                st.rerun()

        with st.expander("Audit trail", expanded=False):
            if st.session_state.audit_trail:
                clear_col, remove_col = st.columns([1, 2])

                with clear_col:
                    if st.button("Clear audit trail"):
                        st.session_state.audit_trail = []
                        st.success("Audit trail cleared.")
                        st.rerun()

                with remove_col:
                    audit_tx_ids = sorted({
                        str(row.get("TransactionID", ""))
                        for row in st.session_state.audit_trail
                        if str(row.get("TransactionID", "")).strip()
                    })
                    selected_audit_tx = st.selectbox(
                        "Remove one transaction from audit trail",
                        audit_tx_ids,
                        key="remove_audit_transaction_id",
                    )
                    if st.button("Remove selected transaction"):
                        st.session_state.audit_trail = [
                            row for row in st.session_state.audit_trail
                            if str(row.get("TransactionID", "")) != str(selected_audit_tx)
                        ]
                        st.success(f"Removed audit trail rows for transaction {selected_audit_tx}.")
                        st.rerun()

                audit_df = pd.DataFrame(st.session_state.audit_trail)

                # Only hide the requested audit trail columns.
                audit_df = audit_df.drop(columns=[c for c in ["catboost_risk_score", "previous_status"] if c in audit_df.columns], errors="ignore")

                summary_rows = []

                if "event_type" in audit_df.columns:
                    manual_review_count = int((audit_df["event_type"] == "manual_review_decision").sum())
                    override_count = int((audit_df["event_type"] == "decision_override").sum())
                    summary_rows.extend([
                        {"Action": "Manual review decisions", "Count": manual_review_count},
                        {"Action": "Decision overrides", "Count": override_count},
                    ])

                if "final_status" in audit_df.columns:
                    status_counts = audit_df["final_status"].astype(str).str.lower().value_counts()
                    summary_rows.extend([
                        {"Action": "Approved actions", "Count": int(status_counts.get("approved", 0))},
                        {"Action": "Rejected actions", "Count": int(status_counts.get("blocked", 0))},
                        {"Action": "Sent to review", "Count": int(status_counts.get("pending_review", 0))},
                        {"Action": "Under investigation", "Count": int(status_counts.get("under investigation", 0))},
                    ])

                summary_df = pd.DataFrame(summary_rows)
                summary_df = summary_df[summary_df["Count"] > 0] if not summary_df.empty else summary_df

                if not summary_df.empty:
                    fig = px.bar(
                        summary_df,
                        x="Action",
                        y="Count",
                        color="Action",
                        title="Audit Trail Action Summary",
                        text="Count",
                        color_discrete_sequence=CHART_COLORS,
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        xaxis_title="",
                        yaxis_title="Count",
                        showlegend=False,
                        height=380,
                        margin=dict(l=70, r=35, t=105, b=90),
                    )
                    show_fig(fig)
                else:
                    st.info("No audit actions to summarize yet.")

                # Keep manual review decisions and later overrides as separate audit events.
                # The same TransactionID may appear in multiple rows, one row per action taken.
                audit_order = [
                    "timestamp", "event_type", "TransactionID", "previous_action", "new_action",
                    "final_status", "reason", "TransactionAmt", "notes"
                ]
                audit_cols = [c for c in audit_order if c in audit_df.columns] + [c for c in audit_df.columns if c not in audit_order]
                audit_df = audit_df[audit_cols]

                audit_df = audit_df.rename(columns={
                    "timestamp": "Timestamp",
                    "event_type": "Event type",
                    "TransactionID": "Transaction ID",
                    "previous_action": "Previous action",
                    "new_action": "New action",
                    "final_status": "Final status",
                    "reason": "Reason",
                    "TransactionAmt": "Transaction amount",
                    "notes": "Notes",
                })

                render_table(audit_df, max_rows=None, height_px=420)
                st.download_button("Download audit trail CSV", data=audit_df.to_csv(index=False), file_name="audit_trail.csv", mime="text/csv")
            else:
                st.info("No manual review actions have been submitted in this session yet.")

        with st.expander("Full operational transaction table", expanded=False):
            render_table(operational_view(opdf), max_rows=500, height_px=520)

with eval_tab:
    st.markdown(
        """
<div class="warning-note">
<b>Offline evaluation view:</b> this section uses historical labeled test data. It includes ground-truth metrics such as precision, recall, detected/missed fraud, confusion matrix, and true cost buckets. This would be hidden from bank employees during live operation.
</div>
""",
        unsafe_allow_html=True,
    )

    eval_model_tab, eval_financial_tab, eval_policy_tab = st.tabs(["Model Performance", "Financial Evaluation", "Policy Comparison"])

    with eval_model_tab:
        summary_rows = [
            ["Test AUC-ROC", metrics.get("auc_roc_test")],
            ["Precision", fmt_pct(metrics.get("precision"))],
            ["Recall", fmt_pct(metrics.get("recall"))],
            ["Total cost", f"{fmt_money(total_cost)}  —  Missed fraud + false rejection + review + overload costs."],
            ["Caught fraud", f"{fmt_money(protected_exposure)}  —  fraud caught by system and reviewers"],
        ]

        model_left, model_right = st.columns([1.05, 1.25], gap="large")

        with model_left:
            st.subheader("Evaluation summary")
            render_table_no_scroll(pd.DataFrame(summary_rows, columns=["Metric", "Value"]))

        with model_right:
            st.subheader("Confusion matrix")
            if all(k in metrics for k in ["tp", "fp", "tn", "fn"]):
                z = np.array([[metrics.get("tn", 0), metrics.get("fp", 0)], [metrics.get("fn", 0), metrics.get("tp", 0)]])
                x_labels = ["Predicted legit", "Predicted fraud"]
                y_labels = ["Actual legit", "Actual fraud"]
                fig = go.Figure(data=go.Heatmap(
                    z=z,
                    x=x_labels,
                    y=y_labels,
                    hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
                    colorscale="YlGnBu",
                    showscale=True,
                ))
                max_z = float(np.nanmax(z)) if np.size(z) else 0.0
                for row_i, y_label in enumerate(y_labels):
                    for col_i, x_label in enumerate(x_labels):
                        val = int(z[row_i, col_i])
                        text_color = "#ffffff" if max_z and val >= 0.45 * max_z else "#0f172a"
                        fig.add_annotation(
                            x=x_label,
                            y=y_label,
                            text=f"{val:,}",
                            showarrow=False,
                            font=dict(color="#0f172a", size=20),
                            bgcolor="rgba(255,255,255,0.82)",
                            bordercolor="rgba(15,23,42,0.25)",
                            borderwidth=1,
                            borderpad=3,
                        )
                fig.update_layout(
                    title_text="",
                    height=430,
                    margin=dict(l=70, r=35, t=90, b=80),
                )
                show_fig(fig)
            else:
                st.info("Confusion matrix is unavailable because TP/FP/TN/FN were not provided.")

        st.subheader("Evaluation exports")
        eval_report_txt = build_evaluation_report(transactions_df, metrics, policy, policy_comparison)
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("Download evaluation report", data=eval_report_txt, file_name="evaluation_report.txt", mime="text/plain")
        with e2:
            export_metrics = {k: v for k, v in metrics.items() if k != "auc_roc_val"}
            st.download_button("Download evaluation metrics JSON", data=json.dumps(export_metrics, indent=2), file_name="evaluation_metrics_export.json", mime="application/json")

    with eval_financial_tab:
        st.subheader("Financial evaluation")
        left, right = st.columns(2)
        with left:
            fin_df = pd.DataFrame([
                {"Scenario": "No fraud management", "Value": metrics.get("no_fraud_value", None)},
                {"Scenario": f"{_model_short} + RL", "Value": metrics.get("model_policy_value", -total_cost)},
                {"Scenario": "Oracle", "Value": metrics.get("oracle_value", 0.0)},
            ])
            if fin_df["Value"].isna().any():
                flm = float(policy.get("fraud_loss_multiplier", 3.0))
                if "isFraud" in transactions_df.columns:
                    no_fraud_value = -float((transactions_df["isFraud"].fillna(0) * flm * transactions_df["TransactionAmt"].fillna(0)).sum())
                else:
                    no_fraud_value = -total_cost
                fin_df["Value"] = [no_fraud_value, -total_cost, 0.0]
            fig = px.bar(fin_df, x="Scenario", y="Value", title="Financial Outcome Comparison", text=fin_df["Value"].map(fmt_money), color="Scenario", color_discrete_sequence=CHART_COLORS)
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            show_fig(fig)
        with right:
            st.markdown("""
<div class="section-note">
<b>Financial interpretation:</b> this tab focuses on business cost, protected value, and the value gained compared with weaker policies. Model classification quality is separated into the Model Performance tab.
</div>
""", unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            cost_df = pd.DataFrame([
                {"Cost type": "Missed fraud", "Cost": float(metrics.get("missed_fraud_cost_total", transactions_df.get("missed_fraud_cost", pd.Series([0])).sum()))},
                {"Cost type": "False rejection", "Cost": float(metrics.get("false_rejection_cost_total", transactions_df.get("false_rejection_cost", pd.Series([0])).sum()))},
                {"Cost type": "Manual review", "Cost": float(metrics.get("manual_review_cost_total", transactions_df.get("manual_review_cost", pd.Series([0])).sum()))},
                {"Cost type": "Overload", "Cost": float(metrics.get("overload_cost_total", transactions_df.get("overload_cost", pd.Series([0])).sum()))},
            ])
            fig = px.bar(cost_df, x="Cost type", y="Cost", title="Cost Breakdown", text=cost_df["Cost"].map(fmt_money), color="Cost type", color_discrete_map=COST_COLORS)
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            show_fig(fig)
        with right:
            detected_fraud = metrics.get("tp")
            missed_fraud = metrics.get("fn")
            if detected_fraud is None or missed_fraud is None:
                if "isFraud" in transactions_df.columns:
                    detected_fraud = int(((transactions_df["isFraud"] == 1) & (transactions_df["rl_action"].isin(["review", "reject"]))).sum())
                    missed_fraud = int(((transactions_df["isFraud"] == 1) & (transactions_df["rl_action"] == "accept")).sum())
                else:
                    detected_fraud = 0
                    missed_fraud = 0
            fraud_result_df = pd.DataFrame([
                {"Result": "Detected fraud", "Count": int(detected_fraud)},
                {"Result": "Missed fraud", "Count": int(missed_fraud)},
            ])
            fig = px.bar(fraud_result_df, x="Result", y="Count", title="Detected Fraud vs Missed Fraud", text="Count", color="Result", color_discrete_map=FRAUD_RESULT_COLORS)
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            show_fig(fig)

        left, right = st.columns(2)
        with left:
            if "isFraud" in transactions_df.columns and "TransactionAmt" in transactions_df.columns:
                detected_fraud_amount = float(transactions_df.loc[
                    (transactions_df["isFraud"] == 1) & (transactions_df["rl_action"].isin(["review", "reject"])),
                    "TransactionAmt"
                ].sum())
                missed_fraud_amount = float(transactions_df.loc[
                    (transactions_df["isFraud"] == 1) & (transactions_df["rl_action"] == "accept"),
                    "TransactionAmt"
                ].sum())
            else:
                detected_fraud_amount = float(metrics.get("protected_exposure", 0.0))
                flm = float(policy.get("fraud_loss_multiplier", 3.0) or 3.0)
                missed_fraud_amount = float(metrics.get("missed_fraud_cost_total", 0.0)) / max(flm, 1e-9)

            fraud_amount_df = pd.DataFrame([
                {"Result": "Detected fraud", "Amount": detected_fraud_amount},
                {"Result": "Missed fraud", "Amount": missed_fraud_amount},
            ])
            fig = px.bar(
                fraud_amount_df,
                x="Result",
                y="Amount",
                title="Detected Fraud vs Missed Fraud Amount",
                text=fraud_amount_df["Amount"].map(fmt_money),
                color="Result",
                color_discrete_map=FRAUD_RESULT_COLORS,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="Amount ($)")
            show_fig(fig)
        with right:
            st.markdown("""
            <div class="section-note">
            <b>Amount view:</b> this chart uses dollars instead of counts, so it shows the transaction value of detected fraud compared with fraud value that was still missed.
            </div>
            """, unsafe_allow_html=True)


    with eval_policy_tab:
        st.subheader("Policy comparison")
        if policy_comparison is not None and not policy_comparison.empty:
            pc = policy_comparison.copy()
            for c in ["T_low", "T_high", "review_rate", "total_cost", "profit_gain", "protected_exposure"]:
                if c in pc.columns:
                    pc[c] = pd.to_numeric(pc[c], errors="coerce")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(pc, x="policy_name", y="total_cost", title="Grid Search vs RL vs Deployed: Total Cost", text=pc["total_cost"].map(fmt_money), color="policy_name", color_discrete_sequence=CHART_COLORS)
                fig.update_traces(textposition="outside")
                fig.update_xaxes(title_text="")
                show_fig(fig)
            with c2:
                fig = px.bar(pc, x="policy_name", y="profit_gain", title="Grid Search vs RL vs Deployed: Profit Gain", text=pc["profit_gain"].map(fmt_pct), color="policy_name", color_discrete_sequence=CHART_COLORS)
                fig.update_traces(textposition="outside")
                fig.update_xaxes(title_text="")
                fig.update_yaxes(tickformat=".0%")
                fig.update_layout(showlegend=False)
                show_fig(fig)

        else:
            st.warning("`policy_comparison.csv` was not found. Add it to show grid-search vs RL threshold comparison.")

        with st.expander("Full labeled transaction output table", expanded=False):
            render_table(transactions_df, max_rows=500, height_px=520)
