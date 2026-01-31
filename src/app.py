import streamlit as st
import pandas as pd
import plotly.express as px

from data_simulator import FinancialDataStreamer
from drift_engine import DriftDetector
from llm_explainer import stream_drift_explanation

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML Drift Monitor", layout="wide")
st.title("🛡️ ML Model Drift Monitor")

# ---------------- SESSION STATE INIT ----------------
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.streamer = FinancialDataStreamer(
        "data/PS_20174392719_1491204439457_log.csv"
    )
    st.session_state.detector = DriftDetector()
    st.session_state.drift_history = []
    st.session_state.show_ai_report = False
    st.session_state.last_amount_drift = None

    with st.spinner("Initializing Reference Profile..."):
        ref_batch = st.session_state.streamer.get_next_batch(inject_drift=False)
        st.session_state.detector.set_reference(ref_batch)

    st.success("Reference Profile Locked! 🔒")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Simulation Controls")
inject_drift = st.sidebar.checkbox("💉 Inject Drift (Simulate Failure)", value=False)

process_clicked = st.sidebar.button("▶️ Process Next Batch")

# ---------------- PROCESS NEXT BATCH ----------------
if process_clicked:
    st.session_state.show_ai_report = False  # reset AI view

    current_batch = st.session_state.streamer.get_next_batch(
        inject_drift=inject_drift
    )

    if current_batch is not None:
        report = st.session_state.detector.detect_drift(current_batch)
        amount_drift = report["amount"]

        st.session_state.last_amount_drift = amount_drift

        st.session_state.drift_history.append({
            "batch_index": len(st.session_state.drift_history) + 1,
            "ks_statistic": amount_drift["ks_statistic"],
            "status": "DRIFT" if amount_drift["is_drifting"] else "OK",
        })

    else:
        st.warning("End of Data Stream.")

# ---------------- DASHBOARD ----------------
if st.session_state.last_amount_drift is not None:

    amount_drift = st.session_state.last_amount_drift

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Batch Size", 100000)

    with col2:
        st.metric(
            "Drift Score (KS-Stat)",
            amount_drift["ks_statistic"],
            delta_color="inverse",
        )

    with col3:
        if amount_drift["is_drifting"]:
            st.error("🔴 DRIFT DETECTED")
            if st.button("🤖 Generate AI Analysis Report"):
                st.session_state.show_ai_report = True
        else:
            st.success("🟢 SYSTEM HEALTHY")

# ---------------- AI REPORT (PERSISTENT) ----------------
if st.session_state.show_ai_report and st.session_state.last_amount_drift:

    st.markdown("### 📝 AI Root Cause Analysis")

    with st.chat_message("assistant"):
        st.write_stream(
            stream_drift_explanation({
                "feature": "amount",
                **st.session_state.last_amount_drift
            })
        )

# ---------------- DRIFT CHART ----------------
if st.session_state.drift_history:

    st.subheader("Drift Severity Over Time")

    chart_df = pd.DataFrame(st.session_state.drift_history)

    fig = px.line(
        chart_df,
        x="batch_index",
        y="ks_statistic",
        title="Feature: Amount (KS Statistic)",
        markers=True,
    )

    fig.add_hline(
        y=0.1,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold (0.1)",
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- RAW STATS ----------------
with st.expander("🔎 View Detailed Statistical Report"):
    if st.session_state.last_amount_drift:
        st.json(st.session_state.last_amount_drift)
    else:
        st.info("No batch processed yet.")
