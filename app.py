"""Streamlit interface. Run: uv run streamlit run app.py"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(page_title="Customer Service Quality Auditor", page_icon="🔎", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_URL}{path}", timeout=50, **kwargs)
    response.raise_for_status()
    return response.json()


def compact(text: str) -> str:
    words = text.split()
    return " ".join(words[:4]) + ("…" if len(words) > 4 else "")


def chart(title: str, values: dict, key: str) -> None:
    frame = pd.DataFrame({key: list(values), "Cases": list(values.values())})
    if frame.empty:
        st.info(f"No completed assessments yet for {title.lower()}.")
    else:
        st.plotly_chart(px.bar(frame, x=key, y="Cases", title=title, color=key), use_container_width=True)


def metrics(summary: dict) -> None:
    items = [
        ("Cases reviewed", summary["reviewed"]), ("High risk", summary["high_risk"]),
        ("Medium risk", summary["medium_risk"]), ("Low risk", summary["low_risk"]),
        ("Possible violations", summary["possible_violations"]), ("Common issue", compact(summary["common_issue"])),
    ]
    for column, (label, value) in zip(st.columns(6), items):
        column.metric(label, value)


@st.fragment(run_every="2s")
def live_status(batch_id: str) -> None:
    try:
        batch = api("GET", f"/batches/{batch_id}")
        done = batch["completed"] + batch["failed"]
        st.progress(done / max(batch["total"], 1), text=f"{done} of {batch['total']} cases complete")
        st.caption(f"Queued: {batch['queued']} · Processing: {batch['processing']} · Failed: {batch['failed']}")
        events = api("GET", f"/batches/{batch_id}/events")
        st.code("\n".join(f"[{e['level'].upper()}] {e['message']}" for e in events[-20:]) or "Waiting for a worker…")
    except requests.RequestException as error:
        st.error(f"Cannot reach FastAPI: {error}")


@st.fragment(run_every="3s")
def live_dashboard() -> None:
    try:
        summary = api("GET", "/dashboard/summary")
        metrics(summary)
        left, right = st.columns(2)
        with left:
            chart("Risk distribution", summary["risk_distribution"], "Risk")
            chart("Agent quality", summary["agent_quality"], "Agent")
        with right:
            chart("Policy status", summary["policy_distribution"], "Status")
            chart("Customer sentiment", summary["sentiment_distribution"], "Sentiment")
    except requests.RequestException as error:
        st.error(f"Cannot load dashboard: {error}")


def case_review() -> None:
    try:
        cases = api("GET", "/cases")
    except requests.RequestException as error:
        st.error(error)
        return
    st.subheader("Review a completed case")
    if not cases:
        st.info("Completed assessments will appear here automatically.")
        return
    frame = pd.DataFrame(cases)
    cols = st.columns(4)
    filters = {
        "risk": cols[0].selectbox("Risk", ["All", *sorted(frame.risk.unique())]),
        "agent": cols[1].selectbox("Agent", ["All", *sorted(frame.agent.unique())]),
        "issue": cols[2].selectbox("Issue", ["All", *sorted(frame.issue.unique())]),
        "policy_status": cols[3].selectbox("Policy status", ["All", *sorted(frame.policy_status.unique())]),
    }
    filtered = frame.copy()
    for field, value in filters.items():
        if value != "All":
            filtered = filtered[filtered[field] == value]
    st.dataframe(filtered[["file", "type", "agent", "issue", "risk", "agent_score", "policy_status"]], hide_index=True, use_container_width=True)
    if filtered.empty:
        st.info("No completed cases match the selected filters.")
        return
    case_id = st.selectbox("Select a case", filtered.case_id.tolist(), format_func=lambda item: filtered.loc[filtered.case_id == item, "file"].iloc[0])
    detail = api("GET", f"/cases/{case_id}")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Conversation / transcription")
        if detail["case"]["type"] == "audio":
            st.text_area("OpenAI transcription", detail["audio"]["transcription"] or "Still processing", height=240, disabled=True)
        else:
            for message in detail["messages"]:
                st.markdown(f"**{message['speaker'].title()}:** {message['text']}")
    with right:
        st.markdown("#### AI assessment")
        st.table(pd.DataFrame((detail["assessment"] or {}).items(), columns=["Field", "Assessment"]))


def dashboard_page() -> None:
    st.title("Customer Service Quality Auditor — Live Dashboard")
    st.caption("Metrics refresh automatically while jobs run.")
    live_dashboard()
    case_review()


def start_page() -> None:
    st.title("Customer Service Quality Auditor")
    st.write("Analyse synthetic chats and audio calls against airline policy using RAG.")
    with st.expander("First-time setup"):
        st.write("Stores 18 chats in PostgreSQL and generates 18 spoken synthetic audio files with metadata.")
        if st.button("Initialize chats and audio"):
            try:
                with st.spinner("Creating database records and spoken audio files…"):
                    st.success(api("POST", "/admin/seed"))
            except requests.RequestException as error:
                st.error(error)
    if st.button("Start Chat Analysis", type="primary", use_container_width=True):
        try:
            batch = api("POST", "/batches")
            st.session_state.batch_id = batch["batch_id"]
            st.success(f"Created {batch['total_jobs']} analysis jobs.")
        except requests.RequestException as error:
            st.error(error)
    if batch_id := st.session_state.get("batch_id"):
        st.subheader("Live processing logs")
        live_status(batch_id)
        st.markdown(f"<a href='/?view=dashboard&batch_id={batch_id}' target='_blank'>View Live Dashboard ↗</a>", unsafe_allow_html=True)


if st.query_params.get("view") == "dashboard":
    dashboard_page()
else:
    start_page()
