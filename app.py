import json
import streamlit as st
from datetime import datetime

from core import (
    GLOBAL_PARAMS, PARAM_PROFILES, _observer,
    run_personas, run_usp, run_competitive, run_message_house, run_retention, run_evidence,
    export_excel
)

# Track page load
_observer.log_custom_metric("PageLoads", 1, "Count")

st.set_page_config(page_title="Zero‑Data Persona & Messaging Studio", page_icon="🧠", layout="wide")
st.title("🧠 Zero‑Data Persona & Messaging Studio")
st.caption("Craft narrative personas, sharp USPs, and a message house — no historical data required.")

# Initialize metrics
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'persona_runs': 0, 'usp_runs': 0, 'message_runs': 0, 'evidence_runs': 0,
        'total_tokens': 0, 'errors': [], 'session_start': datetime.now()
    }
    # Log session start
    _observer.log_event("session_start", {"timestamp": datetime.now().isoformat()})
    _observer.log_custom_metric("SessionStarts", 1, "Count")

with st.sidebar:
    st.header("📊 Usage Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Persona Generations", st.session_state.metrics['persona_runs'])
        st.metric("USP Generations", st.session_state.metrics['usp_runs'])
    with col2:
        st.metric("Message Houses", st.session_state.metrics['message_runs'])
        st.metric("Evidence Runs", st.session_state.metrics['evidence_runs'])
    
    if st.session_state.metrics['errors']:
        st.error(f"Errors: {len(st.session_state.metrics['errors'])}")
        if st.button("View Errors"):
            for error in st.session_state.metrics['errors'][-3:]:
                st.text(f"{error['time']}: {error['error'][:100]}...")
    
    # Session info
    session_duration = datetime.now() - st.session_state.metrics['session_start']
    st.caption(f"Session: {session_duration.seconds//60}m {session_duration.seconds%60}s")
    
    st.markdown("---")
    st.header("Global model params")
    g_temp = st.slider("temperature", 0.0, 1.0, float(GLOBAL_PARAMS["temperature"]), 0.05)
    g_top_p = st.slider("top_p", 0.5, 1.0, float(GLOBAL_PARAMS["top_p"]), 0.01)
    g_top_k = st.slider("top_k", 1, 500, int(GLOBAL_PARAMS["top_k"]), 1)
    g_max_tokens = st.slider("max_tokens", 256, 4000, int(GLOBAL_PARAMS["max_tokens"]), 64)
    # Live update globals for downstream merges
    GLOBAL_PARAMS["temperature"] = g_temp
    GLOBAL_PARAMS["top_p"] = g_top_p
    GLOBAL_PARAMS["top_k"] = g_top_k
    GLOBAL_PARAMS["max_tokens"] = g_max_tokens

    st.markdown("---")
    st.header("Per-module overrides")
    module_choice = st.selectbox("Module", ["persona", "usp", "comp", "message", "retention", "evidence"])
    mp = PARAM_PROFILES[module_choice]
    mp["temperature"] = st.slider(f"{module_choice}: temperature", 0.0, 1.0, float(mp["temperature"]), 0.05)
    mp["top_p"] = st.slider(f"{module_choice}: top_p", 0.5, 1.0, float(mp["top_p"]), 0.01)
    mp["top_k"] = st.slider(f"{module_choice}: top_k", 1, 500, int(mp["top_k"]), 1)
    mp["max_tokens"] = st.slider(f"{module_choice}: max_tokens", 256, 4000, int(mp["max_tokens"]), 64)
    st.info("Params update live. You can switch modules to override another profile.")

st.subheader("1) Business input")
col1, col2 = st.columns(2)
with col1:
    business_desc = st.text_area(
        "Business description",
        placeholder="e.g., Privacy-first AI CRM for freelancers; automate follow-ups, inbox triage, and proposal reminders.",
        height=140
    )
    categories = st.tags_input("Product/Service categories", ["CRM", "Freelancers", "Productivity"]) if hasattr(st, "tags_input") else \
        st.text_input("Product/Service categories (comma-separated)", "CRM, Freelancers, Productivity")
with col2:
    regions = st.text_input("Target regions (comma-separated)", "India, SEA")
    tone = st.selectbox("Preferred tone", ["practical", "friendly", "bold", "premium"])

st.subheader("2) Optional public URLs for evidence")
urls_raw = st.text_area(
    "Public URLs (one per line) — your site, competitor pages, relevant articles",
    placeholder="https://example.com\nhttps://competitor.com/pricing\nhttps://news.site/category-trend"
)
url_list = [u.strip() for u in urls_raw.splitlines() if u.strip()]

st.subheader("3) Product inputs for USPs")
col3, col4 = st.columns(2)
with col3:
    features_raw = st.text_area("Key features (one per line)", "Automated inbox triage\nProposal templates\nFollow-up scheduling")
with col4:
    benefits_raw = st.text_area("Key benefits (one per line)", "Save 5+ hours/week\nClose more proposals\nNever miss a follow-up")
category_norms_raw = st.text_area("Category norms to respect/avoid (optional)", "Must be privacy-first\nAvoid 'all-in-one' cliché")

def to_list(raw: str) -> list:
    return [x.strip() for x in raw.splitlines() if x.strip()]

# Run buttons
run_cols = st.columns(5)
do_evidence = run_cols[0].button("Run Evidence")
do_personas = run_cols[1].button("Generate Personas")
do_usp = run_cols[2].button("Distill USPs")
do_message = run_cols[3].button("Build Message House")
do_export = run_cols[4].button("Export Excel")

# Session state for artifacts
if "artifacts" not in st.session_state:
    st.session_state["artifacts"] = {
        "evidence": None,
        "personas": None,
        "usp": None,
        "competitive": None,
        "message_house": None,
        "retention": None,
    }

# Evidence
if do_evidence and url_list:
    with st.spinner("Fetching and summarizing public evidence..."):
        try:
            ev = run_evidence(
                business_desc=business_desc,
                categories=[c.strip() for c in (categories if isinstance(categories, list) else categories.split(",")) if c.strip()],
                urls=url_list
            )
            st.session_state["artifacts"]["evidence"] = ev
            st.session_state.metrics['evidence_runs'] += 1
            _observer.log_event("user_action", {"action": "generate_evidence", "urls_count": len(url_list)})
            _observer.log_custom_metric("EvidenceGenerations", 1, "Count", {"URLCount": str(len(url_list))})
            st.success("Evidence summarized.")
            st.json(ev)
        except Exception as e:
            st.session_state.metrics['errors'].append({"time": datetime.now().isoformat(), "error": str(e)})
            st.error(f"Error generating evidence: {str(e)}")

# Personas
if do_personas:
    with st.spinner("Generating narrative personas..."):
        try:
            ps = run_personas(
                business_desc=business_desc,
                categories=[c.strip() for c in (categories if isinstance(categories, list) else categories.split(",")) if c.strip()],
                regions=[r.strip() for r in regions.split(",") if r.strip()],
                tone=tone
            )
            st.session_state["artifacts"]["personas"] = ps
            st.session_state.metrics['persona_runs'] += 1
            _observer.log_event("user_action", {"action": "generate_personas", "tone": tone})
            _observer.log_custom_metric("PersonaGenerations", 1, "Count", {"Tone": tone})
            st.success("Personas ready.")
            st.json(ps)
        except Exception as e:
            st.session_state.metrics['errors'].append({"time": datetime.now().isoformat(), "error": str(e)})
            st.error(f"Error generating personas: {str(e)}")

# USPs
if do_usp:
    with st.spinner("Distilling USP grid..."):
        try:
            usp = run_usp(
                features=to_list(features_raw),
                benefits=to_list(benefits_raw),
                category_norms=to_list(category_norms_raw) if category_norms_raw.strip() else None
            )
            st.session_state["artifacts"]["usp"] = usp
            st.session_state.metrics['usp_runs'] += 1
            _observer.log_event("user_action", {"action": "generate_usp", "features_count": len(to_list(features_raw))})
            _observer.log_custom_metric("USPGenerations", 1, "Count", {"FeatureCount": str(len(to_list(features_raw)))})
            st.success("USP grid ready.")
            st.json(usp)
        except Exception as e:
            st.session_state.metrics['errors'].append({"time": datetime.now().isoformat(), "error": str(e)})
            st.error(f"Error generating USP: {str(e)}")

# Message house (requires personas + usp)
if do_message:
    personas = st.session_state["artifacts"]["personas"]
    usp = st.session_state["artifacts"]["usp"]
    if not personas or not usp:
        st.error("Please generate Personas and USPs first.")
    else:
        with st.spinner("Synthesizing message house and retention plan..."):
            try:
                mh = run_message_house(personas, usp)
                st.session_state["artifacts"]["message_house"] = mh

                # Build simple objection themes for retention
                objection_themes = []
                if isinstance(personas, list):
                    for p in personas:
                        obs = p.get("objections", [])
                        if isinstance(obs, list):
                            objection_themes.extend(obs[:2])
                objection_themes = list({o for o in objection_themes})

                ret = run_retention(personas, objection_themes=objection_themes)
                st.session_state["artifacts"]["retention"] = ret
                st.session_state.metrics['message_runs'] += 1
                _observer.log_event("user_action", {"action": "generate_message_house", "personas_count": len(personas)})
                _observer.log_custom_metric("MessageHouseGenerations", 1, "Count", {"PersonaCount": str(len(personas))})

                st.success("Message house and retention plan ready.")
                st.subheader("Message House")
                st.json(mh)
                st.subheader("Retention Plan")
                st.json(ret)
            except Exception as e:
                st.session_state.metrics['errors'].append({"time": datetime.now().isoformat(), "error": str(e)})
                st.error(f"Error generating message house: {str(e)}")

# Export
if do_export:
    arts = st.session_state["artifacts"]
    if not (arts["personas"] and arts["usp"] and arts["message_house"]):
        st.error("Need Personas, USPs, and Message House to export. Generate those first.")
    else:
        try:
            comp = arts["evidence"] if arts["evidence"] else {"signals": [], "implications": [], "conflicts": []}
            bytes_xlsx = export_excel(
                personas=arts["personas"],
                usp=arts["usp"],
                comp=comp,
                message_house=arts["message_house"],
                retention=arts["retention"] or {}
            )
            _observer.log_event("user_action", {"action": "export_excel"})
            _observer.log_custom_metric("ExcelExports", 1, "Count")
            st.download_button(
                label="Download Excel (DAX‑ready)",
                data=bytes_xlsx,
                file_name="zero_data_persona_suite.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.session_state.metrics['errors'].append({"time": datetime.now().isoformat(), "error": str(e)})
            st.error(f"Error exporting: {str(e)}")

st.markdown("---")
with st.expander("📊 View Usage Analytics"):
    if st.button("Generate Usage Report"):
        try:
            from metrics_analyzer import MetricsAnalyzer
            analyzer = MetricsAnalyzer()
            report = analyzer.generate_report()
            st.markdown(report)
        except Exception as e:
            st.error(f"Could not generate report: {e}")
    
    if st.button("Download Metrics Log"):
        try:
            with open("app_metrics.jsonl", "r") as f:
                st.download_button(
                    label="Download app_metrics.jsonl",
                    data=f.read(),
                    file_name="app_metrics.jsonl",
                    mime="application/json"
                )
        except:
            st.error("No metrics log found")

st.caption("Tip: Tune module overrides in the sidebar (temperature, top_p, top_k, max_tokens) to balance creativity and precision per task.")
st.caption("📊 Observability: Metrics are logged locally and sent to CloudWatch (if configured)")
