import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
MANIFEST_PATH = ROOT / "reports" / "artifacts" / "data_manifest.json"
VALIDATION_PATH = ROOT / "reports" / "artifacts" / "validation_report.json"
QUALITY_REPORT = ROOT / "reports" / "artifacts" / "data_quality_report.html"
REGISTRY_PATH = ROOT / "reports" / "artifacts" / "registry.jsonl"


st.set_page_config(page_title="EPC South West Dashboard", layout="wide")

st.title("EPC South West Dashboard")
st.caption("Interactive summary of the dissertation pipeline outputs.")

col1, col2, col3 = st.columns(3)

if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text())
    col1.metric("Rows", f"{manifest.get('rows', 0):,}")
    col2.metric("Years", f"{manifest.get('year_min')}–{manifest.get('year_max')}")
    col3.metric("Local Authorities", f"{manifest.get('local_authorities')}")
else:
    col1.warning("Manifest not found. Run `ewhei manifest`.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Policy", "Quality", "Registry"])

with tab1:
    st.subheader("Trends Over Time")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, usecols=["YEAR", "CURRENT_ENERGY_EFFICIENCY", "CO2_EMISSIONS_CURRENT"])
        yearly = df.groupby("YEAR", as_index=False).agg(
            mean_epc=("CURRENT_ENERGY_EFFICIENCY", "mean"),
            mean_co2=("CO2_EMISSIONS_CURRENT", "mean"),
        )
        st.line_chart(yearly, x="YEAR", y=["mean_epc", "mean_co2"])
    else:
        st.info("Data not found. Run `ewhei features`.")

with tab2:
    st.subheader("Policy Period Summary")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, usecols=["POLICY_PERIOD", "CURRENT_ENERGY_EFFICIENCY", "CO2_EMISSIONS_CURRENT", "BELOW_C"])
        summary = df.groupby("POLICY_PERIOD", as_index=False).agg(
            mean_epc=("CURRENT_ENERGY_EFFICIENCY", "mean"),
            mean_co2=("CO2_EMISSIONS_CURRENT", "mean"),
            below_c_share=("BELOW_C", "mean"),
        )
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Data not found. Run `ewhei features`.")

with tab3:
    st.subheader("Data Quality")
    if VALIDATION_PATH.exists():
        validation = json.loads(VALIDATION_PATH.read_text())
        st.json(validation)
    else:
        st.info("Validation report not found. Run `ewhei validate`.")

    if QUALITY_REPORT.exists():
        st.markdown("Open the HTML quality report:")
        st.code(str(QUALITY_REPORT))
    else:
        st.info("Quality report not found. Run `ewhei quality`.")

with tab4:
    st.subheader("Artifact Registry")
    if REGISTRY_PATH.exists():
        rows = [json.loads(line) for line in REGISTRY_PATH.read_text().splitlines() if line.strip()]
        st.dataframe(pd.DataFrame(rows).sort_values("timestamp_utc", ascending=False), use_container_width=True)
    else:
        st.info("Registry not found. Run any pipeline step to create it.")
