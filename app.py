
from pathlib import Path

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Digital Twin | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# Dashboard heading
st.title("⚙️ Digital Twin for Predictive Maintenance")

st.markdown("""
Use the sections below to explore the historical bearing data,
anomaly detection results, vibration trends, and monitoring warnings.
""")

st.divider()


st.markdown("""
Monitor bearing vibration patterns using historical sensor data
and unsupervised anomaly detection.

This prototype uses the NASA IMS bearing dataset.
Anomaly flags indicate unusual patterns, not confirmed failures.
""")

# Find the saved results file
PROJECT_FOLDER = Path(__file__).resolve().parent
data_path = PROJECT_FOLDER / "ims_anomaly_results.csv"

# Load the results
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data

if not data_path.exists():
    st.error(f"Results file not found: {data_path}")
    st.stop()

df = load_data(str(data_path))

# Date range filter
st.subheader("Filter by Date")

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

selected_dates = st.date_input(
    "Select the period to analyse",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_df = df[
        (df["timestamp"].dt.date >= start_date) &
        (df["timestamp"].dt.date <= end_date)
    ]
else:
    filtered_df = df.copy()


# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Recordings", f"{len(df):,}")

with col2:
    st.metric(
        "Isolation Forest Flags",
        f"{int(df['isolation_forest_anomaly'].sum()):,}"
    )

with col3:
    st.metric(
        "Percentile Baseline Flags",
        f"{int(df['percentile_baseline_anomaly'].sum()):,}"
    )

# Recent measurements
st.subheader("Recent Recorded Measurements")

st.dataframe(
    df.sort_values("timestamp", ascending=False).head(10),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "These are the latest records in the historical dataset, "
    "not live sensor readings."
)






# Anomaly timeline
st.subheader("Anomaly Detection Timeline")

st.markdown(
    "Explore anomaly detections during the selected period."
)

timeline_data = (
    filtered_df.set_index("timestamp")[
        "isolation_forest_anomaly"
    ]
    .resample("1D")
    .sum()
)

st.line_chart(timeline_data)

st.caption(
    "Daily counts of unusual readings. An anomaly is not a "
    "confirmed mechanical failure."
)





# Vibration trends
st.subheader("Bearing Vibration Trends")

vibration_data = filtered_df.set_index("timestamp")[
    ["ch7_std", "ch6_std", "ch6_rms"]
]

st.line_chart(vibration_data)

st.caption(
    "Vibration features for the selected period."
)



# Summary for the selected period
st.subheader("Selected Period Summary")

total_readings = len(filtered_df)

if total_readings > 0:
    isolation_flags = int(
        filtered_df["isolation_forest_anomaly"].sum()
    )

    baseline_flags = int(
        filtered_df["percentile_baseline_anomaly"].sum()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Readings in Selected Period", total_readings)

    with col2:
        st.metric("Isolation Forest Flags", isolation_flags)

    with col3:
        st.metric("Percentile Baseline Flags", baseline_flags)

    st.caption(
        "Flag counts are method-specific. A flagged reading indicates "
        "an unusual pattern, not a confirmed failure."
    )
else:
    st.warning("No readings are available for the selected dates.")



# Anomaly status indicator
st.subheader("Monitoring Status")

if len(filtered_df) > 0:
    anomaly_rate = (
        filtered_df["isolation_forest_anomaly"].mean() * 100
    )

    st.metric(
        "Flagged Readings (%)",
        f"{anomaly_rate:.2f}%"
    )

    if anomaly_rate == 0:
        st.success(
            "No readings were flagged during the selected period."
        )
    elif anomaly_rate < 10:
        st.info(
            "Some unusual readings were detected. "
            "Continue monitoring the vibration trends."
        )
    else:
        st.warning(
            "A high proportion of readings were flagged. "
            "Review the vibration trends and investigate further."
        )

    st.caption(
        "These are prototype thresholds for displaying model results, "
        "not validated equipment-health limits. Flags alone do not "
        "confirm a mechanical fault."
    )



# Compare anomaly detection methods
st.subheader("Anomaly Detection Method Comparison")

if len(filtered_df) > 0:
    isolation_count = int(
        filtered_df["isolation_forest_anomaly"].sum()
    )

    baseline_count = int(
        filtered_df["percentile_baseline_anomaly"].sum()
    )

    comparison_data = pd.DataFrame({
        "Method": ["Isolation Forest", "Percentile Baseline"],
        "Flagged Readings": [isolation_count, baseline_count]
    })

    st.bar_chart(
        comparison_data.set_index("Method")
    )

    st.caption(
        "The percentile baseline flags a reading if any feature "
        "exceeds its training-set 99th percentile. Different methods "
        "can flag different readings; neither confirms equipment failure."
    )



# Detailed anomaly records
st.subheader("Flagged Recording Details")

flagged_records = filtered_df[
    filtered_df["isolation_forest_anomaly"] == 1
].copy()

if not flagged_records.empty:
    st.write(
        f"Found {len(flagged_records)} flagged recordings "
        "in the selected date range."
    )

    st.dataframe(
        flagged_records[
            [
                "timestamp",
                "isolation_forest_score",
                "ch7_std",
                "ch6_std",
                "ch6_rms"
            ]
        ].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    csv_data = flagged_records.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download flagged recordings as CSV",
        data=csv_data,
        file_name="flagged_recordings.csv",
        mime="text/csv"
    )
else:
    st.info("No flagged recordings in the selected date range.")



# Project methodology
st.subheader("How This Digital Twin Works")

st.markdown("""
The prototype analyses historical bearing vibration data to identify
unusual patterns that may require further investigation.

**Workflow**

1. **Data collection:** Vibration recordings from the NASA IMS bearing dataset.
2. **Feature extraction:** Calculate RMS, standard deviation,
   peak-to-peak amplitude, and kurtosis for eight sensor channels.
3. **Anomaly detection:** Use Isolation Forest to identify unusual
   combinations of vibration features.
4. **Baseline comparison:** Compare the model's results with a
   percentile-based threshold method.
5. **Monitoring:** Explore vibration trends, flagged recordings,
   and results for selected dates.

**Important limitation:** This is a software prototype using historical
data, not a live-connected physical machine. An anomaly indicates an
unusual pattern; it does not establish that a bearing has failed.
""")



# Advanced feature: Rolling vibration trend
st.subheader("Early-Warning Analysis: Rolling Vibration Trend")

if len(filtered_df) > 0:
    trend_data = filtered_df.sort_values("timestamp").copy()

    trend_data["ch7_std_rolling_mean"] = (
        trend_data["ch7_std"]
        .rolling(window=24, min_periods=1)
        .mean()
    )

    chart_data = trend_data.set_index("timestamp")[
        ["ch7_std", "ch7_std_rolling_mean"]
    ]

    st.line_chart(chart_data)

    st.caption(
        "The rolling mean smooths the ch7_std vibration feature "
        "over 24 recordings. It reveals trends but does not predict "
        "the time of bearing failure."
    )
else:
    st.info("No recordings are available for the selected date range.")



# Advanced feature: Baseline-relative warning
st.subheader("Baseline-Relative Warning")

# Training-period reference values from our earlier analysis
TRAINING_MEDIAN_CH7_STD = 0.0802
TRAINING_99TH_PERCENTILE_CH7_STD = 0.0913

if len(filtered_df) > 0:
    warning_data = filtered_df.sort_values("timestamp").copy()

    warning_data["rolling_mean"] = (
        warning_data["ch7_std"]
        .rolling(window=24, min_periods=24)
        .mean()
    )

    latest_rolling_mean = warning_data["rolling_mean"].dropna()

    if not latest_rolling_mean.empty:
        latest_value = float(latest_rolling_mean.iloc[-1])

        st.metric(
            "Latest 24-Recording Rolling Mean",
            f"{latest_value:.4f}",
            delta=(
                f"{(latest_value / TRAINING_MEDIAN_CH7_STD - 1) * 100:.1f}% "
                "vs training median"
            )
        )

        if latest_value > TRAINING_99TH_PERCENTILE_CH7_STD:
            st.warning(
                "The rolling mean exceeds the training-period "
                "99th percentile. Investigate the vibration trend."
            )
        else:
            st.info(
                "The rolling mean is at or below the training-period "
                "99th percentile reference."
            )

        st.caption(
            "This is a reference-based warning, not a validated "
            "failure threshold. It does not estimate remaining useful life."
        )
    else:
        st.info(
            "Select a period containing at least 24 recordings "
            "to calculate the rolling-mean warning."
        )
else:
    st.info("No recordings are available for the selected date range.")



# Advanced feature: Short-term trend direction
st.subheader("Recent Vibration Trend Direction")

if len(filtered_df) >= 48:
    trend_check = filtered_df.sort_values("timestamp").copy()

    recent_mean = trend_check["ch7_std"].tail(24).mean()
    previous_mean = trend_check["ch7_std"].iloc[-48:-24].mean()

    change_pct = (
        (recent_mean - previous_mean) / previous_mean * 100
        if previous_mean != 0
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Previous 24-Recording Mean", f"{previous_mean:.4f}")

    with col2:
        st.metric("Latest 24-Recording Mean", f"{recent_mean:.4f}")

    with col3:
        st.metric("Change Between Windows", f"{change_pct:+.2f}%")

    if change_pct > 5:
        st.warning(
            "Vibration variability is increasing. "
            "Review the recent readings and monitoring history."
        )
    elif change_pct < -5:
        st.info(
            "Vibration variability has decreased compared with "
            "the previous 24 recordings."
        )
    else:
        st.info(
            "Vibration variability is relatively stable between "
            "the two selected windows."
        )

    st.caption(
        "The ±5% bands are exploratory display rules, not validated "
        "engineering thresholds. This comparison describes recent "
        "behaviour; it is not a failure prediction."
    )
else:
    st.info(
        "Select a period containing at least 48 recordings "
        "to compare two consecutive 24-recording windows."
    )



# Advanced feature: Combined monitoring assessment
st.subheader("Combined Monitoring Assessment")

if len(filtered_df) >= 48:
    assessment_data = filtered_df.sort_values("timestamp").copy()

    recent_mean = assessment_data["ch7_std"].tail(24).mean()
    previous_mean = assessment_data["ch7_std"].iloc[-48:-24].mean()

    change_pct = (
        (recent_mean - previous_mean) / previous_mean * 100
        if previous_mean != 0
        else 0
    )

    latest_rolling = (
        assessment_data["ch7_std"]
        .tail(24)
        .mean()
    )

    exceeds_baseline = latest_rolling > 0.0913
    increasing = change_pct > 5

    if exceeds_baseline and increasing:
        st.warning(
            "Elevated and increasing: the recent vibration mean "
            "exceeds the training reference and has increased "
            "compared with the preceding window."
        )
        assessment = "Elevated and increasing"
    elif exceeds_baseline:
        st.warning(
            "Elevated relative to the training reference, but the "
            "latest window is not increasing by more than 5%."
        )
        assessment = "Elevated relative to baseline"
    elif increasing:
        st.info(
            "Increasing trend detected, although the recent mean "
            "does not exceed the training reference."
        )
        assessment = "Increasing trend"
    else:
        st.success(
            "Neither assessment rule is triggered for the latest "
            "two windows."
        )
        assessment = "No combined warning triggered"

    st.write(f"**Assessment:** {assessment}")

    st.caption(
        "This combines an exploratory baseline threshold and a "
        "short-term trend rule. It is not a validated health score "
        "or a prediction of remaining useful life."
    )
else:
    st.info(
        "Select at least 48 recordings to calculate the combined assessment."
    )



# Advanced feature: Historical warning timeline
st.subheader("Historical Early-Warning Timeline")

if len(filtered_df) >= 48:
    history = filtered_df.sort_values("timestamp").copy()

    history["rolling_mean_24"] = (
        history["ch7_std"].rolling(24).mean()
    )

    history["previous_mean_24"] = (
        history["ch7_std"].shift(24).rolling(24).mean()
    )

    history["trend_change_pct"] = (
        (history["rolling_mean_24"] - history["previous_mean_24"])
        / history["previous_mean_24"].replace(0, float("nan"))
        * 100
    )

    history["baseline_exceeded"] = (
        history["rolling_mean_24"] > 0.0913
    )

    history["trend_increasing"] = (
        history["trend_change_pct"] > 5
    )

    history["combined_warning"] = (
        history["baseline_exceeded"]
        & history["trend_increasing"]
    )

    history["warning_value"] = (
        history["combined_warning"].astype(int)
    )

    st.line_chart(
        history.set_index("timestamp")["warning_value"]
    )

    warning_count = int(history["combined_warning"].sum())

    st.metric(
        "Recordings Meeting Both Warning Rules",
        warning_count
    )

    st.caption(
        "1 = both exploratory warning rules are met; 0 = they are not. "
        "This is a retrospective rule-based timeline, not a validated "
        "failure forecast. Rolling windows overlap, so consecutive "
        "flags are not independent events."
    )
else:
    st.info(
        "Select at least 48 recordings to generate the historical "
        "warning timeline."
    )



# Advanced feature: Downloadable monitoring report
st.subheader("Download Monitoring Report")

if len(filtered_df) > 0:
    report = filtered_df.sort_values("timestamp").copy()

    report["rolling_mean_24"] = (
        report["ch7_std"].rolling(24, min_periods=24).mean()
    )

    previous_mean = (
        report["ch7_std"].shift(24).rolling(24, min_periods=24).mean()
    )

    report["trend_change_pct"] = (
        (report["rolling_mean_24"] - previous_mean)
        / previous_mean.replace(0, float("nan"))
        * 100
    )

    report["baseline_exceeded"] = (
        report["rolling_mean_24"] > 0.0913
    )

    report["trend_increasing"] = report["trend_change_pct"] > 5

    report["combined_warning"] = (
        report["baseline_exceeded"]
        & report["trend_increasing"]
    )

    report["monitoring_note"] = (
        "Unusual behaviour indicator; not a confirmed failure"
    )

    report_csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Monitoring Report (CSV)",
        data=report_csv,
        file_name="bearing_monitoring_report.csv",
        mime="text/csv"
    )

    st.caption(
        "The report contains historical measurements and rule-based "
        "indicators for the selected period. Initial rolling-window "
        "values may be blank because earlier readings are needed."
    )
else:
    st.info("No readings are available for the selected date range.")



# Advanced feature: Exploratory anomaly severity classification
st.subheader("Anomaly Severity Classification")

if len(filtered_df) > 0:
    severity_data = filtered_df.copy()

    def classify_severity(row):
        if row["isolation_forest_anomaly"] == 0:
            return "Not flagged"
        elif row["isolation_forest_score"] < -0.10:
            return "Level 3 - Highest priority"
        elif row["isolation_forest_score"] < -0.05:
            return "Level 2 - Review"
        else:
            return "Level 1 - Monitor"

    severity_data["severity"] = severity_data.apply(
        classify_severity, axis=1
    )

    severity_counts = (
        severity_data["severity"]
        .value_counts()
        .reindex(
            [
                "Not flagged",
                "Level 1 - Monitor",
                "Level 2 - Review",
                "Level 3 - Highest priority"
            ],
            fill_value=0
        )
    )

    st.bar_chart(severity_counts)

    st.dataframe(
        severity_data[
            [
                "timestamp",
                "isolation_forest_score",
                "isolation_forest_anomaly",
                "severity"
            ]
        ].sort_values("timestamp", ascending=False).head(20),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Severity levels are illustrative score bands for this prototype, "
        "not validated engineering risk categories. They do not establish "
        "failure probability or remaining useful life."
    )
else:
    st.info("No recordings are available for the selected date range.")



