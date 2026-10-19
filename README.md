# Digital-Twin-for-Predictive-Maintenance

A software-based digital twin prototype for bearing condition monitoring using historical vibration data from the NASA IMS Bearing Dataset and machine learning-based anomaly detection.

## Overview

Unexpected bearing failures can lead to equipment downtime, maintenance costs, and reduced operational efficiency. Predictive maintenance approaches analyse equipment condition data to identify unusual behaviour that may require further investigation.

This project explores a data-driven approach to bearing condition monitoring. It processes historical vibration recordings, extracts statistical features, applies anomaly detection methods, and presents the results through an interactive Streamlit dashboard.

**Note:** This is a historical-data-driven prototype. It does not currently use live IoT sensor readings or confirm bearing failures.

## Objectives

- Analyse historical bearing vibration data.
- Extract statistical features that describe vibration behaviour.
- Detect unusual patterns using machine learning.
- Compare Isolation Forest results with a percentile-based baseline.
- Visualise vibration trends, anomaly flags, and monitoring indicators.
- Explore how a digital twin interface can support condition monitoring.

## System Architecture

The project follows this workflow:

1. **Data source:** Historical vibration recordings from the NASA IMS Bearing Dataset.
2. **Feature extraction:** Calculate RMS, standard deviation, peak-to-peak amplitude, and kurtosis for eight vibration channels.
3. **Anomaly detection:** Apply Isolation Forest and a percentile-based baseline to identify unusual recordings.
4. **Results processing:** Store timestamps, anomaly scores, anomaly flags, and selected vibration features.
5. **Dashboard:** Display bearing-state indicators, vibration trends, flagged recordings, method comparisons, and evaluation summaries.

## Technologies Used

- Python
- Pandas
- Streamlit
- Scikit-learn
- Statistical feature extraction
- Machine learning-based anomaly detection

## Anomaly Detection Methods

### Isolation Forest

Isolation Forest identifies observations that differ from the patterns learned from the training data. The model produces anomaly scores and flags recordings according to its decision threshold.

### Percentile-Based Baseline

The baseline compares vibration features with thresholds calculated from the training data. A recording is flagged when one or more monitored features exceed their corresponding thresholds.

The baseline is used as a comparison method. Its sensitivity depends on the selected features and threshold.

## Dashboard Features

- Latest recorded bearing-state summary
- Anomaly status and Isolation Forest score
- Historical vibration trend visualisations
- Date-range filtering
- Anomaly detection timeline
- Comparison of the two anomaly detection methods
- Flagged recording details
- CSV export of monitoring results
- Model limitations and interpretation guidance

### Live Dashboard

https://digital-twin-for-predictive-maintenance-g3nz4zbtjhtguwz6lsigtm.streamlit.app/

Explore historical bearing vibration trends, anomaly detection results, bearing-state indicators, and model evaluation through the interactive dashboard.

## Dataset

The project uses the NASA IMS Bearing Dataset, which contains vibration measurements collected during bearing experiments.

Dataset source: [NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/)

The current dashboard uses a processed results file named `ims_anomaly_results.csv`.

## Installation and Usage

### Prerequisites

- Python 3.10 or a compatible Python version
- pip

### 1. Clone the repository

```bash
git clone https://github.com/harshiniip/Digital-Twin-for-Predictive-Maintenance.git
cd Digital-Twin-for-Predictive-Maintenance
```

### 2. Create a virtual environment 

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Model Evaluation

The dashboard compares the anomaly flags produced by Isolation Forest and the percentile-based baseline.

Anomaly detection results indicate unusual patterns, not confirmed failures. Because verified fault labels are not available for every evaluated recording in the current workflow, the results should not be interpreted as validated predictive accuracy.

## Limitations

- The current implementation analyses historical rather than live sensor data.
- Anomaly flags do not confirm mechanical failure.
- The model's behaviour may be affected by changes in vibration distributions over time.
- The percentile baseline can flag many recordings when multiple features exceed their thresholds.
- Verified labels and further experiments are required for a reliable assessment of model performance.
- The system does not estimate remaining useful life.

## Future Improvements

- Evaluate alternative anomaly detection approaches.
- Investigate frequency-domain vibration features.
- Validate model behaviour using verified bearing-condition labels.
- Investigate persistent anomalies and changes in vibration trends.
- Integrate live sensor data if hardware becomes available.
- Explore remaining useful life estimation after suitable validation.

## Project Scope

This project is a software-based prototype demonstrating a historical-data-driven workflow for bearing condition monitoring. It is intended for educational and experimental purposes, not as a validated industrial safety or maintenance system.

## Acknowledgements

The project uses the NASA IMS Bearing Dataset for experimental analysis.

## License

No licence has been specified yet. Add an appropriate open-source licence if you intend to permit others to reuse or modify this project.
