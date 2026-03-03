from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd

def generate_monitoring_report(reference_df, current_df):
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])
    
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html("./mlops/reports/drift_report.html")
    print("📊 Drift report generated at ./mlops/reports/drift_report.html")