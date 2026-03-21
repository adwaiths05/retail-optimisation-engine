import os

import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report
from sqlalchemy import create_engine, text

from src.core.config import settings


def load_reference_window(sample_size: int = 4000) -> pd.DataFrame:
    ref_df = pd.read_csv(
        "./data/raw/order_products__prior.csv",
        usecols=["product_id", "reordered", "add_to_cart_order"],
    ).head(sample_size)
    ref_df["event_value"] = 1.0
    ref_df["event_hour"] = 12
    return ref_df[["product_id", "reordered", "add_to_cart_order", "event_value", "event_hour"]]


def load_current_window(sample_size: int = 4000) -> pd.DataFrame:
    engine = create_engine(settings.SYNC_DATABASE_URL)
    query = text(
        """
        SELECT
            e.product_id,
            CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END AS reordered,
            CASE
                WHEN e.event_type = 'view' THEN 1
                WHEN e.event_type = 'click' THEN 2
                WHEN e.event_type = 'cart_add' THEN 3
                ELSE 4
            END AS add_to_cart_order,
            CASE
                WHEN e.event_type = 'view' THEN 0.25
                WHEN e.event_type = 'click' THEN 0.5
                WHEN e.event_type = 'cart_add' THEN 0.75
                ELSE 1.0
            END AS event_value,
            EXTRACT(HOUR FROM e.timestamp) AS event_hour
        FROM experiment_events e
        WHERE e.timestamp > NOW() - INTERVAL '7 days'
        ORDER BY e.timestamp DESC
        LIMIT :sample_size
        """
    )
    with engine.begin() as conn:
        curr_df = pd.read_sql(query, conn, params={"sample_size": sample_size})
    return curr_df


def generate_monitoring_report(reference_df: pd.DataFrame | None = None, current_df: pd.DataFrame | None = None):
    reference_df = reference_df if reference_df is not None else load_reference_window()
    current_df = current_df if current_df is not None else load_current_window()

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs("./mlops/reports", exist_ok=True)
    report.save_html("./mlops/reports/drift_report.html")
    print("📊 Drift report generated at ./mlops/reports/drift_report.html")


if __name__ == "__main__":
    generate_monitoring_report()