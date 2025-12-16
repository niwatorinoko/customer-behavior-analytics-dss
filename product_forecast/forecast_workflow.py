import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import BytesIO


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Loads CSV data, attempting UTF-8 SIG first, then falling back to Shift-JIS."""
    try:
        return pd.read_csv(BytesIO(file_bytes), encoding="utf_8_sig")
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(file_bytes), encoding="shift_jis", errors="ignore")


@st.cache_data
def run_forecast_model(df: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
    """
    Performs sales forecasting using a simple Linear Regression model for each product.
    """
    results = []

    for product in df["Product"].unique():
        subset = df[df["Product"] == product].copy()
        # Daily sales count
        subset = subset.groupby("Date").size().reset_index(name="SalesCount")

        # Skip if data is insufficient for modeling
        if subset.empty or len(subset) < 5:
            continue

        # Feature engineering: Convert date to a sequential day index
        subset["DayIndex"] = (subset["Date"] - subset["Date"].min()).dt.days
        X = subset[["DayIndex"]]
        y = subset["SalesCount"]

        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(X, y)

        # Evaluate model performance
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        # Predict future sales
        last_idx = subset["DayIndex"].max()
        future_idx = np.array([[last_idx + days_ahead]])
        future_sales = model.predict(future_idx)[0]

        results.append({
            "Product": product,
            "TotalSales": int(y.sum()),
            "PredictedSales": round(float(future_sales), 2),
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R²": round(float(r2), 2)
        })

    if not results:
        return pd.DataFrame(columns=["Product", "TotalSales", "PredictedSales", "MAE", "RMSE", "R²"])

    return pd.DataFrame(results).sort_values("TotalSales", ascending=False)


def run_forecast_tab():
    st.header("📦 Product Sales Forecasting (Step-by-step Workflow)")

    # ① Data Preprocessing
    st.subheader("① Data Preprocessing")

    uploaded_file = st.file_uploader("Upload Sales Data CSV", type="csv")

    if not uploaded_file:
        st.info("Upload a CSV to proceed to the next step.")
        # Clear session states if no file is present
        st.session_state.pop("product_summary", None)
        st.session_state["forecast_done"] = False
        st.session_state["product_ready"] = False
        return

    # Use the unified load_csv function
    df = load_csv(uploaded_file.getvalue())
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    if "Date" not in df.columns or "Product" not in df.columns:
        st.error("❌ The 'Date' and 'Product' columns are required.")
        return

    # Force alignment of timezones (remove timezone info for simplicity)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)

    if df["Date"].isna().sum() > 0:
        st.warning("⚠️ Rows with invalid dates will be excluded.")
        df = df.dropna(subset=["Date"])

    st.success("Data preprocessing complete!")

    # ② Set Analysis Period
    st.subheader("② Set Analysis Period")

    if df.empty:
        st.error("❌ Dataframe is empty after preprocessing.")
        return

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    start_date = st.date_input("Select Start Date", min_date)
    end_date = st.date_input("Select End Date", max_date)

    if start_date > end_date:
        st.error("❌ The start date must be before the end date.")
        return

    # Filter data for the selected period
    df_period = df[
        (df["Date"] >= pd.to_datetime(start_date)) &
        (df["Date"] <= pd.to_datetime(end_date))
    ]

    st.write(f"Data points within the period: {len(df_period)}")
    st.dataframe(df_period.head())

    if df_period.empty:
        st.warning("⚠️ No data available for the specified period.")
        return

    st.success("Period setting complete!")

    # ③ Aggregate Sales within the Period
    st.subheader("③ Aggregate Sales within the Period")
    grouped = df_period.groupby("Product").size().reset_index(name="SalesCount")
    grouped = grouped.sort_values("SalesCount", ascending=False)
    st.dataframe(grouped)

    if grouped.empty:
        st.warning("⚠️ No aggregation results.")
        return

    st.success("Sales aggregation complete!")

    # ④ Set Forecast Horizon
    st.subheader("④ How many days ahead to forecast?")
    days_ahead = st.number_input(
        "Enter the number of days to forecast (e.g., 7)",
        min_value=1,
        max_value=180,
        value=7,
        step=1
    )
    st.success(f"Creating a forecast for the next {days_ahead} days.")

    # ⑤ Sales Forecasting
    st.subheader("⑤ Sales Forecasting (Machine Learning Model)")
    with st.spinner("Training model and generating forecast..."):
        forecast_df = run_forecast_model(df_period, days_ahead)

    st.dataframe(forecast_df)

    st.success("Sales forecast complete!")
    st.session_state["product_summary"] = forecast_df
    st.session_state["forecast_done"] = True
    st.session_state["product_ready"] = True

    if not st.session_state.get("rerun_triggered_forecast", False):
        st.session_state["rerun_triggered_forecast"] = True
        st.rerun()
