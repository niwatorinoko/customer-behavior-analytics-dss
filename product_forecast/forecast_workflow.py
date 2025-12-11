import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    try:
        return pd.read_csv(BytesIO(file_bytes), encoding="utf_8_sig")
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(file_bytes), encoding="shift_jis", errors="ignore")


@st.cache_data
def run_forecast_model(df: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
    results = []

    for product in df["Product"].unique():
        subset = df[df["Product"] == product].copy()
        subset = subset.groupby("Date").size().reset_index(name="SalesCount")

        if subset.empty or len(subset) < 5:
            continue

        subset["DayIndex"] = (subset["Date"] - subset["Date"].min()).dt.days
        X = subset[["DayIndex"]]
        y = subset["SalesCount"]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

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
    st.header("📦 商品販売予測（ステップ式ワークフロー・タブなし）")

    # ① データの整形
    st.subheader("① データの整形")

    uploaded_file = st.file_uploader("販売データCSVをアップロード", type="csv")

    if not uploaded_file:
        st.info("CSV をアップロードすると次のステップが表示されます。")
        return

    df = load_csv(uploaded_file.getvalue())
    st.write("アップロードされたデータ：")
    st.dataframe(df.head())

    if "Date" not in df.columns or "Product" not in df.columns:
        st.error("❌ 'Date' および 'Product' カラムが必要です。")
        return

    # 🔴 ここを修正：タイムゾーン付き／なしを強制的に揃える
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)

    if df["Date"].isna().sum() > 0:
        st.warning("⚠️ 無効な日付がある行は除外されます。")
        df = df.dropna(subset=["Date"])

    st.success("データ整形完了！")

    # ② 期間を設定
    st.subheader("② 分析する期間を設定")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    start_date = st.date_input("開始日を選択", min_date)
    end_date = st.date_input("終了日を選択", max_date)

    if start_date > end_date:
        st.error("❌ 開始日は終了日より前にしてください。")
        return

    # ここで比較しても、df["Date"] は tz なし、start/end も tz なしなので OK
    df_period = df[
        (df["Date"] >= pd.to_datetime(start_date)) &
        (df["Date"] <= pd.to_datetime(end_date))
    ]

    st.write(f"期間内データ数：{len(df_period)}")
    st.dataframe(df_period.head())

    if df_period.empty:
        st.warning("⚠️ 指定期間にデータがありません。")
        return

    st.success("期間設定完了！")

    # ③ 指定期間内の売上集計
    st.subheader("③ 指定期間内の売上集計")
    grouped = df_period.groupby("Product").size().reset_index(name="SalesCount")
    grouped = grouped.sort_values("SalesCount", ascending=False)
    st.dataframe(grouped)

    if grouped.empty:
        st.warning("⚠️ 集計結果がありません。")
        return

    st.success("売上集計完了！")

    # ④ 予測したい先の日数を設定
    st.subheader("④ 何日先を予測しますか？")
    days_ahead = st.number_input(
        "予測したい日数を入力してください（例：7）",
        min_value=1,
        max_value=180,
        value=7,
        step=1
    )
    st.success(f"{days_ahead} 日先の予測を作成します。")

    # ⑤ 販売予測
    st.subheader("⑤ 販売予測（機械学習モデル）")
    with st.spinner("モデルを学習し、予測を生成しています…"):
        forecast_df = run_forecast_model(df_period, days_ahead)

    st.dataframe(forecast_df)

    st.session_state["product_summary"] = forecast_df
    st.session_state["forecast_done"] = True
    st.session_state["product_ready"] = True

    st.success("✨ 販売予測が完了しました！")
