import streamlit as st
import pandas as pd

def run_forecast_tab():
    st.header("📦 商品販売予測（試験実装）")
    st.write("日時と商品名（＋数量）のデータから販売傾向を簡易的に分析します。")

    uploaded_file = st.file_uploader("販売データCSVをアップロード", type="csv", key="forecast")

    if not uploaded_file:
        st.info("CSVファイルをアップロードしてください。")
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("① アップロードしたデータ")
    st.dataframe(df.head())

    # 日付形式変換
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 日付が欠損している場合は警告
    if df["Date"].isna().sum() > 0:
        st.warning("⚠️ 一部のDate列に無効な日付があります。")

    # 集計例：商品ごとの売上件数
    st.subheader("② 商品別の販売集計（簡易）")
    if "Product" in df.columns:
        product_summary = df.groupby("Product").size().reset_index(name="SalesCount")
        st.dataframe(product_summary.sort_values("SalesCount", ascending=False))
    else:
        st.error("❌ 'Product' 列が存在しません。CSVのカラム名を確認してください。")
        return

    # 日別トレンド（任意）
    if "Date" in df.columns:
        st.subheader("③ 日別販売数の推移")
        daily_sales = df.groupby("Date").size().reset_index(name="SalesCount")
        st.line_chart(daily_sales.set_index("Date")["SalesCount"])

    st.success("簡易的な販売分析が完了しました。")
    st.session_state["forecast_done"] = True

    # 消さないで！！
    if not st.session_state.get("rerun_triggered", False):
        st.session_state["rerun_triggered"] = True
        st.rerun()