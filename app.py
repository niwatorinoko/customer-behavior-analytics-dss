import streamlit as st
import pandas as pd
import os

# 顧客分析モジュール
from customer_segmentation.data_preprocessing import preprocess_retail_data
from customer_segmentation.rfm import calculate_rfm
from customer_segmentation.clustering import cluster_rfm
from report_generator import generate_llm_report

# 商品予測モジュール（新規）
from product_forecast.forecast_workflow import run_forecast_tab

# ===============================================
# メインアプリ
# ===============================================

st.set_page_config(page_title="Marketing DSS", layout="wide")
st.title("📊 Marketing Decision Support System")

st.sidebar.title("🧠 レポート生成設定")

# チェック可能状態の判定
customer_ready = "rfm_clustered" in st.session_state
product_ready = "product_summary" in st.session_state

# チェックボックス
use_customer = st.sidebar.checkbox(
    "顧客セグメンテーション結果を使用", 
    value=False, 
    disabled=not customer_ready
)

use_product = st.sidebar.checkbox(
    "商品販売予測結果を使用", 
    value=False, 
    disabled=not product_ready
)

# レポート生成ボタン
if st.sidebar.button("📄 レポートを生成する"):
    if not use_customer and not use_product:
        st.sidebar.warning("少なくとも1つ選択してください。")
    else:
        st.sidebar.info("Geminiによるレポート生成中...")

        # データ準備
        data_summary = {}
        if use_customer:
            data_summary["rfm"] = st.session_state.get("cluster_means")
        if use_product:
            data_summary["forecast"] = st.session_state.get("product_summary")

        # レポート種別判定
        if use_customer and use_product:
            mode = "combined"
        elif use_customer:
            mode = "customer"
        else:
            mode = "product"

        try:
            report = generate_llm_report(data_summary, mode=mode)
            st.subheader("📄 生成されたレポート")
            st.write(report)
        except Exception as e:
            st.error(f"レポート生成に失敗しました: {e}")


tab1, tab2 = st.tabs(["🧍‍♂️ 顧客セグメンテーション", "📦 商品販売予測"])

# ============================================================
# 🧍‍♂️ タブ①：顧客セグメンテーション（既存部分）
# ============================================================

with tab1:
    try:
        st.header("🧍‍♂️ 顧客セグメンテーション（RFM分析）")
        uploaded_file = st.file_uploader("顧客データCSVをアップロード", type="csv", key="customer")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("① アップロードしたデータ")
            st.dataframe(df.head())

            # 前処理
            df_clean = preprocess_retail_data(df)
            st.subheader("② 前処理後のデータ")
            st.dataframe(df_clean.head())

            # RFM計算
            rfm = calculate_rfm(df_clean)
            st.subheader("③ RFMテーブル")
            st.dataframe(rfm.head())

            # クラスタリング
            k = st.slider("クラスタ数 (k)", 2, 10, 4)
            rfm_clustered, model = cluster_rfm(rfm, k)

            st.subheader("④ クラスタ結果")
            st.dataframe(rfm_clustered.head())

            # クラスタ平均
            st.subheader("⑤ クラスタ別平均")
            cluster_means = rfm_clustered.groupby("Cluster").mean()
            st.dataframe(cluster_means)

            # LLMレポート生成
            st.subheader("⑥ 自動レポート生成（LLM）")

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.info("LLMレポートを使うには、環境変数 GEMINI_API_KEY を設定してください（.env に記載して実行）。")
            elif st.button("レポートを生成する"):
                with st.spinner("レポート生成中..."):
                    try:
                        report_text = generate_llm_report(cluster_means)
                        st.markdown("### 📄 生成されたレポート")
                        st.write(report_text)
                    except Exception as e:
                        st.error(f"レポート生成中にエラーが発生しました: {e}")

            st.success("顧客分析が完了しました！")
            # RFM・クラスタ結果をセッションに保存
            st.session_state["rfm_clustered"] = rfm_clustered
            st.session_state["cluster_means"] = cluster_means

    except Exception as e:
        st.error(f"顧客分析中にエラーが発生しました: {e}")

# ============================================================
# 📦 タブ②：商品販売予測（新規追加）
# ============================================================

with tab2:
    try:
        run_forecast_tab()
    except Exception as e:
        st.error(f"商品販売予測中にエラーが発生しました: {e}")

