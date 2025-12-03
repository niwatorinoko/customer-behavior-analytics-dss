import streamlit as st
from customer_segmentation.segmentation_workflow import run_segmentation_tab
from product_forecast.forecast_workflow import run_forecast_tab
from report_generator import generate_llm_report, export_report_to_pdf

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
            report_text = generate_llm_report(data_summary, mode=mode)
            st.markdown("### 📄 生成されたレポート")
            st.write(report_text)

            # PDF出力
            pdf_path = export_report_to_pdf(report_text)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 PDFをダウンロード",
                    data=f,
                    file_name="marketing_report.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(str(e))



tab1, tab2 = st.tabs(["🧍‍♂️ 顧客セグメンテーション", "📦 商品販売予測"])

# ============================================================
# 🧍‍♂️ タブ①：顧客セグメンテーション（既存部分）
# ============================================================

with tab1:
    try:
        run_segmentation_tab()
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

