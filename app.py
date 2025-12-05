import streamlit as st
from customer_segmentation.segmentation_workflow import run_segmentation_tab
from product_forecast.forecast_workflow import run_forecast_tab
from report_generator import generate_llm_report, export_report_to_pdf
import base64

st.set_page_config(page_title="Marketing DSS", layout="wide")
st.title("📊 Marketing Decision Support System")


# ============================================================
# サイドバー：レポート生成設定
# ============================================================
st.sidebar.title("Generate Report Settings")

# チェック可能状態の判定
customer_ready = st.session_state.get("rfm_done", False)
product_ready = st.session_state.get("forecast_done", False)

# チェックボックス
use_customer = st.sidebar.checkbox(
    "顧客セグメンテーション結果を使用",
    value=st.session_state.get("use_customer", False),
    disabled=not customer_ready,
    key="use_customer"
)

use_product = st.sidebar.checkbox(
    "商品販売予測結果を使用",
    value=st.session_state.get("use_product", False),
    disabled=not product_ready,
    key="use_product"
)

# レポート生成ボタン
if st.sidebar.button("Generate Report"):
    if not use_customer and not use_product:
        st.sidebar.warning("少なくとも1つ選択してください。")
    else:
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
            with st.spinner("Generating report..."):
                report_text = generate_llm_report(data_summary, mode=mode)
                
            with st.spinner("Exporting to PDF..."):
                pdf_path = export_report_to_pdf(report_text)

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            b64 = st.session_state.get("pdf_b64", None)
            if not b64:
                b64 = base64.b64encode(pdf_bytes).decode()
                st.session_state["pdf_b64"] = b64

            href = f'<a href="data:application/pdf;base64,{b64}" download="marketing_report.pdf">📥 Click here to download your report automatically</a>'

            st.sidebar.success("Report generated successfully!")
            st.sidebar.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(str(e))

tab1, tab2 = st.tabs(["🧍‍♂️ 顧客セグメンテーション", "📦 商品販売予測"])

# ============================================================
# タブ①：顧客セグメンテーション
# ============================================================

with tab1:
    try:
        run_segmentation_tab()
    except Exception as e:
        st.error(f"顧客分析中にエラーが発生しました: {e}")

# ============================================================
# タブ②：商品販売予測
# ============================================================

with tab2:
    try:
        run_forecast_tab()
    except Exception as e:
        st.error(f"商品販売予測中にエラーが発生しました: {e}")

