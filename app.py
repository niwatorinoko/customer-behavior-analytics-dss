import streamlit as st
import pandas as pd
import os

from data_preprocessing import preprocess_retail_data
from rfm import calculate_rfm
from clustering import cluster_rfm
from report_generator import generate_llm_report, to_pdf_bytes

if "report_pdf" not in st.session_state:
    st.session_state["report_pdf"] = None
if "trigger_download" not in st.session_state:
    st.session_state["trigger_download"] = False

st.title("Customer Segmentation DSS")
st.write("CSVをアップロードするだけでRFM分析＋クラスタリングを実行します。")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type="csv")

if uploaded_file:
    # CSV読み込み
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
    elif st.button("生成してPDFをダウンロード"):
        with st.spinner("レポート生成中..."):
            try:
                report_text = generate_llm_report(cluster_means)
                st.markdown("### 📄 生成されたレポート")
                st.write(report_text)
                st.session_state["report_pdf"] = to_pdf_bytes(report_text)
                st.session_state["trigger_download"] = True
            except Exception as e:
                st.error(f"レポート生成中にエラーが発生しました: {e}")

    if st.session_state.get("report_pdf"):
        st.download_button(
            label="PDFをダウンロード",
            data=st.session_state["report_pdf"],
            file_name="cluster_report.pdf",
            mime="application/pdf",
            key="download_report",
        )

        if st.session_state.get("trigger_download"):
            st.markdown(
                """
                <script>
                const btn = window.parent.document.querySelector('button[aria-label="PDFをダウンロード"]');
                if (btn) { btn.click(); }
                </script>
                """,
                unsafe_allow_html=True,
            )
            st.session_state["trigger_download"] = False

    st.success("分析が完了しました！")
