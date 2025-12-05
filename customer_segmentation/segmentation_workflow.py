import streamlit as st
import pandas as pd
from customer_segmentation.data_preprocessing import preprocess_retail_data
from customer_segmentation.rfm import calculate_rfm
from customer_segmentation.clustering import cluster_rfm

def run_segmentation_tab():
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

        st.success("顧客分析が完了しました！")
        st.session_state["rfm_done"] = True
        st.session_state["rfm_clustered"] = rfm_clustered
        st.session_state["cluster_means"] = cluster_means
        
        # 消さないで！！
        if not st.session_state.get("rerun_triggered", False):
            st.session_state["rerun_triggered"] = True
            st.rerun()