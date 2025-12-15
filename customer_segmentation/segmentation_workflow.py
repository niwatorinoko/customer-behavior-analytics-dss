import streamlit as st
import pandas as pd
from customer_segmentation.data_preprocessing import preprocess_retail_data
from customer_segmentation.rfm import calculate_rfm
from customer_segmentation.clustering import cluster_rfm

def run_segmentation_tab():
    st.header("🧍‍♂️ Customer Segmentation (RFM Analysis)")
    uploaded_file = st.file_uploader("Upload Customer Data CSV", type="csv", key="customer")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("① Uploaded Data")
        st.dataframe(df.head())

        # 前処理
        df_clean = preprocess_retail_data(df)
        st.subheader("② Preprocessed Data")
        st.dataframe(df_clean.head())

        # RFM計算
        rfm = calculate_rfm(df_clean)
        st.subheader("③ RFM Table")
        st.dataframe(rfm.head())

        # クラスタリング
        k = st.slider("Number of Clusters (k)", 2, 10, 4)
        rfm_clustered, model = cluster_rfm(rfm, k)

        st.subheader("④ Clustering Results")
        st.dataframe(rfm_clustered.head())

        # クラスタ平均
        st.subheader("⑤ Cluster-wise Means")
        cluster_means = rfm_clustered.groupby("Cluster").mean()
        st.dataframe(cluster_means)

        st.success("Customer analysis completed!")
        st.session_state["rfm_done"] = True
        st.session_state["rfm_clustered"] = rfm_clustered
        st.session_state["cluster_means"] = cluster_means
        
        # Keep this line!
        if not st.session_state.get("rerun_triggered", False):
            st.session_state["rerun_triggered"] = True
            st.rerun()