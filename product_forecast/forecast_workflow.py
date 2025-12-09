import streamlit as st
import pandas as pd
import numpy as np

def simulate_product_forecast(product_summary: pd.DataFrame) -> pd.DataFrame:
    """
    集計結果に基づき、販売予測の精度指標をダミーで追加するシミュレーション関数。
    実際には機械学習モデルを組み込みます。
    """
    
    # 精度指標をダミーで生成
    # 簡易的に販売件数が多いほど（おそらく重要な商品ほど）精度が高いと仮定
    
    def get_dummy_metrics(count):
        # MAE (Mean Absolute Error), RMSE (Root Mean Square Error), R² (R-squared)
        if count > 500:
            return 15.0, 25.0, 0.85
        elif count > 100:
            return 25.0, 40.0, 0.75
        else:
            return 35.0, 50.0, 0.65
            
    # 新しい列を計算して追加
    metrics = product_summary['SalesCount'].apply(
        lambda x: pd.Series(get_dummy_metrics(x))
    )
    metrics.columns = ['MAE', 'RMSE', 'R²']
    
    forecast_summary = pd.concat([product_summary, metrics], axis=1)
    
    return forecast_summary.sort_values(by='SalesCount', ascending=False)

def run_forecast_tab():
    st.header("📦 商品販売予測（試験実装）")
    st.write("日時と商品名（＋数量）のデータから販売傾向を簡易的に分析し、レポート用サマリーを作成します。")

    uploaded_file = st.file_uploader("販売データCSVをアップロード", type="csv", key="forecast")

    if not uploaded_file:
        # ファイルがない場合はセッションステートをクリア
        st.session_state.pop("product_summary", None)
        st.session_state["product_ready"] = False
        return
    try:
        uploaded_file.seek(0) # 念のためポインタをリセット
        df = pd.read_csv(uploaded_file, encoding='utf_8_sig') 
        st.success(f"ファイル `{uploaded_file.name}` の読み込みが完了しました (UTF-8 SIG)。")
    
    except UnicodeDecodeError:
        # 2. 失敗したらShift-JISで再試行
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='shift_jis')
            st.warning("⚠️ ファイルがShift-JISとして読み込まれました。")
        except Exception as e_sjis:
            # 3. それでも失敗したら、エラーを無視してShift-JISで読み込み（最終手段）
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='shift_jis', errors='ignore')
                st.error("🚨 致命的なエンコーディングエラー。不正な文字を無視して読み込みました。データを確認してください。")
            except Exception as e_ignore:
                st.error(f"ファイルの読み込みに失敗しました。エンコーディングを確認してください: {e_ignore}")
                st.session_state.pop("product_summary", None)
                st.session_state["product_ready"] = False
                return
    
    except Exception as e:
        # その他の一般的な読み込みエラー
        st.error(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        st.session_state.pop("product_summary", None)
        st.session_state["product_ready"] = False
        return
    
    # --- ファイルアップロード後の処理 ---
    
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
    if "Product" not in df.columns:
        st.error("❌ 'Product' 列が存在しません。CSVのカラム名を確認してください。")
        st.session_state.pop("product_summary", None)
        st.session_state["product_ready"] = False
        return
        
    # 商品別集計を実行
    product_summary = df.groupby("Product").size().reset_index(name="SalesCount")
    st.dataframe(product_summary.sort_values("SalesCount", ascending=False))

    # 日別トレンド（任意）
    if "Date" in df.columns:
        st.subheader("③ 日別販売数の推移")
        daily_sales = df.groupby("Date").size().reset_index(name="SalesCount")
        st.line_chart(daily_sales.set_index("Date")["SalesCount"])
        
    st.success("簡易的な販売分析が完了しました。レポート用サマリーを作成します。")
    
    # 1. 予測シミュレーションの実行 (即時実行)
    forecast_summary_df = simulate_product_forecast(product_summary)

    # 2. 結果の表示
    st.subheader("④ 予測サマリー（レポート連携用）")
    st.info("販売件数に基づき、予測精度指標をシミュレーションしています。")
    st.dataframe(forecast_summary_df)
        
    # 3. レポート連携のためにセッションステートに保存
    st.session_state["product_summary"] = forecast_summary_df
        
    # レポート生成のチェックボックスを有効化するためのフラグを立てる
    st.session_state["product_ready"] = True 

    st.success("全ての分析とサマリーの作成が完了しました。左側のレポート生成設定で、この結果を選択できます。")
    st.session_state["forecast_done"] = True

    # 消さないで！！
    if not st.session_state.get("rerun_triggered", False):
        st.session_state["rerun_triggered"] = True
        st.rerun()
