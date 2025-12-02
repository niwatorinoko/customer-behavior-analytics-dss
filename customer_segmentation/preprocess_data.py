import pandas as pd
# =========================
#  前処理関数
# =========================
def preprocess_retail_data(df):
    # カラム名の標準化
    df = df.rename(columns={
        "Customer ID": "CustomerID",
        "Invoice": "InvoiceNo"
    })

    # 日付変換
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # 型変換
    df["CustomerID"] = df["CustomerID"].astype("Int64")

    # 不正データ除外
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])

    # Monetary
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    return df