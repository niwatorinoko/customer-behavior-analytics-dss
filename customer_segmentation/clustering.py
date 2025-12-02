# =========================
#  クラスタリング関数
# =========================
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_rfm(rfm, k=4):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(rfm_scaled)

    rfm["Cluster"] = labels
    return rfm, model
