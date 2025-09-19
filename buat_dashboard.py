# Nama file: buat_dashboard.py

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. Setup Halaman ---
st.set_page_config(page_title="Dashboard Analisis Gaming", layout="wide")
st.title("🎮 Gaming Habits and Psychological Well-being: An international dataset about the Anxiety, Life Satisfaction and Social Phobia of over 13000 gamers")

# --- 2. Memuat Data ---
file_path = "GamingStudy_data.csv"

try:
    df = pd.read_csv(file_path, encoding="latin1")
    st.success(f"Berhasil memuat data dari '{file_path}'")
except FileNotFoundError:
    st.error(f"File '{file_path}' tidak ditemukan. Pastikan ada di folder yang sama dengan script ini.")
    st.stop()

# --- 3. Pembersihan Data ---
df = df[(df['Age'] >= 18) & (df['Age'] < 66)]
df['Hours_capped'] = df['Hours'].clip(upper=100)

# --- 4. Layout Dashboard ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Distribusi Usia",
    "Status Pekerjaan",
    "Jam Bermain",
    "Game Populer",
    "Jam Main vs Kecemasan",
    "Clustering Gamer"
])

# --- 5. Grafik ---
# Tab 1: Distribusi Usia
with tab1:
    fig_age = px.histogram(df, x="Age", nbins=20, color_discrete_sequence=["#5DADE2"])
    fig_age.update_layout(title="Distribusi Usia Gamer", bargap=0.1)
    st.plotly_chart(fig_age, use_container_width=True)

# Tab 2: Status Pekerjaan
with tab2:
    work_counts = df['Work'].value_counts().reset_index()
    work_counts.columns = ["Work", "count"]
    fig_work = px.pie(work_counts, values="count", names="Work",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_work.update_layout(title="Distribusi Status Pekerjaan Gamer")
    st.plotly_chart(fig_work, use_container_width=True)

# Tab 3: Jam Bermain
with tab3:
    fig_hours = px.histogram(df, x="Hours_capped", nbins=20, color_discrete_sequence=["#48C9B0"])
    fig_hours.update_layout(title="Distribusi Jam Bermain per Minggu (dibatasi 100 jam)", bargap=0.1)
    st.plotly_chart(fig_hours, use_container_width=True)

# Tab 4: Game Populer
with tab4:
    game_counts = df['Game'].value_counts().nlargest(10).reset_index()
    game_counts.columns = ["Game", "count"]
    fig_games = px.bar(game_counts, x="count", y="Game", orientation="h",
                       color="count", color_continuous_scale="viridis")
    fig_games.update_layout(title="10 Game Paling Populer")
    st.plotly_chart(fig_games, use_container_width=True)

# Tab 5: Jam Main vs Kecemasan
with tab5:
    fig_corr = px.scatter(df, x="Hours", y="GAD_T",
                          trendline="ols",
                          opacity=0.3,
                          color_discrete_sequence=["#E74C3C"])
    fig_corr.update_layout(title="Jam Bermain vs Skor Kecemasan (GAD)")
    st.plotly_chart(fig_corr, use_container_width=True)

    # tampilkan korelasi
    r = df[['Hours', 'GAD_T']].corr().loc['Hours', 'GAD_T']
    st.info(f"📌 Korelasi Pearson (r) antara Jam Bermain dan Skor Kecemasan: **{r:.3f}**")

# Tab 6: Clustering Gamer
# Tab 6: Clustering Gamer
with tab6:
    st.subheader("🔎 Clustering Gamer berdasarkan Pola Bermain & Kesehatan Mental")

    features = ["Hours_capped", "GAD_T", "SWL_T", "SPIN_T"]
    X = df[features].dropna()

    # normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    X["Cluster"] = kmeans.fit_predict(X_scaled)

    # gabungkan cluster kembali ke df asli (hanya pada baris yang valid)
    df = df.join(X["Cluster"], how="left")

    # scatter plot cluster
    fig_cluster = px.scatter(
        df.dropna(subset=["Cluster"]),
        x="Hours_capped", y="GAD_T",
        color="Cluster",
        hover_data=["SWL_T", "SPIN_T"],
        title="Cluster Gamer: Jam Bermain vs Kecemasan",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # distribusi cluster
    cluster_counts = df["Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Jumlah Gamer"]
    st.dataframe(cluster_counts)

    st.success("Cluster sudah berhasil dibuat dan ditampilkan 🎉")
