import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ======================================================
# SESSION STATE (WELCOME)
# ======================================================
if "welcome" not in st.session_state:
    st.session_state.welcome = True

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Klasifikasi Kemiskinan Indonesia",
    page_icon="📊",
    layout="centered"
)

# ======================================================
# STYLE / TEMA
# ======================================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f766e, #0369a1);
}

.main {
    background: linear-gradient(135deg, #0f766e, #0369a1);
    padding: 30px;
}

div.block-container {
    background: rgba(248, 250, 252, 0.96);
    border-radius: 24px;
    padding: 2.5rem;
    max-width: 1050px;
    margin: auto;
    box-shadow: 0 20px 40px rgba(0,0,0,0.25);
}

h1 {
    color: #111827;
    text-align: center;
    font-weight: 800;
}

h2, h3 {
    color: #065f46;
    font-weight: 700;
}

p {
    color: #1f2937;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #bbf7d0, #bae6fd);
    border-radius: 16px;
    padding: 18px;
    color: #064e3b;
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
}

.stDataFrame {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 10px;
}

.stAlert-success {
    background-color: #dcfce7;
    color: #065f46;
    border-left: 6px solid #16a34a;
}

.stAlert-info {
    background-color: #e0f2fe;
    color: #075985;
    border-left: 6px solid #0284c7;
}

hr {
    border: none;
    height: 3px;
    background: linear-gradient(to right, #22c55e, #38bdf8);
    margin: 32px 0;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HALAMAN SELAMAT DATANG
# ======================================================
if st.session_state.welcome:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e0f2fe, #f0fdf4);
        padding: 40px;
        border-radius: 22px;
        text-align: center;
        margin-top: 80px;
    ">
        <h1>👋 Selamat Datang</h1>
        <p style="font-size:18px;">
        Aplikasi Analisis dan Klasifikasi<br>
        <b>Status Kemiskinan Penduduk Indonesia</b>
        </p>
        <p style="color:gray;">
        Proyek UAS Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Mulai Analisis"):
        st.session_state.welcome = False
        st.rerun()

    # PENTING: HENTIKAN RENDER DI SINI
    st.stop()

# ======================================================
# DASHBOARD UTAMA
# ======================================================
st.title("Klasifikasi Status Kemiskinan Penduduk Indonesia")

st.markdown(
    "<div style='height:4px; width:90px; background:#22c55e; "
    "margin: 14px auto 26px auto; border-radius:10px;'></div>",
    unsafe_allow_html=True
)

st.image(
    "https://images.unsplash.com/photo-1509099836639-18ba1795216d",
    width=720
)

st.markdown(
    "<p style='text-align:center; color:#374151;'>"
    "Aplikasi ini menyajikan analisis eksploratif dan klasifikasi "
    "status kemiskinan penduduk Indonesia menggunakan "
    "<b>machine learning</b> berbasis data.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ======================================================
# LOAD DATA
# ======================================================
data = pd.read_csv("data_kemiskinan.csv", sep=";")
data.columns = data.columns.str.strip()

st.subheader("📂 Preview Dataset")
st.dataframe(data.head())

st.markdown("---")

# ======================================================
# PENCARIAN WILAYAH
# ======================================================
st.subheader("🔍 Pencarian Wilayah")

wilayah_col = st.selectbox(
    "Pilih kolom wilayah",
    data.columns
)

search_text = st.text_input(
    "Ketik nama wilayah (contoh: Aceh, Bangka Belitung)"
)

if search_text:
    data = data[
        data[wilayah_col].astype(str)
        .str.contains(search_text, case=False, na=False)
    ]

st.write(f"Jumlah data setelah filter: **{data.shape[0]}**")
st.dataframe(data)

st.markdown("---")

# ======================================================
# PILIH KOLOM PERSENTASE KEMISKINAN
# ======================================================
st.subheader("📊 Pilih Kolom Persentase Kemiskinan")

persen_col = st.selectbox(
    "Pilih kolom persentase penduduk miskin",
    data.select_dtypes(include=["float", "int"]).columns
)

# ======================================================
# BUAT TARGET OTOMATIS
# ======================================================
median_value = data[persen_col].median()

data["Status_Kemiskinan"] = data[persen_col].apply(
    lambda x: 1 if x >= median_value else 0
)

st.success("Target otomatis dibuat: **0 = Tidak Miskin | 1 = Miskin**")

st.markdown("---")

# ======================================================
# PREPROCESSING
# ======================================================
st.subheader("⚙️ Preprocessing Data")

data_encoded = data.copy()
le = LabelEncoder()

for col in data_encoded.select_dtypes(include="object").columns:
    data_encoded[col] = le.fit_transform(data_encoded[col])

X = data_encoded.drop("Status_Kemiskinan", axis=1)
y = data_encoded["Status_Kemiskinan"]

# ======================================================
# SPLIT DATA
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# PILIH MODEL
# ======================================================
st.subheader("🤖 Model Machine Learning")

model_option = st.selectbox(
    "Pilih Model",
    ("Random Forest",)
)

model = RandomForestClassifier(random_state=42)

# ======================================================
# TRAIN & EVALUASI
# ======================================================
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("📈 Evaluasi Model")

accuracy = accuracy_score(y_test, y_pred)
st.metric("Akurasi Model", f"{accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
st.pyplot(fig)

# ======================================================
# FOOTER
# ======================================================
st.info(
    "Aplikasi ini dikembangkan sebagai bagian dari **UAS Machine Learning**, "
    "mencakup analisis eksploratif, preprocessing, klasifikasi, dan evaluasi model."
)
