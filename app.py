import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ==========================
# 1. KONFIGURASI & STATE
# ==========================
st.set_page_config(
    page_title="TrafficSignVision",
    page_icon="🚦",
    layout="centered"
)

# Inisialisasi Session State untuk Navigasi
if 'halaman_aktif' not in st.session_state:
    st.session_state.halaman_aktif = "🏠 Beranda"

def pindah_ke_klasifikasi():
    st.session_state.halaman_aktif = "🔍 Fitur Klasifikasi"

# ==========================
# 2. GAYA TAMPILAN (CSS)
# ==========================
st.markdown("""
    <style>
        /* Background Utama */
        .stApp { background-color: #f7fff9; }
        
        /* Kontainer Putih di Tengah */
        .block-container {
            background-color: #ffffff;
            border-radius: 16px;
            box-shadow: 0px 4px 12px rgba(0, 100, 0, 0.1);
            padding: 2rem;
            margin-top: 2rem;
        }

        /* Sidebar Hijau Muda */
        [data-testid="stSidebar"] {
            background-color: #d8f3dc !important;
        }

        h1, h2, h3 { color: #1b4332; font-weight: 700; }
        p, li, div { color: #2d6a4f; font-size: 1rem; }
        
        /* Tombol Utama (Teks Putih) */
        .stButton > button {
            background-color: #2d6a4f !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            border: none;
            width: 100%;
            padding: 0.6rem;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #40916c !important;
            color: #ffffff !important;
            transform: scale(1.02);
        }
        .stButton > button p { color: #ffffff !important; }

        /* Styling Tab (Teks Terbaca) */
        .stTabs [data-baseweb="tab"] {
            background-color: #e8f5e9;
            border-radius: 4px;
            padding: 10px 20px;
            gap: 5px;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #2d6a4f !important;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2d6a4f !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #ffffff !important;
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 0.85rem;
            color: #52b788;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px dashed #b7e4c7;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 3. LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/TrafficSignVision.h5")
        return model
    except Exception as e:
        st.error("⚠️ Model tidak ditemukan. Pastikan file ada di folder 'models/'.")
        return None

model = load_model()

class_labels = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
    'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles > 3.5 tons'
]

class_labels_id = {
    'Speed limit (20km/h)': 'Batas kecepatan 20 km/jam',
    'Speed limit (30km/h)': 'Batas kecepatan 30 km/jam',
    'Speed limit (50km/h)': 'Batas kecepatan 50 km/jam',
    'Speed limit (60km/h)': 'Batas kecepatan 60 km/jam',
    'Speed limit (70km/h)': 'Batas kecepatan 70 km/jam',
    'Speed limit (80km/h)': 'Batas kecepatan 80 km/jam',
    'End of speed limit (80km/h)': 'Akhir batas kecepatan 80 km/jam',
    'Speed limit (100km/h)': 'Batas kecepatan 100 km/jam',
    'Speed limit (120km/h)': 'Batas kecepatan 120 km/jam',
    'No passing': 'Dilarang menyalip',
    'No passing for vehicles over 3.5 metric tons': 'Dilarang menyalip kendaraan >3,5 ton',
    'Right-of-way at the next intersection': 'Prioritas di persimpangan berikutnya',
    'Priority road': 'Jalan utama',
    'Yield': 'Memberi jalan',
    'Stop': 'Berhenti',
    'No vehicles': 'Kendaraan dilarang',
    'Vehicles over 3.5 metric tons prohibited': 'Kendaraan >3,5 ton dilarang',
    'No entry': 'Dilarang masuk',
    'General caution': 'Hati-hati',
    'Dangerous curve to the left': 'Tikungan tajam ke kiri',
    'Dangerous curve to the right': 'Tikungan tajam ke kanan',
    'Double curve': 'Tikungan ganda',
    'Bumpy road': 'Jalan bergelombang',
    'Slippery road': 'Jalan licin',
    'Road narrows on the right': 'Jalan menyempit di kanan',
    'Road work': 'Pekerjaan jalan',
    'Traffic signals': 'Lampu lalu lintas',
    'Pedestrians': 'Pejalan kaki',
    'Children crossing': 'Anak-anak menyeberang',
    'Bicycles crossing': 'Sepeda menyeberang',
    'Beware of ice/snow': 'Waspada es/salju',
    'Wild animals crossing': 'Hewan liar menyeberang',
    'End of all speed and passing limits': 'Akhir semua batas kecepatan dan larangan menyalip',
    'Turn right ahead': 'Belok kanan',
    'Turn left ahead': 'Belok kiri',
    'Ahead only': 'Maju saja',
    'Go straight or right': 'Maju atau belok kanan',
    'Go straight or left': 'Maju atau belok kiri',
    'Keep right': 'Tetap di kanan',
    'Keep left': 'Tetap di kiri',
    'Roundabout mandatory': 'Bundaran wajib',
    'End of no passing': 'Akhir larangan menyalip',
    'End of no passing by vehicles > 3.5 tons': 'Akhir larangan menyalip kendaraan >3,5 ton'
}

# ==========================
# 4. NAVIGASI (SIDEBAR)
# ==========================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3097/3097144.png", width=80)
st.sidebar.title("🌿 Navigasi")
st.sidebar.write("Pilih menu di bawah ini:")

menu = st.sidebar.radio(
    "Menu",
    ["🏠 Beranda", "🔍 Fitur Klasifikasi", "ℹ️ Tentang Aplikasi"], 
    key="halaman_aktif"
)

# ==========================
# 5. KONTEN HALAMAN
# ==========================
if menu == "🏠 Beranda":
    st.title("🚦 TrafficSignVision")
    st.subheader("Kenali Rambu Lalu Lintas dengan Cepat")
    
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.markdown("""
        **Selamat Datang di TrafficSignVision!**
        
        Aplikasi ini adalah solusi cerdas untuk membantu Anda mencari tahu arti rambu lalu lintas secara instan.
        
        ### Mengapa Aplikasi Ini Dibuat?
        Rambu lalu lintas adalah bahasa universal di jalan raya. Namun, dengan banyaknya jenis simbol, terkadang kita lupa atau tidak yakin artinya.
        
        **Salah mengartikan** rambu bisa berbahaya. TrafficSignVision hadir sebagai solusi belajar yang mudah untuk **membantu masalah ini**.
        
        ### Fitur Utama:
        * **📤 Klasifikasi via Upload:** Unggah gambar rambu dari galeri Anda.
        * **📸 Deteksi via Webcam:** Gunakan kamera Anda untuk memindai rambu secara langsung.
        * **💡 Hasil Akurat:** Dapatkan nama rambu beserta tingkat keyakinan (akurasi) prediksinya.
        * **📊 Statistik Detail:** Lihat 3 tebakan teratas untuk rambu yang paling mirip.
        """)
        
        st.write("") 
        st.button("🚀 Coba Klasifikasi Sekarang", on_click=pindah_ke_klasifikasi)

    with col2:
        st.image(
            "https://cdn.pixabay.com/animation/2023/06/13/15/13/15-13-03-816_512.gif",
            caption="Mendeteksi rambu lalu lintas...",
            use_container_width=True
        )

elif menu == "🔍 Fitur Klasifikasi":
    st.title("🔍 Deteksi Rambu")
    st.write("Pilih metode input gambar di bawah ini:")

    tab1, tab2 = st.tabs(["📤 Upload Gambar", "📸 Ambil Foto (Webcam)"])
    input_image = None 

    with tab1:
        uploaded_file = st.file_uploader("Pilih file (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file: input_image = uploaded_file

    with tab2:
        camera_image = st.camera_input("Arahkan rambu ke kamera")
        if camera_image: input_image = camera_image

    if input_image is not None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            image = Image.open(input_image).convert("RGB")
            st.image(image, caption="Gambar Input", use_container_width=True)

            if st.button("🔍 Analisis Gambar"):
                if model is None:
                    st.error("Model belum dimuat.")
                else:
                    with st.spinner('Sedang menganalisis...'):
                        time.sleep(0.5)
                        # Preprocessing
                        img_resized = image.resize((30, 30))
                        img_array = np.array(img_resized, dtype=np.float32)
                        img_array = np.expand_dims(img_array, axis=0)

                        # Prediksi
                        prediction = model.predict(img_array)
                        predicted_class = int(np.argmax(prediction))
                        confidence = float(np.max(prediction)) * 100

                        st.success("Selesai!")

                        # Hasil Prediksi Inggris
                        st.markdown(f"""
                        <div style="
                            background-color: #d8f3dc;
                            padding: 20px;
                            border-radius: 12px;
                            border-left: 8px solid #2d6a4f;
                            margin-top: 20px;
                            margin-bottom: 20px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        ">
                            <h3 style="margin:0; color: #1b4332;">Hasil: {class_labels[predicted_class]}</h3>
                            <p style="margin-top: 5px; color: #2d6a4f;">Akurasi: <b>{confidence:.2f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Hasil Prediksi Bahasa Indonesia
                        st.markdown(f"""
                        <div style="
                            margin-top:5px; 
                            padding:10px; 
                            background-color:#e0f7fa; 
                            border-radius:8px;
                        ">
                            <p style="margin:0; color:#00796b;">
                                <b>Terjemahan (Bahasa Indonesia):</b> {class_labels_id[class_labels[predicted_class]]}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Statistik 3 Tebakan Teratas
                        with st.expander("Lihat Detail Statistik"):
                            probs = dict(zip(class_labels, prediction[0]))
                            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                            for label, prob in sorted_probs[:3]:
                                st.progress(float(prob))
                                st.caption(f"{label} ({prob*100:.2f}%) - {class_labels_id[label]}")

elif menu == "ℹ️ Tentang Aplikasi": 
    st.title("ℹ️ Tentang TrafficSignVision")
    st.write("") 

    st.subheader("Gambaran Proyek")
    st.markdown("""
    TrafficSignVision adalah platform berbasis web untuk klasifikasi simbol rambu lalu lintas.  
    Dengan memanfaatkan model *Machine Learning* canggih berupa Convolutional Neural Network (**CNN**), aplikasi ini mampu mengenali pola visual pada gambar rambu, seperti bentuk, warna, dan simbol, sehingga dapat mengklasifikasikan rambu secara instan.  
    Model ini dapat bekerja baik pada gambar yang diunggah maupun melalui kamera secara langsung, memberikan nama rambu beserta tingkat keyakinannya secara real-time.
    """)

    st.markdown("---") 

    st.subheader("Bagaimana Model Ini Dilatih?")
    st.markdown("""
    Model ini dilatih menggunakan dataset publik **GTSRB (German Traffic Sign Recognition Benchmark)**. 
    Dataset ini sangat besar, berisi lebih dari 39.000 gambar rambu lalu lintas yang dibagi menjadi 43 jenis (kelas).  
    """)

# ==========================
# 6. FOOTER GLOBAL (HANYA BERANDA & TENTANG)
# ==========================
def show_footer():
    st.markdown("""
    <div class='footer'>
        <p>Dibuat dengan 💚 oleh <b>Inna Putri Meida</b> | © 2025 TrafficSignVision</p>
    </div>
    """, unsafe_allow_html=True)

# Panggil footer hanya jika menu saat ini Beranda atau Tentang Aplikasi
if menu in ["🏠 Beranda", "ℹ️ Tentang Aplikasi"]:
    show_footer()