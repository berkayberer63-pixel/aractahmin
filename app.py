import csv
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    layout="wide",
    page_title="Ford Araç Fiyat Tahmini",
)

# CSS dosyasını yükle
def css_yukle(dosya_yolu):
    try:
        with open(dosya_yolu, "r", encoding="utf-8") as dosya:
            css = dosya.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

css_yukle("style.css")



# ================= 1) VERİ SETİNİ OKUMA VE MODEL EĞİTİMİ (EKSİK VERİ KONTROLLÜ) =================
@st.cache_resource
def modeli_egit_ve_hazirla(agac_sayisi, agac_derinligi):
    sutunlar = ["year", "mileage", "tax", "mpg", "engineSize"]
    X, y = [], []
    
    with open("ford.csv", encoding="utf-8") as dosya:
        for r in csv.DictReader(dosya):
            try:
                yil = float(r["year"])
                if yil > 2025:
                    continue
                
                satir_degerleri = [float(r[s]) for s in sutunlar]
                fiyat = float(r["price"])
                
                X.append(satir_degerleri)
                y.append(fiyat)
            except (ValueError, KeyError, TypeError):
                continue

    X, y = np.array(X), np.array(y)

    np.random.seed(42)
    sira = np.random.permutation(len(X))
    X, y = X[sira], y[sira]

    def agac_kur(X, y, derinlik):
        if derinlik == 0 or len(y) < 30:
            return y.mean()
        en_iyi_hata, en_iyi = None, None
        for j in range(X.shape[1]):
            for esik in np.percentile(X[:, j], [20, 40, 60, 80]):
                sol = X[:, j] <= esik
                if sol.sum() == 0 or sol.sum() == len(y):
                    continue
                hata = ((y[sol] - y[sol].mean()) ** 2).sum() + ((y[~sol] - y[~sol].mean()) ** 2).sum()
                if en_iyi_hata is None or hata < en_iyi_hata:
                    en_iyi_hata, en_iyi = hata, (j, esik, sol)
        if en_iyi is None:
            return y.mean()
        j, esik, sol = en_iyi
        return (j, esik, agac_kur(X[sol], y[sol], derinlik - 1), agac_kur(X[~sol], y[~sol], derinlik - 1))

    def agac_tahmin(agac, x):
        while isinstance(agac, tuple):
            j, esik, sol, sag = agac
            agac = sol if x[j] <= esik else sag
        return agac

    def rf_tahmin(agaclar, X):
        return np.array([np.mean([agac_tahmin(a, x) for a in agaclar]) for x in X])

    def lr_tahmin(w, X):
        return np.column_stack([np.ones(len(X)), X]) @ w

    def olc(yt, p):
        r2 = 1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        return {
            "r2": round(float(r2), 4),
            "rmse": round(float(np.sqrt(((yt - p) ** 2).mean()))),
        }

    modeller, metrikler = {}, {}
    for oran in (80, 90):
        n = int(len(X) * oran / 100)
        X_egitim, y_egitim, X_test, y_test = X[:n], y[:n], X[n:], y[n:]

        A = np.column_stack([np.ones(n), X_egitim])
        w = np.linalg.solve(A.T @ A, A.T @ y_egitim)
        agaclar = []
        for _ in range(agac_sayisi):
            i = np.random.randint(0, n, n)
            agaclar.append(agac_kur(X_egitim[i], y_egitim[i], agac_derinligi))

        modeller[f"lr{oran}"] = lambda x, w=w: float(lr_tahmin(w, x)[0])
        modeller[f"rf{oran}"] = lambda x, a=agaclar: float(rf_tahmin(a, x)[0])
        metrikler[f"lr{oran}"] = olc(y_test, lr_tahmin(w, X_test))
        metrikler[f"rf{oran}"] = olc(y_test, rf_tahmin(agaclar, X_test))

    return modeller, metrikler


# ================= 2) STREAMLIT ARAYÜZÜ (FORM İLE KONTROL) =================
st.title("Ford Araç Fiyat Tahmini ve Model Karşılaştırma")

if "hesaplandi" not in st.session_state:
    st.session_state.hesaplandi = False

with st.sidebar.form("arac_formu"):
    st.header("Araç Özellikleri")
    yil = st.number_input("Üretim Yılı", value=2018, step=1)
    km = st.number_input("Kilometre", value=20000, step=1000)
    vergi = st.number_input("Vergi (£)", value=150, step=10)
    mpg = st.number_input("MPG (Yakıt Tüketimi)", value=55.0, step=0.5)
    motor = st.number_input("Motor Hacmi (L)", value=1.5, step=0.1)

    st.markdown("---")
    st.header("Random Forest Ayarları")
    secilen_agac_sayisi = st.slider("Ağaç Sayısı (n_estimators)", min_value=3, max_value=25, value=8, step=1)
    secilen_derinlik = st.slider("Ağaç Derinliği (max_depth)", min_value=2, max_value=8, value=4, step=1)

    submit_buton = st.form_submit_button("Fiyatı Tahmin Et")

    if submit_buton:
        st.session_state.hesaplandi = True
        st.session_state.x_giris = np.array([[yil, km, vergi, mpg, motor]])
        st.session_state.agac_sayisi = secilen_agac_sayisi
        st.session_state.agac_derinligi = secilen_derinlik

aktif_agac_sayisi = st.session_state.get("agac_sayisi", 8)
aktif_derinlik = st.session_state.get("agac_derinligi", 4)

with st.spinner("Modeller yükleniyor..."):
    modeller, metrikler = modeli_egit_ve_hazirla(aktif_agac_sayisi, aktif_derinlik)

if st.session_state.hesaplandi:
    x_giris = st.session_state.x_giris
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### %80 Eğitim / %20 Test")
        st.metric("Lineer Regresyon", f"£{max(0, modeller['lr80'](x_giris)):,.0f}".replace(",", "."))
        st.metric(f"Random Forest ({aktif_agac_sayisi} Ağaç, {aktif_derinlik} Derinlik)", f"£{max(0, modeller['rf80'](x_giris)):,.0f}".replace(",", "."))

    with col2:
        st.markdown("### %90 Eğitim / %10 Test")
        st.metric("Lineer Regresyon", f"£{max(0, modeller['lr90'](x_giris)):,.0f}".replace(",", "."))
        st.metric(f"Random Forest ({aktif_agac_sayisi} Ağaç, {aktif_derinlik} Derinlik)", f"£{max(0, modeller['rf90'](x_giris)):,.0f}".replace(",", "."))

st.divider()

st.subheader("Model Başarı Oranları")
tablo_verisi = [
    {"Model": "Lineer Regresyon", "Eğitim Oranı": "%80", "Başarı (R²)": f"%{metrikler['lr80']['r2']*100:.1f}", "RMSE (Hata)": f"£{metrikler['lr80']['rmse']}"},
    {"Model": "Lineer Regresyon", "Eğitim Oranı": "%90", "Başarı (R²)": f"%{metrikler['lr90']['r2']*100:.1f}", "RMSE (Hata)": f"£{metrikler['lr90']['rmse']}"},
    {"Model": f"Random Forest ({aktif_agac_sayisi} Ağaç, {aktif_derinlik} Derinlik)", "Eğitim Oranı": "%80", "Başarı (R²)": f"%{metrikler['rf80']['r2']*100:.1f}", "RMSE (Hata)": f"£{metrikler['rf80']['rmse']}"},
    {"Model": f"Random Forest ({aktif_agac_sayisi} Ağaç, {aktif_derinlik} Derinlik)", "Eğitim Oranı": "%90", "Başarı (R²)": f"%{metrikler['rf90']['r2']*100:.1f}", "RMSE (Hata)": f"£{metrikler['rf90']['rmse']}"}
]
st.table(tablo_verisi)

st.divider()

# ================= 3) GRAFİK BÖLÜMÜ (Ağaç Derinliğinin Etkisi) =================
st.subheader("Ağaç Derinliğinin Başarıya Etkisi")
st.write(f"*(Ağaç Sayısı: {aktif_agac_sayisi} - %80 Eğitim Seti)*")

derinlikler = [2, 3, 4, 5, 6, 7] # Test edilecek derinlikler
basari_oranlari = []

with st.spinner("Grafik için farklı derinlikler test ediliyor..."):
    for d in derinlikler:
        # Sadece R2 sonucunu almak için fonksiyonumuzu farklı derinlik değerleriyle çağırıyoruz
        _, test_metrikleri = modeli_egit_ve_hazirla(aktif_agac_sayisi, d)
        basari_oranlari.append(test_metrikleri['rf80']['r2'] * 100)

# Veriyi tablo haline getirip çizdiriyoruz
df_grafik = pd.DataFrame({
    "Derinlik": derinlikler,
    "Başarı Oranı (%)": basari_oranlari
}).set_index("Derinlik")

# Ekranı ikiye bölüyoruz (col1 sol yarı, col2 sağ yarı)
# [1, 1] oranı ekranı tam ortadan ikiye böler. Grafiği daha büyük/küçük isterseniz oranları [6, 4] gibi değiştirebilirsiniz.
# Ekranı ikiye bölüyoruz (Grafik sağa doğru uzamasın diye)
col1, col2 = st.columns([1, 1])

with col1:
    # Plotly'nin veriyi okuyabilmesi için index'i sıfırlıyoruz
    df_reset = df_grafik.reset_index()
    
    # px.area ile alan grafiği oluşturup, markers=True ile kesişim noktalarını ekliyoruz
    fig = px.area(
        df_reset, 
        x="Derinlik", 
        y="Başarı Oranı (%)", 
        markers=True # Kesişim yerlerine nokta koyan sihirli ayar
    )
    
    # İsteğe bağlı: Arka planı transparan yapmak ve renkleri temanıza uydurmak için
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    
    # Grafiği ekrana basıyoruz
    st.plotly_chart(fig, use_container_width=True)