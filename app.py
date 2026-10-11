from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_auto_mpg_rf_model.pkl"
HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7"
    "?auto=format&fit=crop&w=1800&q=85"
)
COUNTRY_TO_ORIGIN = {
    "ABŞ": "USA",
    "Avropa": "Europe",
    "Yaponiya": "Japan",
}

st.set_page_config(
    page_title="Auto MPG | Yanacaq kalkulyatoru",
    page_icon="🚘",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background-color: #f5f7fa;}
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model faylı tapılmadı: {MODEL_PATH}. "
            "best_auto_mpg_rf_model.pkl faylını app.py ilə eyni qovluğa yerləşdirin."
        )
    return joblib.load(MODEL_PATH)


st.image(HERO_IMAGE_URL, use_container_width=True)
st.title("🚘 Avtomobilinizin yanacaq sərfiyyatını təxmin edin")
st.write(
    "Avtomobil göstəricilərini daxil edin və modelin təxmini yanacaq "
    "səmərəliliyini MPG və L/100 km vahidlərində öyrənin."
)
st.caption(
    "Qeyd: nəticə Auto-MPG məlumat bazasında öyrədilmiş modelin təxminidir; "
    "real sərfiyyat sürüş şəraiti və avtomobilin texniki vəziyyətinə görə dəyişə bilər."
)

with st.sidebar:
    st.header("Avtomobil göstəriciləri")
    st.write("Məlumatları daxil edin:")

    cylinders = st.selectbox("Silindr sayı", [3, 4, 5, 6, 8], index=1)
    displacement = st.slider(
        "Mühərrik həcmi (cubic inch)",
        min_value=60,
        max_value=500,
        value=140,
        help="Auto-MPG modelində mühərrik həcmi cubic inch vahidində saxlanılıb.",
    )
    horsepower = st.slider(
        "Mühərrik gücü (At Gücü)",
        min_value=40,
        max_value=250,
        value=90,
    )
    weight = st.slider(
        "Avtomobilin tam çəkisi (lbs)",
        min_value=1500,
        max_value=5000,
        value=2400,
        help="Model lbs (funt) vahidi ilə öyrədilib. Kq dəyəri daxil etməyin.",
    )
    acceleration = st.slider(
        "Sürətlənmə vaxtı (saniyə)",
        min_value=8.0,
        max_value=25.0,
        value=15.5,
        step=0.5,
        help=(
            "Təlim məlumatındakı acceleration göstəricisinə uyğun saniyə dəyəri "
            "daxil edin. Müasir 0-100 km/s göstəricisi ilə ölçmə tam eyni olmaya bilər."
        ),
    )
    model_year = st.slider(
        "Buraxılış ili",
        min_value=1970,
        max_value=1982,
        value=1980,
    )
    country_label = st.selectbox(
        "İstehsalçı ölkə",
        list(COUNTRY_TO_ORIGIN),
    )
    brand = st.text_input("Avtomobil markası", value="ford").strip().lower()

    calculate = st.button(
        "Yanacaq sərfiyyatını hesabla",
        type="primary",
        use_container_width=True,
    )

st.subheader("Təxmin")
if calculate:
    if not brand:
        st.error("Zəhmət olmasa avtomobil markasını daxil edin.")
    else:
        try:
            model = load_model()
            car = pd.DataFrame(
                [
                    {
                        "cylinders": cylinders,
                        "displacement": displacement,
                        "horsepower": horsepower,
                        "weight": weight,
                        "acceleration": acceleration,
                        # Model Auto-MPG-də ili 70-82 formasında görüb.
                        "model_year": model_year - 1900,
                        "origin": COUNTRY_TO_ORIGIN[country_label],
                        "brand": brand,
                    }
                ]
            )

            # Pipeline əlavə nisbətləri hesablayır və nadir markaları "other" edir.
            log_mpg = float(model.predict(car)[0])
            mpg = float(np.expm1(log_mpg))
            if not np.isfinite(mpg) or mpg <= 0:
                st.error("Model etibarlı sərfiyyat təxmini qaytarmadı.")
            else:
                liters_per_100km = 235.215 / mpg
                mpg_col, metric_col = st.columns(2)
                mpg_col.metric("Təxmini yanacaq səmərəliliyi", f"{mpg:.2f} MPG")
                metric_col.metric("Təxmini sərfiyyat", f"{liters_per_100km:.2f} L/100 km")
                st.caption(
                    "Hesablama: MPG = expm1(model proqnozu); "
                    "L/100 km = 235.215 / MPG."
                )
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            st.error(f"Modeli yükləmək və ya proqnoz hesablamaq mümkün olmadı: {exc}")

st.divider()
st.caption(
    "Model 1970–1982-ci illərin Auto-MPG məlumatları ilə qurulub. "
    "Bu demo peşəkar yanacaq sərfiyyatı ölçməsini əvəz etmir."
)
