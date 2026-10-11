import os
import pickle
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import auto_mpg_transformers
import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best_auto_mpg_rf_model.pkl")
HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1552519507-da3b142c6e3d"
    "?q=80&w=1000&auto=format&fit=crop"
)
COUNTRY_OPTIONS = ["USA", "Europe", "Japan"]

st.set_page_config(
    page_title="Auto-MPG Yanacaq Təxmini",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model faylı tapılmadı: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def load_hero_image():
    with urlopen(HERO_IMAGE_URL, timeout=5) as response:
        return response.read()


st.title("🚗 Avtomobilinizin Yanacaq Sərfiyyatını Təxmin Edin")
st.write(
    "Parametrləri daxil edin və modelin təxmini yanacaq sərfiyyatını görün."
)

try:
    st.image(load_hero_image(), width="stretch")
except (HTTPError, URLError, TimeoutError, OSError) as exc:
    st.warning(f"Şəkil yüklənmədi, lakin tətbiqdən istifadə edə bilərsiniz: {exc}")

try:
    model = load_model()
except (
    OSError,
    ImportError,
    ValueError,
    EOFError,
    AttributeError,
    pickle.UnpicklingError,
) as exc:
    model = None
    st.error(f"Modeli yükləmək mümkün olmadı: {exc}")

with st.sidebar:
    st.header("Avtomobil Parametrləri")

    cylinders = st.selectbox("Silindr Sayı", [3, 4, 5, 6, 8], index=1)
    displacement = st.slider(
        "Mühərrik Həcmi (cu.in)", 60.0, 500.0, 150.0
    )
    horsepower = st.slider("Mühərrik Gücü (At Gücü)", 40.0, 250.0, 100.0)
    weight = st.slider("Avtomobilin Çəkisi (lbs)", 1500.0, 5000.0, 2800.0)
    acceleration = st.slider(
        "Sürətlənmə (saniyə)", 8.0, 25.0, 15.0
    )
    model_year = st.slider("Buraxılış İli (70-82)", 70, 82, 76)
    origin = st.selectbox("İstehsalçı Ölkə", COUNTRY_OPTIONS)
    brand = st.text_input("Avtomobil markası", value="ford").strip().lower()

st.subheader("Təxmini nəticə")
if model is not None and st.button(
    "Yanacaq Sərfiyyatını Hesabla", type="primary"
):
    if not brand:
        st.error("Zəhmət olmasa avtomobil markasını daxil edin.")
    else:
        input_data = pd.DataFrame(
            [
                {
                    "cylinders": cylinders,
                    "displacement": displacement,
                    "horsepower": horsepower,
                    "weight": weight,
                    "acceleration": acceleration,
                    "model_year": model_year,
                    "origin": origin,
                    "brand": brand,
                }
            ]
        )

        try:
            pred_log = float(model.predict(input_data)[0])
            mpg_pred = float(np.expm1(pred_log))
            if not np.isfinite(mpg_pred) or mpg_pred <= 0:
                st.error("Model etibarlı sərfiyyat təxmini qaytarmadı.")
            else:
                l_100km = 235.215 / mpg_pred
                col1, col2 = st.columns(2)
                col1.metric(
                    label="Təxmini Yanacaq Səmərəliliyi",
                    value=f"{mpg_pred:.2f} MPG",
                )
                col2.metric(
                    label="Təxmini Sərfiyyat",
                    value=f"{l_100km:.2f} L/100 km",
                )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            st.error(f"Yanacaq sərfiyyatını hesablamaq mümkün olmadı: {exc}")
