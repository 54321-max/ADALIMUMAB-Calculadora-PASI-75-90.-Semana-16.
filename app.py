# -*- coding: utf-8 -*-
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models_ada")

st.set_page_config(page_title="Calculadora Adalimumab PASI", layout="centered")
st.title("Calculadora predictiva - Adalimumab (semana 16)")
st.caption("Herramienta de apoyo a la decisión clínica. No sustituye el juicio clínico.")

# Inputs
pasi = st.number_input("PASI basal", 0.0, 80.0, 20.0, step=0.5)
edad = st.number_input("Edad (años)", 18, 100, 45, step=1)
imc = st.number_input("IMC", 15.0, 60.0, 27.0, step=0.1)
sexo = st.selectbox("Sexo", ["Varón", "Mujer"])
artritis_txt = st.selectbox("Artritis psoriásica", ["No", "Sí"])
nprev = st.number_input("Nº tratamientos previos", 0, 20, 0, step=1)

if st.button("Calcular probabilidad"):
    m75 = joblib.load(MODELS_DIR / "ada_PASI75_w16.joblib")
    m90 = joblib.load(MODELS_DIR / "ada_PASI90_w16.joblib")

    X = pd.DataFrame([{
        "Sexo": sexo,
        "EDAD": int(edad),
        "IMC": float(imc),
        "PASI INICIAL ADA": float(pasi),
        "ARTRITIS": 1 if artritis_txt == "Sí" else 0,
        "N tratamientos previos": int(nprev),
    }])

    X75 = X.reindex(columns=m75.feature_names_in_)
    X90 = X.reindex(columns=m90.feature_names_in_)

    p75 = m75.predict_proba(X75)[0, 1]
    p90 = m90.predict_proba(X90)[0, 1]

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Probabilidad PASI75 (semana 16)", f"{p75*100:.1f}%")
    with c2:
        st.metric("Probabilidad PASI90 (semana 16)", f"{p90*100:.1f}%")

    # Panel transparencia (si quieres)
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        with st.expander("Transparencia del modelo"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            st.write("**Variables incluidas:** " + ", ".join(meta.get("features", [])))
            st.write("**Métricas internas (validación cruzada):**")
            for k, info in meta.get("models", {}).items():
                st.write(f"- **{k}** | n={info.get('n')} | eventos={info.get('pos')} | AUC={info.get('auc'):.3f} | Brier={info.get('brier'):.3f}")
