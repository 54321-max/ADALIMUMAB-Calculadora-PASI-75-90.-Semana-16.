# -*- coding: utf-8 -*-
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DATA_PATH = "excel adalimumab.xlsx"
OUTDIR = Path("models_ada")
OUTDIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Columnas del Excel
COL_H = "HOMBRE"
COL_M = "MUJER"
COL_EDAD = "EDAD"
COL_IMC = "IMC"
COL_PASI_BASE = "PASI INICIAL ADA"
COL_ARTRITIS = "ARTRITIS"
COL_NPREV = "N tratamientos previos"

TARGET_75 = "PASI75WK16"
TARGET_90 = "PASI90WK16"

FEATURES = ["Sexo", COL_EDAD, COL_IMC, COL_PASI_BASE, COL_ARTRITIS, COL_NPREV]

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_sexo(df):
    # Convierte HOMBRE/MUJER (0/1) en una sola columna categórica Sexo
    df = df.copy()
    h = pd.to_numeric(df.get(COL_H, pd.Series([None]*len(df))), errors="coerce")
    m = pd.to_numeric(df.get(COL_M, pd.Series([None]*len(df))), errors="coerce")

    sexo = pd.Series([None]*len(df), index=df.index, dtype="object")
    sexo[(h == 1) & (m != 1)] = "Varón"
    sexo[(m == 1) & (h != 1)] = "Mujer"

    # Si viene ya codificado raro, lo dejamos como NaN y se imputará
    df["Sexo"] = sexo
    return df

def build_model(num_cols, cat_cols):
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols),
    ])

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([("prep", preprocess), ("mlp", mlp)])
    # calibración sigmoid (útil en tamaños muestrales clínicos)
    return CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=3)

def eval_oof(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "auc": float(roc_auc_score(y, probs)),
        "prauc": float(average_precision_score(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
    }

def main():
    df = pd.read_excel(DATA_PATH)
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")].copy()

    # Construye Sexo desde HOMBRE/MUJER
    df = build_sexo(df)

    # Numéricos
    num_cols = [COL_EDAD, COL_IMC, COL_PASI_BASE, COL_ARTRITIS, COL_NPREV]
    df = to_numeric(df, num_cols + [TARGET_75, TARGET_90])

    # Filtra filas válidas (target presente)
    df = df[df[TARGET_75].notna() & df[TARGET_90].notna()].copy()

    X = df[FEATURES].copy()

    cat_cols = ["Sexo"]
    num_cols_used = [c for c in FEATURES if c not in cat_cols]

    metadata = {"features": FEATURES, "models": {}}

    for target_name, target_col in [("PASI75_w16", TARGET_75), ("PASI90_w16", TARGET_90)]:
        y = df[target_col].astype(int)
        if y.nunique() < 2:
            print(f"[SKIP] {target_name}: solo una clase.")
            continue

        model = build_model(num_cols_used, cat_cols)
        scores = eval_oof(model, X, y)
        print(f"{target_name}: n={scores['n']} pos={scores['pos']} AUC={scores['auc']:.3f} PR-AUC={scores['prauc']:.3f} Brier={scores['brier']:.3f}")

        model.fit(X, y)
        outpath = OUTDIR / f"ada_{target_name}.joblib"
        joblib.dump(model, outpath)

        metadata["models"][target_name] = {"path": str(outpath), **scores}

    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nListo. Modelos guardados en: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
