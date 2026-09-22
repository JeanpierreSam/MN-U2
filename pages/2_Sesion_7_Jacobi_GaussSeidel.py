"""
2_Sesion_7_Jacobi_GaussSeidel.py
---------------------------------
Interfaz Streamlit para los metodos iterativos de Jacobi y Gauss-Seidel
(Sistemas de Ecuaciones Lineales), generalizado a n incognitas.

Parte de la suite MN-U2. Ejecutar con:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from iterative_solver import (
    A_GAA_CLUSTER,
    A_GP_GS,
    A_GP_JACOBI,
    B_GAA_CLUSTER,
    B_GP_GS,
    B_GP_JACOBI,
    TOL_GAA_CLUSTER,
    TOL_GP_GS,
    X0_GAA_CLUSTER,
    X0_GP_GS,
    criterio_sassenfeld,
    es_diagonal_dominante,
    gauss_seidel,
    jacobi,
)

st.set_page_config(page_title="Jacobi / Gauss-Seidel - MN-U2", page_icon="🔁", layout="wide")

st.title("🔁 Sistemas de Ecuaciones Lineales — Métodos Iterativos")
st.caption(
    "Métodos Numéricos · Sesión 7 · Jacobi y Gauss-Seidel, con verificación de "
    "dominancia diagonal (Jacobi) y criterio de Sassenfeld (Gauss-Seidel)."
)

EJEMPLOS = {
    "(ninguno)": None,
    "GAA - Sesión 7: Balanceo de carga (4x4, Jacobi)": {
        "A": A_GAA_CLUSTER, "b": B_GAA_CLUSTER, "x0": X0_GAA_CLUSTER,
        "tol": TOL_GAA_CLUSTER, "max_iter": 100, "metodo": "Jacobi",
    },
    "GP - Sesión 7: Ejercicio de Jacobi (3x3, 4 iter.)": {
        "A": A_GP_JACOBI, "b": B_GP_JACOBI, "x0": [0, 0, 0],
        # tol=0.0 es inalcanzable a propósito: el ejercicio de la GP pide
        # EXACTAMENTE 4 iteraciones, no un criterio de parada por error.
        "tol": 0.0, "max_iter": 4, "metodo": "Jacobi",
    },
    "GP - Sesión 7: Ejemplo de Gauss-Seidel (3x3)": {
        "A": A_GP_GS, "b": B_GP_GS, "x0": X0_GP_GS,
        "tol": TOL_GP_GS, "max_iter": 100, "metodo": "Gauss-Seidel",
    },
}

# ---------------------------------------------------------------------
# Aplicar un ejemplo pendiente (elegido en el run anterior) ANTES de
# crear los widgets de abajo: Streamlit no permite mutar el
# session_state de un widget (tol7, maxit7, metodo7) despues de que ya
# fue instanciado en el mismo run, asi que el boton "Cargar ejemplo"
# solo deja la eleccion pendiente y fuerza un rerun (ver mas abajo).
# ---------------------------------------------------------------------
pendiente = st.session_state.pop("_pending_ejemplo7", None)
if pendiente is not None:
    ej = EJEMPLOS[pendiente]
    st.session_state.A7 = ej["A"]
    st.session_state.b7 = ej["b"]
    st.session_state.x07 = ej["x0"]
    st.session_state["tol7"] = ej["tol"]
    st.session_state["maxit7"] = ej["max_iter"]
    st.session_state["metodo7"] = ej["metodo"]

# ---------------------------------------------------------------------
# Barra lateral: configuración
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    n = st.number_input("Número de incógnitas (n)", min_value=2, max_value=12, value=4, step=1, key="n7")
    metodo = st.radio("Método", ["Jacobi", "Gauss-Seidel"], key="metodo7")
    tol = st.number_input("Tolerancia (ε)", value=1e-4, min_value=0.0, format="%.6f", step=1e-5, key="tol7")
    max_iter = st.number_input("Máx. iteraciones", value=100, min_value=1, step=1, key="maxit7")
    error_relativo = st.checkbox("Error relativo (si no, absoluto)", value=True, key="errrel7")
    ejemplo_sel = st.selectbox("Cargar ejemplo de la Sesión 7", list(EJEMPLOS.keys()))
    cargar = st.button("Cargar ejemplo", width="stretch")
    st.markdown("---")
    st.caption(
        "Tip: si A no es diagonal dominante, Gauss-Seidel puede seguir convergiendo — "
        "revisa el criterio de Sassenfeld antes de descartar el sistema."
    )

# ---------------------------------------------------------------------
# Estado inicial / reinicio cuando cambia n
# ---------------------------------------------------------------------
if "n7_state" not in st.session_state or st.session_state.n7_state != n:
    st.session_state.n7_state = n
    st.session_state.A7 = np.zeros((n, n)).tolist()
    st.session_state.b7 = np.zeros(n).tolist()
    st.session_state.x07 = np.zeros(n).tolist()

if cargar and EJEMPLOS[ejemplo_sel] is not None:
    ej = EJEMPLOS[ejemplo_sel]
    if len(ej["A"]) == n:
        st.session_state["_pending_ejemplo7"] = ejemplo_sel
        st.rerun()
    else:
        st.warning(f"Este ejemplo es {len(ej['A'])}x{len(ej['A'])}. Ajusta n a {len(ej['A'])} para cargarlo.")

# ---------------------------------------------------------------------
# 1. Matriz A y vectores b, x(0)
# ---------------------------------------------------------------------
st.subheader("1️⃣ Matriz A y vectores b, x⁽⁰⁾")
df_A = pd.DataFrame(
    st.session_state.A7,
    columns=[f"x{j + 1}" for j in range(n)],
    index=[f"F{i + 1}" for i in range(n)],
)
A = st.data_editor(df_A, width="stretch", key="editor_A7").to_numpy(dtype=float)

col_b, col_x0 = st.columns(2)
with col_b:
    df_b = pd.DataFrame([st.session_state.b7], columns=[f"b{i + 1}" for i in range(n)])
    b = st.data_editor(df_b, width="stretch", key="editor_b7").to_numpy(dtype=float).flatten()
with col_x0:
    df_x0 = pd.DataFrame([st.session_state.x07], columns=[f"x{i + 1}⁽⁰⁾" for i in range(n)])
    x0 = st.data_editor(df_x0, width="stretch", key="editor_x07").to_numpy(dtype=float).flatten()

# ---------------------------------------------------------------------
# 2. Análisis preliminar de convergencia
# ---------------------------------------------------------------------
st.subheader("2️⃣ Análisis preliminar de convergencia")
dominante, detalle = es_diagonal_dominante(A)
st.dataframe(pd.DataFrame(detalle), width="stretch")

if dominante:
    st.success("A es estrictamente diagonal dominante (EDD) → convergencia garantizada.")
else:
    st.warning(
        "A NO es EDD por filas. La condición es suficiente, no necesaria (GT, Sesión 7): "
        "puede converger igual, pero no está garantizado."
    )
    if metodo == "Gauss-Seidel":
        beta, converge_sassenfeld = criterio_sassenfeld(A)
        st.dataframe(pd.DataFrame({"i": range(1, n + 1), "β_i": beta}), width="stretch")
        if converge_sassenfeld:
            st.info(f"Criterio de Sassenfeld: max(β_i) = {beta.max():.4f} < 1 → Gauss-Seidel converge.")
        else:
            st.error(f"Criterio de Sassenfeld: max(β_i) = {beta.max():.4f} ≥ 1 → no se garantiza convergencia.")

# ---------------------------------------------------------------------
# 3. Iterar
# ---------------------------------------------------------------------
if st.button("🚀 Ejecutar iteraciones", type="primary", width="stretch"):
    try:
        if metodo == "Jacobi":
            x_final, historial, convergio = jacobi(A, b, x0, tol, int(max_iter), error_relativo)
        else:
            x_final, historial, convergio = gauss_seidel(A, b, x0, tol, int(max_iter), error_relativo)

        st.subheader("3️⃣ Historial de iteraciones")
        tabla = []
        for reg in historial:
            fila = {"k": reg["k"]}
            fila.update({f"x{i + 1}": round(float(reg["x"][i]), 6) for i in range(n)})
            fila["error"] = reg["error"]
            tabla.append(fila)
        df_hist = pd.DataFrame(tabla)
        st.dataframe(df_hist, width="stretch")
        st.line_chart(df_hist.set_index("k")["error"])

        st.subheader("4️⃣ Solución final")
        residuo = A @ x_final - b
        st.write("x ≈", np.round(x_final, 6).tolist())
        st.write("‖residuo‖ =", float(np.linalg.norm(residuo)))
        st.write(
            f"{'Convergió' if convergio else 'NO convergió'} en {len(historial)} iteraciones "
            f"(máx. {int(max_iter)})."
        )

        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar historial (CSV)", csv, f"resultados_{metodo.lower()}.csv", "text/csv")

    except ZeroDivisionError:
        st.error("Algún elemento de la diagonal es 0: reordena las ecuaciones (pivoteo por filas).")

st.markdown("---")
st.caption(
    "Basado en la Guía Teórica (GT) y la Guía de Práctica (GP), Sesión 7 — Métodos "
    "Numéricos, UPeU: métodos de Jacobi y Gauss-Seidel, generalizados a n variables con NumPy."
)
