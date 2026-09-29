"""
3_Sesion_8_Interpolacion_Lagrange.py
------------------------------------
Interfaz Streamlit para la interpolacion polinomica de Lagrange,
generalizada a n+1 nodos (por defecto 4 nodos -> P_3, como pide la GAA).

Parte de la suite MN-U2. Ejecutar con:
    streamlit run app.py
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lagrange_solver import (
    INTERVALO_GAA,
    X_GAA_LATENCIA,
    X_GP_CLASE,
    X_GP_EJ1,
    X_GP_EJ2,
    XEVAL_GAA_LATENCIA,
    Y_GAA_LATENCIA,
    Y_GP_CLASE,
    Y_GP_EJ1,
    Y_GP_EJ2,
    coeficientes_base,
    coeficientes_polinomio,
    lagrange,
    polinomio_a_texto,
    tabla_evaluacion,
)

st.set_page_config(page_title="Interpolación de Lagrange - MN-U2", page_icon="📈", layout="wide")

st.title("📈 Interpolación Polinómica — Lagrange")
st.caption(
    "Métodos Numéricos · Sesión 8 · Polinomios fundamentales L_k(x), polinomio "
    "interpolador P_n(x) = Σ y_k·L_k(x) y evaluación con el algoritmo O(n²) de la GT."
)

EJEMPLOS = {
    "(ninguno)": None,
    "GAA - Sesión 8: Latencia vs. RAM (4 nodos, x=6)": {
        "x": X_GAA_LATENCIA, "y": Y_GAA_LATENCIA, "xeval": XEVAL_GAA_LATENCIA,
        "a": INTERVALO_GAA[0], "b": INTERVALO_GAA[1],
    },
    "GP - Sesión 8: Problema para la clase (3 nodos, x=1)": {
        "x": X_GP_CLASE, "y": Y_GP_CLASE, "xeval": 1, "a": -1, "b": 3,
    },
    "GP - Sesión 8: Ejercicio f(0.8) (3 nodos)": {
        "x": X_GP_EJ1, "y": Y_GP_EJ1, "xeval": 0.8, "a": 0, "b": 1,
    },
    "GP - Sesión 8: f(x)=(3+x)/(1+x), f(0.25) (3 nodos)": {
        "x": X_GP_EJ2, "y": Y_GP_EJ2, "xeval": 0.25, "a": 0.1, "b": 0.4,
    },
}

# ---------------------------------------------------------------------
# Aplicar un ejemplo pendiente ANTES de crear los widgets (mismo patron
# que la Sesion 7: Streamlit no deja mutar el session_state de un widget
# ya instanciado en el mismo run).
# ---------------------------------------------------------------------
for _clave, _valor in {"n8": 4, "xeval8": 6.0, "a8": 2.0, "b8": 12.0}.items():
    st.session_state.setdefault(_clave, _valor)  # defaults = caso GAA

pendiente = st.session_state.pop("_pending_ejemplo8", None)
if pendiente is not None:
    ej = EJEMPLOS[pendiente]
    st.session_state["n8"] = len(ej["x"])
    st.session_state.n8_state = len(ej["x"])
    st.session_state.nodos8 = {"x": list(ej["x"]), "y": list(ej["y"])}
    st.session_state["xeval8"] = float(ej["xeval"])
    st.session_state["a8"] = float(ej["a"])
    st.session_state["b8"] = float(ej["b"])

# ---------------------------------------------------------------------
# Barra lateral: configuracion
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    n_nodos = st.number_input("Número de nodos (n+1)", min_value=2, max_value=12, step=1, key="n8")
    x_eval = st.number_input("Punto a evaluar x_eval", format="%.4f", key="xeval8")
    st.markdown("**Intervalo de la gráfica**")
    a_graf = st.number_input("Desde", format="%.4f", key="a8")
    b_graf = st.number_input("Hasta", format="%.4f", key="b8")
    ejemplo_sel = st.selectbox("Cargar ejemplo de la Sesión 8", list(EJEMPLOS.keys()))
    if st.button("Cargar ejemplo", width="stretch") and EJEMPLOS[ejemplo_sel] is not None:
        st.session_state["_pending_ejemplo8"] = ejemplo_sel
        st.rerun()

# ---------------------------------------------------------------------
# Estado inicial / reinicio cuando cambia el numero de nodos
# ---------------------------------------------------------------------
if "n8_state" not in st.session_state:
    st.session_state.n8_state = 4
    st.session_state.nodos8 = {"x": list(X_GAA_LATENCIA), "y": list(Y_GAA_LATENCIA)}
if st.session_state.n8_state != n_nodos:
    st.session_state.n8_state = n_nodos
    st.session_state.nodos8 = {"x": [float(i) for i in range(n_nodos)], "y": [0.0] * n_nodos}

# ---------------------------------------------------------------------
# 1. Entrada de nodos
# ---------------------------------------------------------------------
st.subheader("1️⃣ Nodos (x_i, y_i)")
df_nodos = pd.DataFrame(
    st.session_state.nodos8,
    index=[f"x{i}" for i in range(n_nodos)],
).astype(float)  # columnas decimales: si fueran int, el editor rechazaria 2.5
df_edit = st.data_editor(
    df_nodos,
    width="stretch",
    key=f"editor_nodos8_{n_nodos}",
    column_config={
        "x": st.column_config.NumberColumn("x", format="%g", step=0.01),
        "y": st.column_config.NumberColumn("y", format="%g", step=0.01),
    },
)
xs = df_edit["x"].to_numpy(dtype=float)
ys = df_edit["y"].to_numpy(dtype=float)

if len(np.unique(xs)) != len(xs):
    st.error("Las abscisas x_i deben ser distintas (Teorema de existencia y unicidad, GT).")
    st.stop()
if a_graf >= b_graf:
    st.error("El intervalo de la gráfica debe cumplir 'Desde' < 'Hasta'.")
    st.stop()

grado = n_nodos - 1

# ---------------------------------------------------------------------
# 2. Polinomios base y polinomio interpolador
# ---------------------------------------------------------------------
st.subheader(f"2️⃣ Polinomios fundamentales L_k(x) y P_{grado}(x)")
filas_base = []
for k in range(n_nodos):
    ck, den = coeficientes_base(xs, k)
    factores = " · ".join(f"(x − {xs[i]:g})" for i in range(n_nodos) if i != k)
    filas_base.append({
        "k": k,
        "L_k(x) (forma producto)": f"{factores} / {den:g}",
        "L_k(x) (forma estándar)": polinomio_a_texto(ck),
    })
st.dataframe(pd.DataFrame(filas_base), width="stretch", hide_index=True)

coef = coeficientes_polinomio(xs, ys)
st.markdown(f"**P_{grado}(x) = {polinomio_a_texto(coef)}**")
st.dataframe(
    pd.DataFrame({f"a{grado - p}": [c] for p, c in enumerate(coef)}),
    width="stretch", hide_index=True,
)

# ---------------------------------------------------------------------
# 3. Evaluacion paso a paso
# ---------------------------------------------------------------------
st.subheader(f"3️⃣ Evaluación en x_eval = {x_eval:g}")
st.dataframe(pd.DataFrame(tabla_evaluacion(xs, ys, x_eval)), width="stretch", hide_index=True)
y_eval = lagrange(xs, ys, x_eval)
st.metric(f"P_{grado}({x_eval:g})", f"{y_eval:.6f}")
if not (xs.min() <= x_eval <= xs.max()):
    st.warning("x_eval está fuera del rango de los nodos: eso es EXTRAPOLACIÓN, el error puede ser grande.")

# ---------------------------------------------------------------------
# 4. Grafica: nodos + curva P_n(x) + punto interpolado resaltado
# ---------------------------------------------------------------------
st.subheader("4️⃣ Gráfica")
x_curva = np.linspace(a_graf, b_graf, 400)
df_curva = pd.DataFrame({"x": x_curva, "y": lagrange(xs, ys, x_curva)})
df_nodos_g = pd.DataFrame({"x": xs, "y": ys})
df_eval = pd.DataFrame({"x": [x_eval], "y": [y_eval], "etiqueta": [f"({x_eval:g}, {y_eval:.4f})"]})

_y_todo = np.concatenate([df_curva["y"].to_numpy(), ys])
_marg = 0.08 * max(float(_y_todo.max() - _y_todo.min()), 1e-9)
_esc_x = alt.Scale(domain=[float(a_graf), float(b_graf)], nice=False)
_esc_y = alt.Scale(domain=[float(_y_todo.min() - _marg), float(_y_todo.max() + _marg)])
curva = alt.Chart(df_curva).mark_line(strokeWidth=2).encode(
    x=alt.X("x:Q", title="x", scale=_esc_x), y=alt.Y("y:Q", title=f"P_{grado}(x)", scale=_esc_y)
)
nodos = alt.Chart(df_nodos_g).mark_circle(size=110, color="#1f77b4").encode(
    x=alt.X("x:Q", scale=_esc_x), y=alt.Y("y:Q", scale=_esc_y), tooltip=["x", "y"]
)
punto = alt.Chart(df_eval).mark_point(
    shape="diamond", size=260, filled=True, color="#d62728"
).encode(
    x=alt.X("x:Q", scale=_esc_x), y=alt.Y("y:Q", scale=_esc_y), tooltip=["x", "y"]
)
etiqueta = alt.Chart(df_eval).mark_text(dy=-16, color="#d62728", fontWeight="bold").encode(
    x=alt.X("x:Q", scale=_esc_x), y=alt.Y("y:Q", scale=_esc_y), text="etiqueta:N"
)
guia = alt.Chart(df_eval).mark_rule(strokeDash=[4, 4], color="#d62728").encode(x=alt.X("x:Q", scale=_esc_x))
st.altair_chart((curva + nodos + guia + punto + etiqueta), width="stretch")

csv = df_curva.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar curva P(x) (CSV)", csv, "curva_lagrange.csv", "text/csv")

st.markdown("---")
st.caption(
    "Basado en la Guía Teórica (GT), la Guía de Práctica (GP) y la GAA, Sesión 8 — "
    "Interpolación, UPeU: algoritmo de Lagrange generalizado a n+1 nodos con NumPy."
)
