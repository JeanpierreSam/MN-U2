"""
4_Sesion_9_Interpolacion_Newton.py
----------------------------------
Interfaz Streamlit para la interpolacion polinomica de Newton con
diferencias divididas, generalizada a n+1 nodos (por defecto 4 nodos ->
P_3, como pide la GAA). Incluye el comparador de grado en tiempo real.

Parte de la suite MN-U2. Ejecutar con:
    streamlit run app.py
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from newton_solver import (
    INTERVALO_GAA,
    X_GAA_LATENCIA,
    X_GP_CLASE,
    X_GP_EJ1,
    X_GP_EJ2,
    X_GT_EJ_PUNTOS,
    X_GT_EJEMPLO,
    X_GT_EXP_SIN,
    X_GT_NG_H1,
    X_GT_TABLA_G3,
    XEVAL_GAA_LATENCIA,
    Y_GAA_LATENCIA,
    Y_GP_CLASE,
    Y_GP_EJ1,
    Y_GP_EJ2,
    Y_GT_EJ_PUNTOS,
    Y_GT_EJEMPLO,
    Y_GT_EXP_SIN,
    Y_GT_NG_H1,
    Y_GT_TABLA_G3,
    coeficientes_estandar,
    diferencias_divididas,
    evaluar_newton,
    forma_newton_texto,
    polinomio_a_texto,
    tabla_terminos,
)

st.set_page_config(page_title="Interpolación de Newton - MN-U2", page_icon="📐", layout="wide")

st.title("📐 Interpolación Polinómica — Newton (Diferencias Divididas)")
st.caption(
    "Métodos Numéricos · Sesión 9 · Tabla de diferencias divididas, polinomio "
    "P_n(x) = Σ f[x0..xk]·∏(x − x_j), evaluación anidada O(n) y comparación de grados."
)

EJEMPLOS = {
    "(ninguno)": None,
    "GAA - Sesión 9: Latencia API vs. carga (4 nodos, x=5)": {
        "x": X_GAA_LATENCIA, "y": Y_GAA_LATENCIA, "xeval": XEVAL_GAA_LATENCIA,
        "a": INTERVALO_GAA[0], "b": INTERVALO_GAA[1],
    },
    "GT - Ejemplo (1,2),(2,5),(4,17),(5,26), x=3": {
        "x": X_GT_EJEMPLO, "y": Y_GT_EJEMPLO, "xeval": 3, "a": 1, "b": 5,
    },
    "GT - Ejercicio (1,2),(3,3),(4,2),(8,10)": {
        "x": X_GT_EJ_PUNTOS, "y": Y_GT_EJ_PUNTOS, "xeval": 5, "a": 1, "b": 8,
    },
    "GT - f(x)=e^x+sen x, x=0.7": {
        "x": X_GT_EXP_SIN, "y": Y_GT_EXP_SIN, "xeval": 0.7, "a": 0, "b": 1,
    },
    "GT - Ejercicio 1 (grado 3, f(4.5))": {
        "x": X_GT_TABLA_G3, "y": Y_GT_TABLA_G3, "xeval": 4.5, "a": 3, "b": 6,
    },
    "GT - Newton-Gregory h=1, f(2.5)": {
        "x": X_GT_NG_H1, "y": Y_GT_NG_H1, "xeval": 2.5, "a": 0, "b": 4,
    },
    "GP - Problema para la clase, f(1)": {
        "x": X_GP_CLASE, "y": Y_GP_CLASE, "xeval": 1, "a": -1, "b": 3,
    },
    "GP - Ejercicio f(0.8)": {
        "x": X_GP_EJ1, "y": Y_GP_EJ1, "xeval": 0.8, "a": 0, "b": 1,
    },
    "GP - f(x)=(3+x)/(1+x), f(0.25)": {
        "x": X_GP_EJ2, "y": Y_GP_EJ2, "xeval": 0.25, "a": 0.1, "b": 0.4,
    },
}

# ---------------------------------------------------------------------
# Aplicar un ejemplo pendiente ANTES de crear los widgets (mismo patron
# que las Sesiones 7 y 8).
# ---------------------------------------------------------------------
for _clave, _valor in {"n9": 4, "xeval9": 5.0, "a9": 1.0, "b9": 7.0}.items():
    st.session_state.setdefault(_clave, _valor)  # defaults = caso GAA

pendiente = st.session_state.pop("_pending_ejemplo9", None)
if pendiente is not None:
    ej = EJEMPLOS[pendiente]
    st.session_state["n9"] = len(ej["x"])
    st.session_state.n9_state = len(ej["x"])
    st.session_state.nodos9 = {"x": [float(v) for v in ej["x"]], "y": [float(v) for v in ej["y"]]}
    st.session_state["xeval9"] = float(ej["xeval"])
    st.session_state["a9"] = float(ej["a"])
    st.session_state["b9"] = float(ej["b"])
    st.session_state.pop("grado9", None)

# ---------------------------------------------------------------------
# Barra lateral: configuracion
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    n_nodos = st.number_input("Número de nodos n", min_value=2, max_value=12, step=1, key="n9")
    x_eval = st.number_input("Punto a interpolar x_eval", format="%.4f", key="xeval9")
    st.markdown("**Intervalo de la gráfica**")
    a_graf = st.number_input("Desde", format="%.4f", key="a9")
    b_graf = st.number_input("Hasta", format="%.4f", key="b9")
    ejemplo_sel = st.selectbox("Cargar ejemplo de la Sesión 9", list(EJEMPLOS.keys()))
    if st.button("Cargar ejemplo", width="stretch") and EJEMPLOS[ejemplo_sel] is not None:
        st.session_state["_pending_ejemplo9"] = ejemplo_sel
        st.rerun()

# ---------------------------------------------------------------------
# Estado inicial / reinicio cuando cambia el numero de nodos
# ---------------------------------------------------------------------
if "n9_state" not in st.session_state:
    st.session_state.n9_state = 4
    st.session_state.nodos9 = {"x": [float(v) for v in X_GAA_LATENCIA], "y": [float(v) for v in Y_GAA_LATENCIA]}
if st.session_state.n9_state != n_nodos:
    st.session_state.n9_state = n_nodos
    st.session_state.nodos9 = {"x": [float(i) for i in range(n_nodos)], "y": [0.0] * n_nodos}

# ---------------------------------------------------------------------
# 1. Entrada de datos: vectores X e Y
# ---------------------------------------------------------------------
st.subheader("1️⃣ Entrada de datos: vectores X e Y")
df_nodos = pd.DataFrame(
    st.session_state.nodos9,
    index=[f"x{i}" for i in range(n_nodos)],
).astype(float)
df_edit = st.data_editor(
    df_nodos,
    width="stretch",
    key=f"editor_nodos9_{n_nodos}",
    column_config={
        "x": st.column_config.NumberColumn("X", format="%g", step=0.01),
        "y": st.column_config.NumberColumn("Y", format="%g", step=0.01),
    },
)
xs = df_edit["x"].to_numpy(dtype=float)
ys = df_edit["y"].to_numpy(dtype=float)

if np.isnan(xs).any() or np.isnan(ys).any():
    st.error("Completa todas las celdas de X e Y.")
    st.stop()
if len(np.unique(xs)) != len(xs):
    st.error("Las abscisas x_i deben ser distintas: si x_i = x_j la diferencia dividida divide entre cero.")
    st.stop()
if a_graf >= b_graf:
    st.error("El intervalo de la gráfica debe cumplir 'Desde' < 'Hasta'.")
    st.stop()

grado_max = n_nodos - 1

# ---------------------------------------------------------------------
# 2. Tabla de diferencias divididas
# ---------------------------------------------------------------------
st.subheader("2️⃣ Tabla de Diferencias Divididas (TDD)")
F = diferencias_divididas(xs, ys)
cols = ["f[x_i]"] + [f"{j}.ª diferencia" for j in range(1, n_nodos)]
df_F = pd.DataFrame(F, columns=cols)
df_F.insert(0, "x_i", xs)
st.dataframe(
    df_F.style.format(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    .apply(lambda fila: ["background-color: #fff3cd; font-weight: bold" if fila.name == 0 and c != "x_i" else ""
                         for c in fila.index], axis=1),
    width="stretch", hide_index=True,
)
st.caption("La primera fila (resaltada) contiene los coeficientes de Newton f[x0], f[x0,x1], f[x0,x1,x2], …")
coef = F[0, :]

# ---------------------------------------------------------------------
# 3. Polinomio en forma de Newton y en forma estandar
# ---------------------------------------------------------------------
st.subheader(f"3️⃣ Polinomio interpolador P_{grado_max}(x)")
st.markdown(f"**Forma de Newton:** P_{grado_max}(x) = {forma_newton_texto(xs, coef)}")
est = coeficientes_estandar(xs, coef)
st.markdown(f"**Forma estándar:** P_{grado_max}(x) = {polinomio_a_texto(est)}")
st.dataframe(
    pd.DataFrame({f"a{grado_max - p}": [c] for p, c in enumerate(est)}),
    width="stretch", hide_index=True,
)

# ---------------------------------------------------------------------
# 4. Evaluacion en x_eval + validacion en los nodos
# ---------------------------------------------------------------------
st.subheader(f"4️⃣ Evaluación en x_eval = {x_eval:g}")
st.dataframe(pd.DataFrame(tabla_terminos(xs, ys, x_eval)), width="stretch", hide_index=True)
y_eval = evaluar_newton(xs, coef, x_eval)
st.metric(f"P_{grado_max}({x_eval:g})", f"{y_eval:.6f}")

if not (xs.min() <= x_eval <= xs.max()):
    st.warning("x_eval está fuera del rango de los nodos: eso es EXTRAPOLACIÓN, el error puede ser grande.")

with st.expander("✅ Validación: P_n(x_i) = y_i en los nodos", expanded=True):
    p_nodos = evaluar_newton(xs, coef, xs)
    df_val = pd.DataFrame({"x_i": xs, "y_i": ys, "P_n(x_i)": p_nodos, "|P_n(x_i) − y_i|": np.abs(p_nodos - ys)})
    st.dataframe(df_val, width="stretch", hide_index=True)
    if np.allclose(p_nodos, ys, atol=1e-9):
        st.success("El polinomio pasa exactamente por todos los nodos.")
    else:
        st.error("El polinomio no reproduce algún nodo (revisar datos).")

# ---------------------------------------------------------------------
# 5. Comparar grados en tiempo real + grafica
# ---------------------------------------------------------------------
st.subheader("5️⃣ Comparación de grado en tiempo real y gráfica")
grado = st.slider("Grado del polinomio a mostrar (usa los primeros grado+1 nodos)",
                  min_value=1, max_value=grado_max, value=grado_max, key="grado9")

df_comp = pd.DataFrame({
    "grado k": list(range(1, grado_max + 1)),
    "nodos usados": [", ".join(f"{v:g}" for v in xs[: k + 1]) for k in range(1, grado_max + 1)],
    f"P_k({x_eval:g})": [evaluar_newton(xs[: k + 1], coef[: k + 1], x_eval) for k in range(1, grado_max + 1)],
})
df_comp["cambio vs. grado anterior"] = df_comp[f"P_k({x_eval:g})"].diff()
st.dataframe(df_comp, width="stretch", hide_index=True)
st.caption("Ventaja de Newton: cada grado nuevo solo agrega un término a la suma; no se rehace la tabla.")

x_curva = np.linspace(a_graf, b_graf, 400)
filas_curva = []
for k in sorted({grado, grado_max}):
    yk = evaluar_newton(xs[: k + 1], coef[: k + 1], x_curva)
    filas_curva.append(pd.DataFrame({"x": x_curva, "y": yk, "polinomio": f"P_{k}(x)"}))
df_curva = pd.concat(filas_curva, ignore_index=True)
y_sel = evaluar_newton(xs[: grado + 1], coef[: grado + 1], x_eval)
df_nodos_g = pd.DataFrame({"x": xs, "y": ys})
df_eval = pd.DataFrame({"x": [x_eval], "y": [y_sel], "etiqueta": [f"P_{grado}({x_eval:g}) = {y_sel:.4f}"]})

_y_todo = np.concatenate([df_curva["y"].to_numpy(), ys])
_marg = 0.08 * max(float(_y_todo.max() - _y_todo.min()), 1e-9)
_esc_x = alt.Scale(domain=[float(a_graf), float(b_graf)], nice=False)
_esc_y = alt.Scale(domain=[float(_y_todo.min() - _marg), float(_y_todo.max() + _marg)])

curvas = alt.Chart(df_curva).mark_line(strokeWidth=2).encode(
    x=alt.X("x:Q", title="x", scale=_esc_x),
    y=alt.Y("y:Q", title="P(x)", scale=_esc_y),
    color=alt.Color("polinomio:N", title=None),
    strokeDash=alt.condition(alt.datum.polinomio == f"P_{grado_max}(x)", alt.value([1, 0]), alt.value([6, 4])),
)
nodos = alt.Chart(df_nodos_g).mark_circle(size=110, color="#1f77b4").encode(
    x=alt.X("x:Q", scale=_esc_x), y=alt.Y("y:Q", scale=_esc_y), tooltip=["x", "y"]
)
punto = alt.Chart(df_eval).mark_point(shape="diamond", size=260, filled=True, color="#d62728").encode(
    x=alt.X("x:Q", scale=_esc_x), y=alt.Y("y:Q", scale=_esc_y), tooltip=["x", "y"]
)
etiqueta = alt.Chart(df_eval).mark_text(dy=-16, color="#d62728", fontWeight="bold").encode(
    x=alt.X("x:Q", scale=_esc_x), y=alt.Y("y:Q", scale=_esc_y), text="etiqueta:N"
)
guia = alt.Chart(df_eval).mark_rule(strokeDash=[4, 4], color="#d62728").encode(x=alt.X("x:Q", scale=_esc_x))
st.altair_chart(curvas + nodos + guia + punto + etiqueta, width="stretch")

csv = df_curva.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar curvas (CSV)", csv, "curva_newton.csv", "text/csv")

st.markdown("---")
st.caption(
    "Basado en la Guía Teórica (GT), la Guía de Práctica (GP) y la GAA, Sesión 9 — "
    "Interpolación, UPeU: diferencias divididas de Newton generalizadas a n nodos con NumPy."
)
