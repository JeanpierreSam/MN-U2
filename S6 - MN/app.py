"""
app.py
------
Interfaz Streamlit para el solver de Sistemas de Ecuaciones Lineales
por Factorizacion LU (Doolittle), generalizado a n incognitas.

Ejecutar con:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from lu_solver import back_substitution, forward_substitution, lu_decomposition, resolver

st.set_page_config(page_title="Solver LU - n variables", page_icon="🧮", layout="wide")

st.title("🧮 Sistemas de Ecuaciones Lineales — Factorización LU")
st.caption(
    "Métodos Numéricos · Sesión 6 · Método de Doolittle generalizado a n variables, "
    "con pivoteo parcial."
)

def estilizar_matriz(df: pd.DataFrame, color_rgb=(37, 99, 235), piso=0.12):
    """
    Colorea las celdas de una matriz segun su magnitud relativa
    (mas oscuro = mayor |valor|), sin depender de matplotlib.

    Fija ademas el color del texto (negro o blanco) segun la luminancia
    de cada celda, para que se lea bien tanto en tema claro como oscuro
    de Streamlit (si no, el texto hereda el color del tema y puede
    volverse invisible sobre un fondo casi blanco).

    `piso` evita que las celdas con valor 0 (o muy pequeño) queden
    blancas puras y se confundan con el fondo de la pagina.
    """
    arr = df.to_numpy(dtype=float)
    max_abs = float(np.max(np.abs(arr))) or 1.0

    def _row_style(row):
        estilos = []
        for v in row:
            frac = min(abs(float(v)) / max_abs, 1.0)
            frac = piso + frac * (1 - piso)  # nunca queda en blanco puro
            r = int(255 - frac * (255 - color_rgb[0]))
            g = int(255 - frac * (255 - color_rgb[1]))
            b = int(255 - frac * (255 - color_rgb[2]))
            luminancia = 0.299 * r + 0.587 * g + 0.114 * b
            color_texto = "#111111" if luminancia > 150 else "#ffffff"
            estilos.append(
                f"background-color: rgb({r},{g},{b}); "
                f"color: {color_texto}; "
                f"border: 1px solid rgba(128,128,128,0.25);"
            )
        return estilos

    return df.style.format("{:.4f}").apply(_row_style, axis=1)


EJEMPLOS = {
    "(ninguno)": None,
    "GAA - Sesión 6 (3x3)": {
        "A": [[4, 2, 1], [12, 10, 5], [-8, 8, 7]],
        "b_list": [[14, 46, 26], [20, 62, 30]],
    },
    "GP - Sesión 6 (3x3)": {
        "A": [[3, 1, 2], [9, 5, 12], [-3, 3, 11]],
        "b_list": [[11, 49, 31]],
    },
}

# ---------------------------------------------------------------------
# Barra lateral: configuración
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    n = st.number_input("Número de incógnitas (n)", min_value=2, max_value=12, value=3, step=1)
    pivoting = st.checkbox("Usar pivoteo parcial", value=True, help="Recomendado para n grande o pivotes cercanos a 0.")
    ejemplo_sel = st.selectbox("Cargar ejemplo de la Sesión 6", list(EJEMPLOS.keys()))
    cargar = st.button("Cargar ejemplo", width="stretch")
    st.markdown("---")
    st.caption(
        "Tip: agrega varios vectores **b** y resuélvelos todos reutilizando "
        "la misma descomposición LU — esa es la idea central de la Fase 2 "
        "de la GAA (balanceo de carga con tráfico variable)."
    )

# ---------------------------------------------------------------------
# Estado inicial / reinicio cuando cambia n o se carga un ejemplo
# ---------------------------------------------------------------------
if "n" not in st.session_state or st.session_state.n != n:
    st.session_state.n = n
    st.session_state.A = np.zeros((n, n)).tolist()
    st.session_state.b_list = [np.zeros(n).tolist()]

if cargar and EJEMPLOS[ejemplo_sel] is not None:
    ej = EJEMPLOS[ejemplo_sel]
    if len(ej["A"]) == n:
        st.session_state.A = ej["A"]
        st.session_state.b_list = ej["b_list"]
    else:
        st.warning(f"Este ejemplo es {len(ej['A'])}x{len(ej['A'])}. Ajusta n a {len(ej['A'])} para cargarlo.")

# ---------------------------------------------------------------------
# 1. Matriz A
# ---------------------------------------------------------------------
st.subheader("1️⃣ Matriz de coeficientes A")
df_A = pd.DataFrame(
    st.session_state.A,
    columns=[f"x{j + 1}" for j in range(n)],
    index=[f"F{i + 1}" for i in range(n)],
)
edited_A = st.data_editor(df_A, width="stretch", key="editor_A")
A = edited_A.to_numpy(dtype=float)
st.session_state.A = A.tolist()

# ---------------------------------------------------------------------
# 2. Vectores b
# ---------------------------------------------------------------------
st.subheader("2️⃣ Vector(es) de términos independientes b")
st.caption("Puedes agregar varios vectores b y resolverlos todos con la misma L y U.")

col_add, col_rm = st.columns(2)
with col_add:
    if st.button("➕ Agregar vector b", width="stretch"):
        st.session_state.b_list.append(np.zeros(n).tolist())
with col_rm:
    if st.button("➖ Quitar último b", width="stretch") and len(st.session_state.b_list) > 1:
        st.session_state.b_list.pop()

b_arrays = []
for idx in range(len(st.session_state.b_list)):
    bvec = st.session_state.b_list[idx]
    if len(bvec) != n:
        bvec = np.zeros(n).tolist()
    df_b = pd.DataFrame([bvec], columns=[f"b{idx + 1}·{i + 1}" for i in range(n)], index=[f"b{idx + 1}"])
    edited_b = st.data_editor(df_b, width="stretch", key=f"editor_b_{idx}")
    arr = edited_b.to_numpy(dtype=float).flatten()
    st.session_state.b_list[idx] = arr.tolist()
    b_arrays.append(arr)

# ---------------------------------------------------------------------
# 3. Calcular
# ---------------------------------------------------------------------
if st.button("🔷 Descomponer (LU) y resolver todo", type="primary", width="stretch"):
    try:
        P, L, U, steps = lu_decomposition(A, pivoting=pivoting)
        st.success("Descomposición LU calculada correctamente (una sola vez).")

        st.subheader("3️⃣ Matrices resultantes")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**P** (permutación)")
            st.dataframe(pd.DataFrame(P).style.format("{:.0f}"), width="stretch")
        with c2:
            st.markdown("**L** (triangular inferior)")
            st.dataframe(
                estilizar_matriz(pd.DataFrame(L), color_rgb=(37, 99, 235)),
                width="stretch",
            )
        with c3:
            st.markdown("**U** (triangular superior)")
            st.dataframe(
                estilizar_matriz(pd.DataFrame(U), color_rgb=(234, 88, 12)),
                width="stretch",
            )

        with st.expander("Ver bitácora paso a paso de la triangulación"):
            for s in steps:
                st.text(s)

        st.subheader("4️⃣ Solución para cada vector b (reutilizando L y U — sin recalcular)")
        resumen = []
        for idx, bvec in enumerate(b_arrays):
            x, y = resolver(P, L, U, bvec)
            residuo = A @ x - bvec
            fila = {"b": np.round(bvec, 4).tolist()}
            fila.update({f"x{i + 1}": round(float(x[i]), 6) for i in range(n)})
            fila["‖residuo‖"] = float(np.linalg.norm(residuo))
            resumen.append(fila)

            with st.container(border=True):
                st.markdown(f"**Vector b{idx + 1} = {np.round(bvec, 4).tolist()}**")
                cy, cx = st.columns(2)
                with cy:
                    st.write("y (de  L y = P b):", np.round(y, 4).tolist())
                with cx:
                    st.write("x (de  U x = y):", np.round(x, 4).tolist())

        st.subheader("5️⃣ Tabla resumen")
        st.dataframe(pd.DataFrame(resumen), width="stretch")

        csv = pd.DataFrame(resumen).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar resultados (CSV)", csv, "resultados_LU.csv", "text/csv")

    except ValueError as e:
        st.error(str(e))

st.markdown("---")
st.caption(
    "Basado en el método de Doolittle (Guía Teórica, Sesión 6 — Métodos Numéricos, UPeU), "
    "generalizado a n variables con NumPy + pivoteo parcial."
)
