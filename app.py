"""
app.py
------
Página de inicio de la suite de Métodos Numéricos (MN-U2). Cada sesión
vive en `pages/` como una página independiente de la misma app.

Ejecutar con:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Métodos Numéricos - MN-U2", page_icon="🧮", layout="wide")

st.title("🧮 Suite de Métodos Numéricos — Unidad 2")
st.caption("UPeU · Métodos Numéricos · Sistemas de Ecuaciones Lineales")

st.markdown(
    """
Elige una sesión en el menú de la izquierda:

- **Sesión 6 — Factorización LU**: descomposición P·A = L·U (método de
  Doolittle) con pivoteo parcial, generalizada a n incógnitas y
  reutilizable para múltiples vectores b sin recalcular la
  descomposición.
- **Sesión 7 — Jacobi y Gauss-Seidel**: métodos iterativos para A x = b,
  con verificación de dominancia diagonal estricta (EDD) y, cuando A no
  es EDD, el criterio de Sassenfeld como alternativa para Gauss-Seidel.
- **Sesión 8 — Interpolación de Lagrange**: polinomios fundamentales
  L_k(x), polinomio interpolador P_n(x) en forma estándar, evaluación
  paso a paso (algoritmo O(n²)) y gráfica de nodos + curva + punto
  interpolado.
- **Sesión 9 — Interpolación de Newton**: tabla de diferencias divididas,
  polinomio en forma de Newton y estándar, evaluación anidada O(n),
  validación en los nodos y comparación de grados en tiempo real.
"""
)

st.page_link("pages/1_Sesion_6_LU.py", label="Ir a Sesión 6 — Factorización LU", icon="🧮")
st.page_link("pages/2_Sesion_7_Jacobi_GaussSeidel.py", label="Ir a Sesión 7 — Jacobi / Gauss-Seidel", icon="🔁")
st.page_link("pages/3_Sesion_8_Interpolacion_Lagrange.py", label="Ir a Sesión 8 — Interpolación de Lagrange", icon="📈")
st.page_link("pages/4_Sesion_9_Interpolacion_Newton.py", label="Ir a Sesión 9 — Interpolación de Newton", icon="📐")

st.markdown("---")
st.caption("Repositorio: [JeanpierreSam/MN-U2](https://github.com/JeanpierreSam/MN-U2)")
