# MN-U2 — Métodos Numéricos, Unidad 2

Suite de Métodos Numéricos de la **Unidad 2** del curso (UPeU): una sola
app de **Streamlit multi-página**, con un `app.py` de inicio y una
página por sesión en `pages/`. Cada sesión aporta su propio módulo de
lógica numérica pura (sin interfaz), reutilizado por su página.

## Estructura

```
MN-U2/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                              <- página de inicio (entry point de Streamlit)
├── lu_solver.py                         <- lógica de la Sesión 6 (LU / Doolittle)
├── iterative_solver.py                  <- lógica de la Sesión 7 (Jacobi / Gauss-Seidel)
├── lagrange_solver.py                   <- lógica de la Sesión 8 (Interpolación de Lagrange)
├── newton_solver.py                     <- lógica de la Sesión 9 (Interpolación de Newton)
└── pages/
    ├── 1_Sesion_6_LU.py                 <- interfaz de la Sesión 6
    ├── 2_Sesion_7_Jacobi_GaussSeidel.py <- interfaz de la Sesión 7
    ├── 3_Sesion_8_Interpolacion_Lagrange.py <- interfaz de la Sesión 8
    └── 4_Sesion_9_Interpolacion_Newton.py   <- interfaz de la Sesión 9
```

## Cómo correr la app localmente

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

Streamlit detecta automáticamente las páginas en `pages/` y arma el menú
de navegación en la barra lateral.

## Cómo desplegar en Streamlit Community Cloud

1. Sube el repo a GitHub (ver comandos más abajo).
2. En [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Selecciona el repo `JeanpierreSam/MN-U2`, la rama `main`.
4. En **Main file path** escribe `app.py`.
5. Todas las sesiones quedan disponibles como páginas de la misma app,
   con una sola URL.

## Cómo agregar una nueva sesión

1. Crea su módulo de lógica pura en la raíz, por ejemplo `nueva_metodo.py`.
2. Crea su página en `pages/`, por ejemplo `pages/3_Sesion_8_....py`,
   que importe ese módulo (mismo patrón que las páginas existentes).
3. Agrega el enlace correspondiente en `app.py` con `st.page_link`.
4. Si la sesión necesita una dependencia nueva, agrégala a
   `requirements.txt` (compartido por toda la app).
5. `git add`, `git commit`, `git push`.

## Sesión 6 — Factorización LU

- `lu_solver.py` — descomposición `P·A = L·U` (Doolittle, con pivoteo
  parcial), sustitución hacia adelante/atrás, y `resolver()` para
  reutilizar `L` y `U` con múltiples vectores `b` sin recalcular la
  descomposición.
- En la página **"Sesión 6"**: edita `A` y uno o varios vectores `b`, ve
  `P`, `L`, `U` como tablas coloreadas, la bitácora de la triangulación,
  y la solución de cada sistema con su residuo `‖A x − b‖`.
- Ejercicio de la GAA verificado: `x (b1) = [3, -6, 14]`,
  `x (b2) = [4.75, -10.1667, 21.3333]`.

## Sesión 7 — Jacobi y Gauss-Seidel

- `iterative_solver.py` — `jacobi()`, `gauss_seidel()`, chequeo de
  dominancia diagonal estricta (`es_diagonal_dominante`) y criterio de
  Sassenfeld (`criterio_sassenfeld`), generalizados a n incógnitas.
- **Caso GAA — Balanceo de carga en un clúster de 4 servidores:** el
  sistema es estrictamente diagonal dominante en sus 4 filas, por lo que
  Jacobi converge desde $\mathbf{x}^{(0)}=\mathbf{0}$; con
  $\epsilon=10^{-4}$ (error relativo) converge en 10 iteraciones a
  $\mathbf{x}\approx(2.5136,\,3.3995,\,3.3376,\,3.3416)$.
- En la página **"Sesión 7"**, elige el método (Jacobi / Gauss-Seidel) y
  opcionalmente carga uno de los ejemplos de la GAA o la GP. Antes de
  iterar, la app muestra la verificación de dominancia diagonal (y el
  criterio de Sassenfeld si Gauss-Seidel y A no es EDD).

## Sesión 8 — Interpolación de Lagrange

- `lagrange_solver.py` — `lagrange()` (algoritmo de la GT con dos bucles
  anidados, O(n²) por punto), `coeficientes_base()` / `coeficientes_polinomio()`
  para la forma estándar, y `tabla_evaluacion()` para el paso a paso.
- **Caso GAA — Latencia vs. memoria RAM de un microservicio:** nodos
  (2,150), (4,85), (8,50), (12,70) →
  P₃(x) = −(43/192)x³ + (227/32)x² − (1651/24)x + 261, y
  **P₃(6) = 55.25 ms**.
- En la página **"Sesión 8"**: edita los nodos y x_eval, ve los L_k(x),
  el polinomio en forma estándar, la tabla de evaluación y la gráfica
  (nodos + curva + punto interpolado resaltado). Incluye los ejercicios
  de la GP como ejemplos precargados.

## Sesión 9 — Interpolación de Newton (diferencias divididas)

- `newton_solver.py` — `diferencias_divididas(X, Y)` (matriz triangular de
  la TDD), `evaluar_newton()` (forma anidada, O(n) por punto),
  `coeficientes_estandar()` para la forma a_n x^n + … + a_0,
  `tabla_terminos()` para el paso a paso, Newton-Gregory
  (`diferencias_finitas`, `coeficientes_newton_gregory`) y `cota_error()`.
- **Caso GAA — Latencia de una API REST vs. carga concurrente:** nodos
  (1,45), (2,65), (4,110), (7,220) (x en ×100 req/s) →
  P₃(x) = (1/3)x³ − (3/2)x² + (133/6)x + 24, y **P₃(5) = 139 ms**
  (P₁(5) = 125, P₂(5) = 135).
- En la página **"Sesión 9"**: edita X e Y, ve la TDD, el polinomio en
  forma de Newton y estándar, la evaluación término a término, la
  validación en los nodos y un deslizador para comparar grados en tiempo
  real sobre la gráfica. Incluye los ejercicios de la GT y la GP como
  ejemplos precargados.
