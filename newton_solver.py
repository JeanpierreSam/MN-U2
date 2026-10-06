"""
newton_solver.py
----------------
Nucleo numerico de la interpolacion polinomica de Newton (diferencias
divididas), generalizado a n+1 nodos (x_i, y_i) con abscisas distintas.

Basado en la Guia Teorica (GT) de la Sesion 9 - Interpolacion, UPeU:

    f[x_k]                 = f(x_k)
    f[x_k, ..., x_{k+i}]   = (f[x_{k+1}, ..., x_{k+i}] - f[x_k, ..., x_{k+i-1}])
                             / (x_{k+i} - x_k)

    P_n(x) = sum_{k=0}^{n} f[x_0, ..., x_k] * prod_{j=0}^{k-1} (x - x_j)

Procedimiento de la GT: (1) organizar los datos, (2) construir la Tabla de
Diferencias Divididas (TDD), (3) tomar la PRIMERA FILA de cada columna,
(4) construir el polinomio, (5) sustituir el x a interpolar.

Ventaja frente a Lagrange (GT, Introduccion): pasar de grado n a grado
n+1 solo agrega UN termino al polinomio; no se rehace el trabajo.

La evaluacion usa la "regla de evaluacion eficiente" (forma anidada /
Horner de Newton), O(n) por punto:

    P(x) = a0 + (x-x0)(a1 + (x-x1)(a2 + ... + (x-x_{n-1}) a_n))

Incluye tambien:
  - Newton-Gregory progresivo (diferencias finitas, nodos equiespaciados):
    a_i = Delta^i f_0 / (i! h^i)
  - Cota superior del error de truncamiento (GT, "Limitante superior").

Caso GAA: latencia (ms) de una API REST vs. carga concurrente (x 100 req/s),
nodos (1,45), (2,65), (4,110), (7,220) -> P_3(5) = 139 ms.
"""

from math import factorial

import numpy as np


def validar_nodos(xs, ys):
    """
    Verifica que xs e ys sean vectores del mismo tamaño, con al menos 2
    nodos y abscisas distintas (si x_i = x_j la diferencia dividida
    dividiria por cero). Lanza ValueError si algo falla.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or xs.shape != ys.shape:
        raise ValueError("X e Y deben ser vectores del mismo tamaño n.")
    if len(xs) < 2:
        raise ValueError("Se necesitan al menos 2 nodos.")
    if len(np.unique(xs)) != len(xs):
        raise ValueError("Las abscisas x_i deben ser distintas entre si (x_i != x_j).")
    return xs, ys


def diferencias_divididas(X, Y):
    """
    Construye la Tabla de Diferencias Divididas (TDD) completa.

    Retorna una matriz triangular superior F de tamaño n x n con el mismo
    formato que la tabla de la GT:

        F[i, 0] = f[x_i]
        F[i, j] = f[x_i, x_{i+1}, ..., x_{i+j}]      (solo si i + j <= n-1)

    Las celdas por debajo de la anti-diagonal quedan en NaN (vacias en la
    tabla). La PRIMERA FILA F[0, :] contiene los coeficientes de Newton.
    """
    xs, ys = validar_nodos(X, Y)
    n = len(xs)
    F = np.full((n, n), np.nan)
    F[:, 0] = ys
    for j in range(1, n):              # orden de la diferencia (columna)
        for i in range(n - j):         # fila
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (xs[i + j] - xs[i])
    return F


def coeficientes_newton(X, Y):
    """Coeficientes de Newton a_k = f[x_0, ..., x_k] (primera fila de la TDD)."""
    return diferencias_divididas(X, Y)[0, :].copy()


def evaluar_newton(X, coef, x):
    """
    Regla de evaluacion eficiente del polinomio de Newton (forma anidada),
    O(n) por punto:

        p = a_n
        for k = n-1 down to 0:
            p = a_k + (x - x_k) * p

    `x` puede ser escalar o arreglo de NumPy (para graficar la curva).
    """
    xs = np.asarray(X, dtype=float)
    coef = np.asarray(coef, dtype=float)
    x = np.asarray(x, dtype=float)
    p = np.full_like(x, coef[-1])
    for k in range(len(coef) - 2, -1, -1):
        p = coef[k] + (x - xs[k]) * p
    return float(p) if p.ndim == 0 else p


def newton(X, Y, x, grado=None):
    """
    Evalua el polinomio de Newton en x. Si se indica `grado` (< n-1), usa
    solo los primeros grado+1 nodos (es decir, corta la suma en el termino
    `grado`): asi se compara P_1, P_2, P_3... "en tiempo real" sin
    recalcular la tabla.
    """
    xs, _ = validar_nodos(X, Y)
    coef = coeficientes_newton(X, Y)
    if grado is not None:
        coef = coef[: grado + 1]
    return evaluar_newton(xs[: len(coef)], coef, x)


def tabla_terminos(X, Y, x_eval):
    """
    Detalle paso a paso de P_n(x_eval): por cada k, el coeficiente
    f[x_0..x_k], el producto prod_{j<k}(x - x_j), el aporte de ese termino
    y la suma acumulada (= P_k(x_eval), el polinomio de grado k).
    """
    xs, _ = validar_nodos(X, Y)
    coef = coeficientes_newton(X, Y)
    filas = []
    prod = 1.0
    acumulado = 0.0
    for k, a in enumerate(coef):
        if k > 0:
            prod *= (x_eval - xs[k - 1])
        aporte = a * prod
        acumulado += aporte
        filas.append({
            "k": k,
            "f[x0..xk]": float(a),
            "∏(x − x_j), j<k": float(prod),
            "término": float(aporte),
            "acumulado = P_k(x)": float(acumulado),
        })
    return filas


def coeficientes_estandar(X, coef):
    """
    Expande la forma de Newton a la forma estandar
    P_n(x) = a_n x^n + ... + a_1 x + a_0. Retorna [a_n, ..., a_0]
    (convencion de np.polyval), multiplicando los binomios (x - x_j).
    """
    xs = np.asarray(X, dtype=float)
    n = len(coef)
    total = np.zeros(n)
    base = np.array([1.0])
    for k in range(n):
        if k > 0:
            base = np.polymul(base, [1.0, -xs[k - 1]])
        termino = coef[k] * base
        total[n - len(termino):] += termino
    return total


def forma_newton_texto(X, coef, decimales=6):
    """Texto de la forma de Newton: a0 + a1(x - x0) + a2(x - x0)(x - x1) + ..."""
    xs = np.asarray(X, dtype=float)
    partes = []
    for k, a in enumerate(coef):
        a = round(float(a), decimales)
        factores = "".join(f"(x - {xs[j]:g})" for j in range(k))
        partes.append(f"{a:g}{factores}")
    return " + ".join(partes).replace("+ -", "- ")


def polinomio_a_texto(coef, var="x", decimales=6):
    """Representa [a_n, ..., a_0] como texto legible: 'a3 x^3 + ... + a0'."""
    n = len(coef) - 1
    partes = []
    for p, a in enumerate(coef):
        grado = n - p
        a = round(float(a), decimales)
        if a == 0:
            continue
        signo = "-" if a < 0 else "+"
        mag = abs(a)
        num = "" if (mag == 1 and grado > 0) else f"{mag:g}"
        if grado == 0:
            termino = f"{mag:g}"
        elif grado == 1:
            termino = f"{num}{var}"
        else:
            termino = f"{num}{var}^{grado}"
        partes.append((signo, termino))
    if not partes:
        return "0"
    texto = ("-" if partes[0][0] == "-" else "") + partes[0][1]
    for signo, termino in partes[1:]:
        texto += f" {signo} {termino}"
    return texto


# ---------------------------------------------------------------------
# Newton-Gregory (nodos equiespaciados) - GT, diapositivas 20-28
# ---------------------------------------------------------------------
def es_equiespaciado(X, tol=1e-12):
    xs = np.asarray(X, dtype=float)
    h = np.diff(xs)
    return bool(len(h) > 0 and np.all(np.abs(h - h[0]) <= tol * max(1.0, abs(h[0])))), (float(h[0]) if len(h) else 0.0)


def diferencias_finitas(Y):
    """
    Tabla de diferencias finitas PROGRESIVAS:
        D[i, 0] = f_i ;  D[i, j] = D[i+1, j-1] - D[i, j-1]  (= Delta^j f_i)
    """
    ys = np.asarray(Y, dtype=float)
    n = len(ys)
    D = np.full((n, n), np.nan)
    D[:, 0] = ys
    for j in range(1, n):
        for i in range(n - j):
            D[i, j] = D[i + 1, j - 1] - D[i, j - 1]
    return D


def coeficientes_newton_gregory(X, Y):
    """a_i = Delta^i f_0 / (i! h^i). Requiere nodos equiespaciados."""
    xs, ys = validar_nodos(X, Y)
    ok, h = es_equiespaciado(xs)
    if not ok:
        raise ValueError("Newton-Gregory requiere nodos igualmente espaciados.")
    D = diferencias_finitas(ys)
    return np.array([D[0, i] / (factorial(i) * h ** i) for i in range(len(xs))])


# ---------------------------------------------------------------------
# Error de interpolacion - GT, "Limitante superior para el error"
# ---------------------------------------------------------------------
def cota_error(X, x, max_deriv):
    """
    |E_n(x)| <= |x - x0|·...·|x - xn| / (n+1)! · max |f^(n+1)(t)|
    `max_deriv` es el maximo de |f^(n+1)| en el intervalo (lo da el usuario,
    porque solo se puede calcular si se conoce f(x) analiticamente).
    """
    xs = np.asarray(X, dtype=float)
    return float(np.prod(np.abs(x - xs)) / factorial(len(xs)) * max_deriv)


# --- Caso GAA: Latencia de una API REST vs. carga concurrente (Sesion 9) ---
# x en unidades de 100 req/s; y en ms. Evaluar en x = 5 (500 req/s).
X_GAA_LATENCIA = [1, 2, 4, 7]
Y_GAA_LATENCIA = [45, 65, 110, 220]
XEVAL_GAA_LATENCIA = 5
INTERVALO_GAA = (1, 7)

# --- GT: Ejemplo "Newton con diferencias divididas" -> f(x) = x^2 + 1, f(3) = 10
X_GT_EJEMPLO = [1, 2, 4, 5]
Y_GT_EJEMPLO = [2, 5, 17, 26]

# --- GT: Ejercicio puntos (1,2), (3,3), (4,2), (8,10) -> P_3(x)
X_GT_EJ_PUNTOS = [1, 3, 4, 8]
Y_GT_EJ_PUNTOS = [2, 3, 2, 10]

# --- GT: Ejercicio f(x) = e^x + sin x -> P_2(x) = 0.62x^2 + 1.93x + 1, f(0.7) = 2.6548
X_GT_EXP_SIN = [0, 0.5, 1]
Y_GT_EXP_SIN = [1, 2.12, 3.55]

# --- GT: Ejercicios 1 (grado 3 "apropiado", f(4.5)): nodos 3, 4, 5, 6 alrededor de 4.5
X_GT_TABLA = [2, 3, 4, 5, 6, 7]
Y_GT_TABLA = [0.13, 0.19, 0.27, 0.38, 0.51, 0.67]
X_GT_TABLA_G3 = [3, 4, 5, 6]
Y_GT_TABLA_G3 = [0.19, 0.27, 0.38, 0.51]

# --- GT: Newton-Gregory (h = 2): -> P(x) = x^3/2 - x^2 - 41x/2 + 0... (forma normalizada)
X_GT_NG_H2 = [-3, -1, 1, 3, 5, 7, 9]
Y_GT_NG_H2 = [39, 19, -21, -57, -65, -21, 99]

# --- GT: Newton-Gregory (h = 1) -> f(2.5)
X_GT_NG_H1 = [0, 1, 2, 3, 4]
Y_GT_NG_H1 = [-1, -1, 1, 29, 131]

# --- GP: (mismos datos de la Sesion 8, ahora resueltos con Newton) ---
X_GP_CLASE = [-1, 0, 3]          # P2(x) = x^2 - 6x + 8 ; f(1) = 3
Y_GP_CLASE = [15, 8, -1]
X_GP_EJ1 = [0, 0.5, 1.0]         # P2(x) = -5.6x^2 + 5.2x + 1.3 ; f(0.8) = 1.876
Y_GP_EJ1 = [1.3, 2.5, 0.9]
X_GP_EJ2 = [0.1, 0.2, 0.4]       # P2(x) = x^2 - 1.8x + 2.99 ; f(0.25) = 2.6025
Y_GP_EJ2 = [2.82, 2.67, 2.43]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    np.set_printoptions(suppress=True, precision=6)

    print("=== GAA: Latencia vs. carga concurrente (Newton, n = 3) ===")
    F = diferencias_divididas(X_GAA_LATENCIA, Y_GAA_LATENCIA)
    print("TDD:\n", F)
    coef = F[0, :]
    print("Coeficientes de Newton:", coef)              # [45, 20, 0.833333, 0.333333]
    print("Forma de Newton: P3(x) =", forma_newton_texto(X_GAA_LATENCIA, coef))
    est = coeficientes_estandar(X_GAA_LATENCIA, coef)
    print("Forma estandar:  P3(x) =", polinomio_a_texto(est))
    # esperado: 0.333333x^3 - 1.5x^2 + 22.166667x + 24
    for fila in tabla_terminos(X_GAA_LATENCIA, Y_GAA_LATENCIA, XEVAL_GAA_LATENCIA):
        print(fila)
    print("P3(5) =", newton(X_GAA_LATENCIA, Y_GAA_LATENCIA, XEVAL_GAA_LATENCIA))   # 139 ms
    for g in (1, 2, 3):
        print(f"  P{g}(5) =", newton(X_GAA_LATENCIA, Y_GAA_LATENCIA, 5, grado=g))  # 125, 135, 139
    print("Chequeo en los nodos:", newton(X_GAA_LATENCIA, Y_GAA_LATENCIA, np.array(X_GAA_LATENCIA)))
    print("Chequeo con np.polyval:", np.polyval(est, 5))

    print("\n=== GT: ejemplo (1,2),(2,5),(4,17),(5,26) ===")
    print("coef =", coeficientes_newton(X_GT_EJEMPLO, Y_GT_EJEMPLO), "| f(3) =",
          newton(X_GT_EJEMPLO, Y_GT_EJEMPLO, 3))                                     # [2,3,1,0] | 10

    print("\n=== GT: ejercicio (1,2),(3,3),(4,2),(8,10) ===")
    c = coeficientes_newton(X_GT_EJ_PUNTOS, Y_GT_EJ_PUNTOS)
    print("coef =", c, "| P3(x) =", polinomio_a_texto(coeficientes_estandar(X_GT_EJ_PUNTOS, c)))

    print("\n=== GT: e^x + sin x ===")
    c = coeficientes_newton(X_GT_EXP_SIN, Y_GT_EXP_SIN)
    print("coef =", c, "| P2(x) =", polinomio_a_texto(coeficientes_estandar(X_GT_EXP_SIN, c)),
          "| f(0.7) =", newton(X_GT_EXP_SIN, Y_GT_EXP_SIN, 0.7))                    # 2.6548

    print("\n=== GT: ejercicio 1 (grado 3, f(4.5)) ===")
    print("f(4.5) ≈", newton(X_GT_TABLA_G3, Y_GT_TABLA_G3, 4.5))

    print("\n=== GT: Newton-Gregory h = 2 ===")
    c = coeficientes_newton_gregory(X_GT_NG_H2, Y_GT_NG_H2)
    print("a_i =", c, "| P(x) =", polinomio_a_texto(coeficientes_estandar(X_GT_NG_H2, c)))
    # a = [39, -10, -2.5, 0.5, 0, 0, 0] ; P(x) = 0.5x^3 - x^2 - 20.5x + 0

    print("\n=== GT: Newton-Gregory h = 1, f(2.5) ===")
    c = coeficientes_newton_gregory(X_GT_NG_H1, Y_GT_NG_H1)
    print("a_i =", c, "| f(2.5) =", evaluar_newton(X_GT_NG_H1, c, 2.5))

    print("\n=== GT: cota de error, f(x) = e^x - x^2 + 1 en x = 1.2, nodos 0.5, 1, 1.5 ===")
    print("|E2(1.2)| <=", round(cota_error([0.5, 1.0, 1.5], 1.2, np.exp(1.5)), 4))  # 0.0314

    print("\n=== GP (con Newton) ===")
    print("Clase f(1) =", newton(X_GP_CLASE, Y_GP_CLASE, 1))                        # 3
    print("Ej1 f(0.8) =", newton(X_GP_EJ1, Y_GP_EJ1, 0.8))                          # 1.876
    a = newton(X_GP_EJ2, Y_GP_EJ2, 0.25)
    print("Ej2 f(0.25) =", a, "| error abs =", abs(a - 3.25 / 1.25))                 # 2.6025 | 0.0025
