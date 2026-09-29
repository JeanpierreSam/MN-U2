"""
lagrange_solver.py
------------------
Nucleo numerico de la interpolacion polinomica de Lagrange, generalizado
a n+1 nodos (x_i, y_i) con abscisas distintas.

Basado en la Guia Teorica (GT) de la Sesion 8 - Interpolacion, UPeU:

    L_k(x) = prod_{i=0, i!=k}^{n} (x - x_i) / (x_k - x_i)
    P_n(x) = sum_{k=0}^{n} y_k * L_k(x)

La evaluacion sigue al pie de la letra el "Algoritmo de Lagrange" de la
GT (dos bucles anidados -> O(n^2) por punto evaluado):

    Soma = 0
    for k = 0 to n
        prod1 = 1 ; prod2 = 1
        for i = 0 to n
            if i != k:
                prod1 = prod1 * (x - x[i])
                prod2 = prod2 * (x[k] - x[i])
        L = prod1 / prod2
        Soma = Soma + L * f[k]

Ademas se obtienen los coeficientes de la forma estandar
P_n(x) = a_n x^n + ... + a_1 x + a_0 expandiendo cada L_k(x) (producto de
binomios), para mostrar el polinomio simplificado que pide la Parte A de
la GAA.

Caso GAA: latencia (ms) vs. memoria RAM (GB) de un microservicio,
4 nodos (2,150), (4,85), (8,50), (12,70) -> P_3(6) = 55.25 ms.
"""

import numpy as np


def validar_nodos(xs, ys):
    """
    Verifica que haya al menos 2 nodos, que xs e ys tengan el mismo largo
    y que las abscisas sean distintas (condicion del Teorema de
    Existencia y Unicidad, GT p.10). Lanza ValueError si algo falla.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or xs.shape != ys.shape:
        raise ValueError("xs e ys deben ser vectores del mismo tamaño.")
    if len(xs) < 2:
        raise ValueError("Se necesitan al menos 2 nodos.")
    if len(np.unique(xs)) != len(xs):
        raise ValueError("Las abscisas x_i deben ser distintas entre si (x_i != x_j).")
    return xs, ys


def base_lagrange(xs, k, x):
    """
    Evalua el polinomio fundamental L_k en x (escalar o arreglo).
    Bucle interno del algoritmo de la GT.
    """
    xs = np.asarray(xs, dtype=float)
    prod1 = np.ones_like(np.asarray(x, dtype=float))
    prod2 = 1.0
    for i in range(len(xs)):
        if i != k:
            prod1 = prod1 * (x - xs[i])
            prod2 = prod2 * (xs[k] - xs[i])
    return prod1 / prod2


def lagrange(xs, ys, x):
    """
    Evalua P_n(x) con el Algoritmo de Lagrange de la GT: dos bucles
    anidados (k externo, i interno) -> O(n^2) operaciones por punto.

    `x` puede ser un escalar o un arreglo de NumPy (para graficar la
    curva completa en un solo llamado).
    """
    xs, ys = validar_nodos(xs, ys)
    n = len(xs)
    x = np.asarray(x, dtype=float)
    soma = np.zeros_like(x)
    for k in range(n):                 # bucle externo
        prod1 = np.ones_like(x)
        prod2 = 1.0
        for i in range(n):             # bucle interno
            if i != k:
                prod1 = prod1 * (x - xs[i])
                prod2 = prod2 * (xs[k] - xs[i])
        L = prod1 / prod2
        soma = soma + L * ys[k]
    return float(soma) if soma.ndim == 0 else soma


def tabla_evaluacion(xs, ys, x_eval):
    """
    Detalle paso a paso de la evaluacion en x_eval (lo que la GAA pide
    "mostrar de manera ordenada"): por cada k, numerador, denominador,
    L_k(x_eval) y el aporte y_k * L_k(x_eval).
    """
    xs, ys = validar_nodos(xs, ys)
    n = len(xs)
    filas = []
    for k in range(n):
        num = 1.0
        den = 1.0
        for i in range(n):
            if i != k:
                num *= (x_eval - xs[i])
                den *= (xs[k] - xs[i])
        Lk = num / den
        filas.append({
            "k": k,
            "x_k": xs[k],
            "y_k": ys[k],
            "numerador ∏(x−x_i)": num,
            "denominador ∏(x_k−x_i)": den,
            "L_k(x)": Lk,
            "y_k·L_k(x)": ys[k] * Lk,
        })
    return filas


def coeficientes_base(xs, k):
    """
    Coeficientes de L_k(x) en forma estandar, de mayor a menor grado
    (convencion de np.polyval). Se construye multiplicando los binomios
    (x - x_i) con np.polymul y dividiendo por el denominador constante.
    """
    xs = np.asarray(xs, dtype=float)
    num = np.array([1.0])
    den = 1.0
    for i in range(len(xs)):
        if i != k:
            num = np.polymul(num, [1.0, -xs[i]])
            den *= (xs[k] - xs[i])
    return num / den, den


def coeficientes_polinomio(xs, ys):
    """
    Coeficientes [a_n, ..., a_1, a_0] de P_n(x) = sum y_k L_k(x).
    """
    xs, ys = validar_nodos(xs, ys)
    coef = np.zeros(len(xs))
    for k in range(len(xs)):
        ck, _ = coeficientes_base(xs, k)
        coef = coef + ys[k] * ck
    return coef


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


# --- Caso GAA: Latencia vs. memoria RAM de un microservicio (Sesion 8) ---
# GAA p.3: 4 nodos -> polinomio de grado 3; evaluar en x = 6 GB.
X_GAA_LATENCIA = [2, 4, 8, 12]
Y_GAA_LATENCIA = [150, 85, 50, 70]
XEVAL_GAA_LATENCIA = 6
INTERVALO_GAA = (2, 12)

# --- GP/GT: "Problema para la clase" (grado 2) -> P2(x) = x^2 - 6x + 8, f(1) = 3
X_GP_CLASE = [-1, 0, 3]
Y_GP_CLASE = [15, 8, -1]

# --- GP/GT: Ejercicio (grado 2) -> estimar f(0.8)
X_GP_EJ1 = [0, 0.5, 1.0]
Y_GP_EJ1 = [1.3, 2.5, 0.9]

# --- GP/GT: Ejercicio f(x) = (3+x)/(1+x), grado 2 -> f(0.25) y error absoluto
X_GP_EJ2 = [0.1, 0.2, 0.4]
Y_GP_EJ2 = [2.82, 2.67, 2.43]


if __name__ == "__main__":
    print("=== GAA: Latencia vs. memoria (Lagrange, n = 3) ===")
    for k in range(4):
        ck, den = coeficientes_base(X_GAA_LATENCIA, k)
        print(f"L{k}(x): denominador = {den:g}, coef = {np.round(ck, 6)}")
    coef = coeficientes_polinomio(X_GAA_LATENCIA, Y_GAA_LATENCIA)
    print("P3(x) =", polinomio_a_texto(coef))
    # esperado: -0.223958x^3 + 7.09375x^2 - 68.791667x + 261
    for fila in tabla_evaluacion(X_GAA_LATENCIA, Y_GAA_LATENCIA, XEVAL_GAA_LATENCIA):
        print(fila)
    print("P3(6) =", lagrange(X_GAA_LATENCIA, Y_GAA_LATENCIA, XEVAL_GAA_LATENCIA))
    # esperado: 55.25 ms
    print("Chequeo en los nodos:", lagrange(X_GAA_LATENCIA, Y_GAA_LATENCIA, np.array(X_GAA_LATENCIA)))
    # esperado: [150, 85, 50, 70]

    print("\n=== GP: problema para la clase ===")
    print("P2(x) =", polinomio_a_texto(coeficientes_polinomio(X_GP_CLASE, Y_GP_CLASE)))
    print("f(1) ≈", lagrange(X_GP_CLASE, Y_GP_CLASE, 1))           # esperado: 3

    print("\n=== GP: ejercicio f(0.8) ===")
    print("P2(x) =", polinomio_a_texto(coeficientes_polinomio(X_GP_EJ1, Y_GP_EJ1)))
    print("f(0.8) ≈", lagrange(X_GP_EJ1, Y_GP_EJ1, 0.8))            # esperado: 1.876

    print("\n=== GP: ejercicio f(x) = (3+x)/(1+x) ===")
    aprox = lagrange(X_GP_EJ2, Y_GP_EJ2, 0.25)
    exacto = (3 + 0.25) / (1 + 0.25)
    print("f(0.25) ≈", aprox, "| exacto =", exacto, "| error abs =", abs(aprox - exacto))
    # esperado: 2.6025 | 2.6 | 0.0025
