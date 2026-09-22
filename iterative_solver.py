"""
iterative_solver.py
--------------------
Nucleo numerico de los metodos iterativos de Jacobi y Gauss-Seidel para
Sistemas de Ecuaciones Lineales, generalizado a n incognitas.

Basado en la Guia Teorica (GT) de la Sesion 7 - Metodos Numericos, UPeU:

    A = D - L - U   (diagonal, triangular inferior y superior estrictas)

Jacobi despeja cada x_i usando SOLO los valores de la iteracion anterior
(iteracion "simultanea"); Gauss-Seidel reutiliza de inmediato los valores
ya actualizados de la misma iteracion (converge, en general, mas rapido).

Ambos requieren a_ii != 0 para todo i. La convergencia esta GARANTIZADA
si A es estrictamente diagonal dominante (EDD) por filas:

    |a_ii| > sum_{j!=i} |a_ij|      para cada fila i

(condicion SUFICIENTE, no necesaria: puede haber convergencia sin EDD).
Para Gauss-Seidel existe ademas el criterio de Sassenfeld (beta_i) como
alternativa mas fina cuando la matriz no es EDD.

El caso de la GAA (balanceo de carga en un cluster de 4 servidores) es
estrictamente diagonal dominante en las 4 filas, por lo que Jacobi
converge para cualquier vector inicial.
"""

import numpy as np


def es_diagonal_dominante(A):
    """
    Verifica si A (n x n) es estrictamente diagonal dominante (EDD) por
    filas: |a_ii| > suma(|a_ij|, j != i) para TODA fila i.

    Retorna
    -------
    dominante : bool
        True si se cumple la condicion en las n filas.
    detalle : list[dict]
        Un registro por fila con el diagonal, la suma de los demas
        elementos y si esa fila individualmente cumple la condicion
        (util para mostrar la demostracion paso a paso, tal como la
        pide el "Analisis Preliminar" de la GAA).
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    detalle = []
    dominante = True
    for i in range(n):
        diagonal = abs(A[i, i])
        suma_resto = float(np.sum(np.abs(A[i, :])) - diagonal)
        cumple = diagonal > suma_resto
        dominante = dominante and cumple
        detalle.append({
            "fila": i + 1,
            "|a_ii|": diagonal,
            "suma_|a_ij|_j!=i": suma_resto,
            "cumple EDD": "SI" if cumple else "NO",
        })
    return dominante, detalle


def jacobi(A, b, x0=None, tol=1e-4, max_iter=100, error_relativo=True):
    """
    Metodo de Jacobi (Pasos 1-7, GT Sesion 7).

    x_i^(k+1) = (b_i - sum_{j!=i} a_ij * x_j^(k)) / a_ii

    Todos los componentes de la iteracion (k+1) se calculan usando
    UNICAMENTE los valores de la iteracion (k) (desplazamientos
    simultaneos) -- a diferencia de Gauss-Seidel.

    Parametros
    ----------
    A, b : array-like (n x n), (n,)
    x0 : array-like (n,) o None
        Aproximacion inicial. Si es None se usa el vector cero, tal como
        indica la GAA: x^(0) = [0,0,0,0]^T.
    tol : float
        Tolerancia del criterio de parada (por defecto 1e-4, igual que
        pide la GAA).
    error_relativo : bool
        Si True, usa el error relativo E = max_i |x_i^(k+1)-x_i^(k)| / |x_i^(k+1)|
        (formula de la GT, sin el factor *100 porque aqui se compara
        directamente contra epsilon en fraccion, no en %). Si False, usa
        el error absoluto E_a = max_i |x_i^(k+1)-x_i^(k)| (para
        reproducir ejercicios de la GP que piden un numero fijo de
        iteraciones o una tolerancia en valor absoluto).

    Retorna
    -------
    x : np.ndarray
        Solucion aproximada final.
    historial : list[dict]
        Un registro por iteracion: {"k", "x", "error"}.
    convergio : bool
        True si el error cayo por debajo de `tol` antes de `max_iter`.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    historial = []
    convergio = False
    for k in range(1, max_iter + 1):
        x_nuevo = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_nuevo[i] = (b[i] - suma) / A[i, i]

        if error_relativo:
            with np.errstate(divide="ignore", invalid="ignore"):
                error = float(np.max(np.abs((x_nuevo - x) / x_nuevo)))
        else:
            error = float(np.max(np.abs(x_nuevo - x)))

        historial.append({"k": k, "x": x_nuevo.copy(), "error": error})
        x = x_nuevo

        if error < tol:
            convergio = True
            break

    return x, historial, convergio


def gauss_seidel(A, b, x0=None, tol=1e-4, max_iter=100, error_relativo=True):
    """
    Metodo de Gauss-Seidel (Pasos 1-9, GT Sesion 7).

    Identico a Jacobi salvo que cada x_i se actualiza usando de
    inmediato los valores YA recalculados de x_1..x_{i-1} en la misma
    iteracion (por eso se sobreescribe el mismo vector x, sin necesitar
    x_nuevo separado -- ver la tabla comparativa de la GT, p.33).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    historial = []
    convergio = False
    for k in range(1, max_iter + 1):
        x_anterior = x.copy()
        for i in range(n):
            suma_izq = sum(A[i, j] * x[j] for j in range(i))
            suma_der = sum(A[i, j] * x_anterior[j] for j in range(i + 1, n))
            x[i] = (b[i] - suma_izq - suma_der) / A[i, i]

        if error_relativo:
            with np.errstate(divide="ignore", invalid="ignore"):
                error = float(np.max(np.abs((x - x_anterior) / x)))
        else:
            error = float(np.max(np.abs(x - x_anterior)))

        historial.append({"k": k, "x": x.copy(), "error": error})

        if error < tol:
            convergio = True
            break

    return x, historial, convergio


def criterio_sassenfeld(A):
    """
    Criterio de Sassenfeld para Gauss-Seidel (GT p.52-56).

    beta_1 = (1/|a11|) * sum_{j=2}^{n} |a1j|
    beta_i = ( sum_{j<i} |aij|*beta_j + sum_{j>i} |aij| ) / |aii|

    Si max(beta_i) < 1, el metodo de Gauss-Seidel converge (condicion
    mas fina que la dominancia diagonal: puede converger aun cuando A
    no sea EDD).
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    beta = np.zeros(n)
    for i in range(n):
        suma = 0.0
        for j in range(n):
            if j == i:
                continue
            peso = beta[j] if j < i else 1.0
            suma += abs(A[i, j]) * peso
        beta[i] = suma / abs(A[i, i])
    return beta, bool(np.max(beta) < 1.0)


# --- Caso GAA: Balanceo de Carga en un Cluster de Servidores (Sesion 7) ---
# GAA p.2: 4 nodos, sistema estrictamente diagonal dominante en sus 4 filas.
A_GAA_CLUSTER = [
    [10, -2, -1, 0],
    [-1, 8, 0, -2],
    [-2, 0, 12, -3],
    [0, -1, -2, 9],
]
B_GAA_CLUSTER = [15, 18, 25, 20]
X0_GAA_CLUSTER = [0, 0, 0, 0]
TOL_GAA_CLUSTER = 1e-4

# --- Caso GP: ejercicio de Jacobi (3x3, 4 iteraciones fijas) --- GP p.2
A_GP_JACOBI = [[5, 1, 1], [3, 4, 1], [3, 3, 6]]
B_GP_JACOBI = [5, 6, 0]

# --- Caso GP: ejemplo de Gauss-Seidel (3x3) --- GP p.3
A_GP_GS = [[10, 2, 1], [1, 5, 1], [2, 3, 10]]
B_GP_GS = [7, -8, 6]
X0_GP_GS = [0.7, -1.6, 0.6]
TOL_GP_GS = 1e-2

# --- Ejemplo del criterio de Sassenfeld --- GT p.54
A_SASSENFELD_EJEMPLO = [[10, 2, -1], [-1, 8, 2], [2, -1, 9]]


if __name__ == "__main__":
    print("=== GAA: Balanceo de carga (Jacobi) ===")
    dominante, detalle = es_diagonal_dominante(A_GAA_CLUSTER)
    for fila in detalle:
        print(fila)
    print("¿A es EDD?", dominante)  # esperado: True en las 4 filas

    x, historial, convergio = jacobi(
        A_GAA_CLUSTER, B_GAA_CLUSTER, X0_GAA_CLUSTER, TOL_GAA_CLUSTER
    )
    for reg in historial:
        print(reg["k"], np.round(reg["x"], 6), reg["error"])
    print("Convergio:", convergio, "en", len(historial), "iteraciones")
    print("Solucion final:", np.round(x, 4))
    # esperado: converge en 10 iteraciones a x = [2.5136, 3.3995, 3.3376, 3.3416]

    print("\n=== GP: Jacobi, 4 iteraciones exactas ===")
    _, historial_gp, _ = jacobi(
        A_GP_JACOBI, B_GP_JACOBI, tol=0.0, max_iter=4, error_relativo=False
    )
    for reg in historial_gp:
        print(reg["k"], np.round(reg["x"], 6))
    # esperado en k=4: x = [0.8875, 0.85625, -1.19375]

    print("\n=== GP: Gauss-Seidel hasta error < 1e-2 ===")
    x_gs, historial_gs, convergio_gs = gauss_seidel(
        A_GP_GS, B_GP_GS, X0_GP_GS, TOL_GP_GS, error_relativo=False
    )
    for reg in historial_gs:
        print(reg["k"], np.round(reg["x"], 6), reg["error"])
    print("Convergio:", convergio_gs, "en", len(historial_gs), "iteraciones")
    # esperado: converge en 3 iteraciones a x = [1.00, -2.00, 1.00]

    print("\n=== Criterio de Sassenfeld (ejemplo GT p.54) ===")
    beta, converge = criterio_sassenfeld(A_SASSENFELD_EJEMPLO)
    print("beta =", np.round(beta, 4), "-> converge:", converge)
    # esperado: beta = [0.30, 0.2875, 0.0986], converge: True
