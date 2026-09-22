"""
lu_solver.py
------------
Nucleo numerico del solver de Sistemas de Ecuaciones Lineales por
Factorizacion LU (metodo de Doolittle), generalizado a n incognitas.

Incluye pivoteo parcial por filas para estabilidad numerica (necesario
en el caso general n x n, aunque los ejemplos de la GP/GAA de 3x3 no
lo requieran). Con pivoteo se obtiene:

    P @ A = L @ U

donde P es una matriz de permutacion. Para resolver A x = b:

    L y = P b   (sustitucion hacia adelante)
    U x = y     (sustitucion hacia atras)

La ventaja clave (la misma idea del ejercicio GAA de la sesion 6):
P, L y U se calculan UNA sola vez y se reutilizan para resolver
tantos vectores b como se necesiten, en O(n^2) cada uno, en vez de
repetir la eliminacion gaussiana completa O(n^3) cada vez.
"""

import numpy as np


def lu_decomposition(A, pivoting=True):
    """
    Descompone A (n x n) en P, L, U mediante Doolittle.

    Parametros
    ----------
    A : array-like (n x n)
    pivoting : bool
        Si True, aplica pivoteo parcial (recomendado para n grande
        o matrices mal condicionadas).

    Retorna
    -------
    P, L, U : np.ndarray
    steps : list[str]
        Bitacora de las operaciones de fila realizadas (para mostrar
        el procedimiento paso a paso en la interfaz).
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("La matriz A debe ser cuadrada (n x n).")

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    steps = []

    for i in range(n):
        if pivoting:
            pivote_fila = int(np.argmax(np.abs(U[i:, i]))) + i
            if pivote_fila != i:
                U[[i, pivote_fila], :] = U[[pivote_fila, i], :]
                P[[i, pivote_fila], :] = P[[pivote_fila, i], :]
                if i > 0:
                    L[[i, pivote_fila], :i] = L[[pivote_fila, i], :i]
                steps.append(f"Pivoteo: F{i + 1} <-> F{pivote_fila + 1}")

        if abs(U[i, i]) < 1e-12:
            raise ValueError(
                f"Pivote nulo en la posicion ({i + 1},{i + 1}); "
                f"la matriz es singular o requiere pivoteo."
            )

        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
            steps.append(f"F{j + 1} = F{j + 1} - ({factor:.4f}) * F{i + 1}")

    return P, L, U, steps


def forward_substitution(L, b):
    """Resuelve L y = b (L triangular inferior, diagonal = 1)."""
    b = np.array(b, dtype=float)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y


def back_substitution(U, y):
    """Resuelve U x = y (U triangular superior)."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def resolver(P, L, U, b):
    """
    Resuelve A x = b reutilizando P, L, U ya calculadas
    (no se recalcula la descomposicion).
    """
    b = np.array(b, dtype=float)
    Pb = P @ b
    y = forward_substitution(L, Pb)
    x = back_substitution(U, y)
    return x, y


if __name__ == "__main__":
    # Autoprueba rapida con el ejercicio de la GAA (Sesion 6)
    A = [[4, 2, 1], [12, 10, 5], [-8, 8, 7]]
    b1 = [14, 46, 26]
    b2 = [20, 62, 30]

    P, L, U, steps = lu_decomposition(A)
    print("P=\n", P)
    print("L=\n", L)
    print("U=\n", U)

    x1, y1 = resolver(P, L, U, b1)
    print("x (b1) =", np.round(x1, 4))  # esperado: [3, -6, 14]

    x2, y2 = resolver(P, L, U, b2)
    print("x (b2) =", np.round(x2, 4))  # esperado: [4.75, -10.1667, 21.3333]
