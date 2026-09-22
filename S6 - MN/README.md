# Solver LU (Doolittle) — n variables

Software para resolver sistemas de ecuaciones lineales $A\mathbf{x}=\mathbf{b}$
mediante **factorización LU** (método de Doolittle, con pivoteo parcial),
generalizado a **n incógnitas**. Hecho para la Sesión 6 de Métodos Numéricos
(GT / GP / GAA).

## Contenido

- `lu_solver.py` — lógica numérica pura (sin interfaz): descomposición `P·A = L·U`,
  sustitución hacia adelante, sustitución hacia atrás, y `resolver()` para
  reutilizar `L` y `U` con múltiples vectores `b` sin recalcular la descomposición.
- `app.py` — interfaz visual en **Streamlit**: permite ingresar `A` y uno o
  varios vectores `b`, ver `P`, `L`, `U` como tablas coloreadas, la bitácora
  de la triangulación, y la solución de cada sistema.
- `requirements.txt` — dependencias (`streamlit`, `numpy`, `pandas`).

## Cómo ejecutarlo

Desde esta carpeta (`S6 - MN`), en una terminal (PowerShell o cmd):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Esto abre la app en el navegador (normalmente `http://localhost:8501`).

## Uso rápido

1. En la barra lateral, define `n` (número de incógnitas) y, si quieres,
   carga uno de los ejemplos de la Sesión 6 (GAA o GP).
2. Edita la matriz `A` y el/los vector(es) `b` directamente en las tablas.
3. Agrega más vectores `b` con el botón **"+ Agregar vector b"** — así se
   demuestra la ventaja de LU: se descompone una sola vez y se reutiliza
   para cada `b` nuevo (idea central de la Fase 2 de la GAA: "balanceo de
   carga" con tráfico variable).
4. Pulsa **"Descomponer (LU) y resolver todo"** para ver `P`, `L`, `U`,
   los pasos de la triangulación y la solución de cada sistema, con su
   residuo `‖A x − b‖` como verificación.

## Probar solo la lógica (sin interfaz)

```bash
python lu_solver.py
```

Corre una autoprueba con el ejercicio de la GAA y debe imprimir
`x (b1) = [3, -6, 14]` y `x (b2) = [4.75, -10.1667, 21.3333]`.

## Notas de diseño

- Se usa **pivoteo parcial** (`P·A = L·U`) porque, a diferencia de los
  ejemplos 3×3 de la guía (donde no hace falta), un solver genérico para
  *n* variables sí puede toparse con un pivote nulo o muy pequeño.
- La visualización de matrices usa `pandas.DataFrame.style` con
  `background_gradient` para resaltar magnitudes, similar a lo que se
  hace manualmente coloreando celdas en Excel.
- Si más adelante se quiere una versión sin instalar nada (solo para
  mostrar resultados), se puede exportar la misma lógica a una página
  HTML/JS como alternativa — pero para trabajo de curso con Python,
  Streamlit es la ruta más simple y rápida de desplegar.
