# MN-U2 — Métodos Numéricos, Unidad 2

Repositorio con el trabajo de la **Unidad 2** del curso de Métodos Numéricos
(UPeU). Cada sesión vive en su propia carpeta, con su propio `app.py`
(Streamlit) y `requirements.txt`, para poder desplegarse de forma
independiente en Streamlit Community Cloud sin afectar a las demás.

## Estructura

```
MN-U2/
├── .gitignore
├── README.md                 <- este archivo
├── S6 - MN/                  <- Sesión 6: Sistemas de Ecuaciones Lineales (LU / Doolittle)
│   ├── app.py                <- interfaz Streamlit
│   ├── lu_solver.py           <- lógica numérica (LU con pivoteo, sustituciones)
│   ├── requirements.txt
│   └── README.md
└── S7 - MN/                  <- (futuras sesiones se agregan igual)
    └── ...
```

## Cómo correr una sesión localmente

```bash
cd "S6 - MN"
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Cómo desplegar una sesión en Streamlit Community Cloud

1. Sube el repo a GitHub (ver comandos más abajo).
2. En [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Selecciona el repo `JeanpierreSam/MN-U2`, la rama `main`.
4. En **Main file path** escribe la ruta a la sesión que quieras desplegar,
   por ejemplo: `S6 - MN/app.py`.
5. Streamlit Cloud busca automáticamente el `requirements.txt` en esa misma
   carpeta (no hace falta que esté en la raíz del repo).
6. Cada sesión (`S6 - MN`, `S7 - MN`, ...) se despliega como una **app
   separada**, con su propia URL, aunque compartan el mismo repositorio.

## Cómo agregar una nueva sesión

1. Crea una carpeta nueva, por ejemplo `S7 - MN/`.
2. Copia dentro `app.py`, el/los módulo(s) de lógica, y su propio
   `requirements.txt`.
3. `git add`, `git commit`, `git push`.
4. Crea una nueva app en Streamlit Cloud apuntando a `S7 - MN/app.py`.
