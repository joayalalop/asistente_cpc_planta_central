# Validador masivo CPC – ENESEM

Aplicación paralela al Motor de Codificación Asistida CPC. Recibe una base con
cualquier número de variables, conserva todas las columnas originales y analiza
la correspondencia entre una descripción (`nombre`) y el CPC registrado (`cpc`).

## Archivos del motor

Copie en esta carpeta los mismos archivos precalculados utilizados por el motor
individual:

- `cpc.xlsx`
- `df_productos.parquet`
- `vec_productos.npz`
- `df_materias.parquet`
- `vec_materias.npz`
- `df_servicios.parquet`
- `vec_servicios.npz`
- `df_comercio.parquet`
- `vec_comercio.npz`

La aplicación carga únicamente los archivos del universo seleccionado. Por
ejemplo, para validar productos solo necesita `cpc.xlsx`,
`df_productos.parquet` y `vec_productos.npz`.

## Formatos admitidos

- Excel (`.xlsx` y `.xls`)
- CSV
- Parquet
- SPSS (`.sav`)

Las únicas variables obligatorias son:

- descripción o nombre;
- CPC registrado.

El CIIU y el tipo de comercio son opcionales. Los nombres originales de las
columnas no tienen que ser exactamente `nombre` y `cpc`: la interfaz permite
seleccionarlas después de cargar la base.

## Ejecución local

Desde esta carpeta:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Salida

El Excel descargable contiene:

- `RESUMEN`: conteos por resultado general;
- `BASE_ENRIQUECIDA`: todas las variables originales más los resultados;
- `OBSERVACIONES`: casos que no fueron clasificados como consistentes.

Por cada fila se incorporan:

- CPC registrado estandarizado y descripción oficial;
- existencia del código en el catálogo;
- tres alternativas CPC y sus similitudes;
- glosa histórica más cercana;
- CIIU asociado e historial de descripciones;
- posición y similitud del CPC registrado;
- dictamen, prioridad y explicación.

## Criterio técnico

Las similitudes semánticas son medidas de cercanía textual, no probabilidades.
Por ello, la aplicación emite dictámenes preliminares como `CONSISTENTE`,
`REVISIÓN MANUAL` o `POSIBLE INCONSISTENCIA`. La decisión definitiva continúa
correspondiendo al analista.

