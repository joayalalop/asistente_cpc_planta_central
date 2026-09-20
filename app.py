from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from motor_cpc import CONFIG_MODULOS, MotorCPC, construir_excel_resultados


st.set_page_config(
    page_title="Validador masivo CPC - ENESEM",
    page_icon="🔎",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent


def _normalizar_nombre_columna(valor: object) -> str:
    return str(valor).strip().casefold().replace(" ", "_")


def sugerir_columna(columnas, candidatos):
    normalizadas = {_normalizar_nombre_columna(c): c for c in columnas}
    for candidato in candidatos:
        if candidato in normalizadas:
            return normalizadas[candidato]
    for columna in columnas:
        texto = _normalizar_nombre_columna(columna)
        if any(candidato in texto for candidato in candidatos):
            return columna
    return None


@st.cache_data(show_spinner=False)
def leer_archivo(contenido: bytes, nombre_archivo: str, hoja: str | None = None):
    extension = Path(nombre_archivo).suffix.lower()
    buffer = io.BytesIO(contenido)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(buffer, sheet_name=hoja or 0, dtype=str)
    if extension == ".csv":
        return pd.read_csv(buffer, sep=None, engine="python", dtype=str)
    if extension == ".parquet":
        return pd.read_parquet(buffer).astype("string")
    if extension == ".sav":
        import pyreadstat

        with tempfile.NamedTemporaryFile(suffix=".sav") as temporal:
            temporal.write(contenido)
            temporal.flush()
            df, _ = pyreadstat.read_sav(temporal.name, apply_value_formats=False)
        return df
    raise ValueError(f"Formato no admitido: {extension}")


def listar_hojas(contenido: bytes, nombre_archivo: str):
    if Path(nombre_archivo).suffix.lower() not in {".xlsx", ".xls"}:
        return []
    return pd.ExcelFile(io.BytesIO(contenido)).sheet_names


def indice_opcion(opciones, valor, predeterminado=0):
    if valor in opciones:
        return opciones.index(valor)
    return predeterminado


@st.cache_resource(show_spinner=False)
def cargar_motor(tipo_modulo: str):
    return MotorCPC(tipo_modulo=tipo_modulo, base_dir=BASE_DIR)


st.title("🔎 Validador masivo de códigos CPC")
st.caption("Encuesta Estructural Empresarial – ENESEM | Herramienta para Planta Central")

st.info(
    "La herramienta conserva todas las variables de la base y agrega las tres mejores "
    "alternativas CPC, sus niveles de similitud y un dictamen preliminar. El resultado "
    "debe utilizarse como apoyo técnico, no como sustituto del criterio del analista."
)

with st.sidebar:
    st.header("Configuración")
    tipo_modulo = st.selectbox(
        "Universo que se analizará",
        options=list(CONFIG_MODULOS),
        format_func=lambda x: CONFIG_MODULOS[x]["etiqueta"],
    )
    config = CONFIG_MODULOS[tipo_modulo]
    st.write(f"Longitud esperada del CPC: **{config['largo_codigo']} dígitos**")
    usar_ciiu = st.checkbox(
        "Priorizar coincidencias por CIIU",
        value=bool(config["usar_ciiu"]),
        disabled=not config["usar_ciiu"],
        help="Si no hay una coincidencia suficiente dentro del CIIU, el motor amplía la búsqueda.",
    )
    umbral_ciiu = st.slider(
        "Umbral para mantener el filtro CIIU",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
        disabled=not usar_ciiu,
    )
    tamano_lote = st.select_slider(
        "Tamaño del bloque de procesamiento",
        options=[16, 32, 64, 128],
        value=64,
        help="Use bloques menores si el servidor tiene poca memoria.",
    )

archivo = st.file_uploader(
    "Suba la base que desea validar",
    type=["xlsx", "xls", "csv", "parquet", "sav"],
    help="Las únicas variables obligatorias son la descripción y el CPC registrado.",
)

if archivo is None:
    st.stop()

contenido = archivo.getvalue()
hojas = listar_hojas(contenido, archivo.name)
hoja = st.selectbox("Hoja del archivo", hojas) if hojas else None

try:
    base = leer_archivo(contenido, archivo.name, hoja)
except Exception as exc:
    st.error(f"No se pudo leer el archivo: {exc}")
    st.stop()

if base.empty:
    st.warning("El archivo no contiene registros.")
    st.stop()

base.columns = [str(c).strip() for c in base.columns]
columnas = list(base.columns)
opciones_con_vacio = ["— No utilizar —"] + columnas

sugerida_nombre = sugerir_columna(
    columnas,
    ["nombre", "descripcion", "descripción", "glosa", "producto", "servicio"],
)
sugerida_cpc = sugerir_columna(columnas, ["cpc", "codigo_cpc", "código_cpc"])
sugerida_ciiu = sugerir_columna(columnas, ["ciiu", "ciiu4", "actividad_ciiu"])
sugerida_tipo_comercio = sugerir_columna(
    columnas, ["tipo_comercio", "mayor_menor", "tipo_venta"]
)

st.subheader("1. Correspondencia de columnas")
col1, col2, col3, col4 = st.columns(4)
with col1:
    columna_nombre = st.selectbox(
        "Descripción o nombre *",
        columnas,
        index=indice_opcion(columnas, sugerida_nombre),
    )
with col2:
    columna_cpc = st.selectbox(
        "CPC registrado *",
        columnas,
        index=indice_opcion(columnas, sugerida_cpc),
    )
with col3:
    columna_ciiu_sel = st.selectbox(
        "CIIU (opcional)",
        opciones_con_vacio,
        index=indice_opcion(opciones_con_vacio, sugerida_ciiu),
        disabled=not usar_ciiu,
    )
with col4:
    columna_tipo_sel = st.selectbox(
        "Tipo de comercio (opcional)",
        opciones_con_vacio,
        index=indice_opcion(opciones_con_vacio, sugerida_tipo_comercio),
        disabled=tipo_modulo != "comercio",
    )

columna_ciiu = None if columna_ciiu_sel == "— No utilizar —" else columna_ciiu_sel
columna_tipo = None if columna_tipo_sel == "— No utilizar —" else columna_tipo_sel

if columna_nombre == columna_cpc:
    st.error("La descripción y el CPC registrado deben corresponder a columnas diferentes.")
    st.stop()

st.caption(f"Registros detectados: {len(base):,} | Variables originales: {len(base.columns):,}")
st.dataframe(base.head(10), use_container_width=True, hide_index=True)

st.subheader("2. Ejecutar validación")
if st.button("Analizar códigos CPC", type="primary", use_container_width=True):
    try:
        motor = cargar_motor(tipo_modulo)
    except Exception as exc:
        st.error(f"No fue posible cargar el motor de {config['etiqueta'].lower()}: {exc}")
        st.stop()

    barra = st.progress(0, text="Preparando el análisis...")

    def actualizar_progreso(procesados, total):
        proporcion = 1.0 if total == 0 else min(procesados / total, 1.0)
        barra.progress(proporcion, text=f"Analizando {procesados:,} de {total:,} registros...")

    try:
        resultados = motor.analizar_base(
            base=base,
            columna_nombre=columna_nombre,
            columna_cpc=columna_cpc,
            columna_ciiu=columna_ciiu,
            columna_tipo_comercio=columna_tipo,
            usar_ciiu=usar_ciiu,
            umbral_ciiu=umbral_ciiu,
            tamano_lote=tamano_lote,
            callback_progreso=actualizar_progreso,
        )
    except Exception as exc:
        barra.empty()
        st.exception(exc)
        st.stop()

    barra.progress(1.0, text="Análisis completado")
    st.session_state["resultados_cpc"] = resultados
    st.session_state["archivo_resultados_cpc"] = construir_excel_resultados(resultados)

if "resultados_cpc" in st.session_state:
    resultados = st.session_state["resultados_cpc"]
    resumen = resultados["resumen"]
    enriquecida = resultados["base_enriquecida"]
    observaciones = resultados["observaciones"]

    st.subheader("3. Resultados")
    metricas = st.columns(5)
    metricas[0].metric("Registros", f"{resumen['total_registros']:,}")
    metricas[1].metric("Consistentes", f"{resumen['consistentes']:,}")
    metricas[2].metric("Revisión", f"{resumen['revision']:,}")
    metricas[3].metric("Posibles inconsistencias", f"{resumen['posibles_inconsistencias']:,}")
    metricas[4].metric("Sin evidencia", f"{resumen['sin_evidencia']:,}")

    filtro = st.multiselect(
        "Filtrar por dictamen",
        sorted(enriquecida["cpc_dictamen"].dropna().unique().tolist()),
    )
    vista = enriquecida
    if filtro:
        vista = vista[vista["cpc_dictamen"].isin(filtro)]

    st.dataframe(vista, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar resultados en Excel",
        data=st.session_state["archivo_resultados_cpc"],
        file_name=f"validacion_cpc_{tipo_modulo}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )

    if not observaciones.empty:
        st.caption(f"La matriz de observaciones contiene {len(observaciones):,} registros priorizados.")

