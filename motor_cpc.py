from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CONFIG_MODULOS = {
    "productos": {
        "etiqueta": "Productos elaborados",
        "prefijo": "productos",
        "largo_codigo": 13,
        "usar_ciiu": True,
        "es_comercio": False,
    },
    "materias": {
        "etiqueta": "Materias primas",
        "prefijo": "materias",
        "largo_codigo": 13,
        "usar_ciiu": False,
        "es_comercio": False,
    },
    "servicios": {
        "etiqueta": "Servicios",
        "prefijo": "servicios",
        "largo_codigo": 8,
        "usar_ciiu": True,
        "es_comercio": False,
    },
    "comercio": {
        "etiqueta": "Comercio",
        "prefijo": "comercio",
        "largo_codigo": 8,
        "usar_ciiu": True,
        "es_comercio": True,
    },
}


PALABRAS_VACIAS = re.compile(
    r"\b(Y|E|DE|DEL|LA|EL|LOS|LAS|EN|CON|PARA|POR|AL|UN|UNA)\b",
    flags=re.IGNORECASE,
)


def normalizar_glosa(texto) -> str:
    if texto is None or pd.isna(texto):
        return ""
    valor = unicodedata.normalize("NFKC", str(texto)).upper().strip()
    valor = re.sub(r"[,\.\-/()#;:_]+", " ", valor)
    valor = PALABRAS_VACIAS.sub(" ", valor)
    return re.sub(r"\s+", " ", valor).strip()


def limpiar_texto_visual(texto) -> str:
    if texto is None or pd.isna(texto):
        return ""
    valor = re.sub(r"[\r\n\t]+", " ", str(texto))
    valor = re.sub(r"\s+", " ", valor)
    return re.sub(r"\s*\|\|\s*", " || ", valor).strip()


def limpiar_codigo(valor, largo: int) -> tuple[str, str]:
    """Devuelve el código estandarizado y una alerta de estructura."""
    if valor is None or pd.isna(valor) or not str(valor).strip():
        return "", "CPC vacío"

    texto = str(valor).strip().replace(" ", "")
    if re.fullmatch(r"\d+\.0+", texto):
        texto = texto.split(".")[0]
    elif "e" in texto.lower():
        try:
            texto = format(Decimal(texto), "f").split(".")[0]
        except InvalidOperation:
            pass

    digitos = re.sub(r"\D", "", texto)
    if not digitos:
        return "", "CPC sin dígitos"
    if len(digitos) > largo:
        return digitos, f"CPC con más de {largo} dígitos"
    return digitos.zfill(largo), ""


def limpiar_ciiu(valor) -> str:
    if valor is None or pd.isna(valor):
        return ""
    digitos = re.sub(r"\D", "", str(valor).split(".")[0])
    return digitos[:4].zfill(4) if digitos else ""


def normalizar_tipo_comercio(valor) -> str:
    texto = normalizar_glosa(valor)
    if "MENOR" in texto or "MINOR" in texto:
        return "POR MENOR"
    if "MAYOR" in texto or "MAYORISTA" in texto:
        return "POR MAYOR"
    return texto


def _normalizar_filas(matriz: np.ndarray) -> np.ndarray:
    matriz = np.asarray(matriz, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas


@dataclass
class Candidato:
    codigo: str
    similitud: float
    glosa_match: str
    ciiu_asociado: str
    historial: str
    tipo_match: str


class MotorCPC:
    def __init__(self, tipo_modulo: str, base_dir: str | Path):
        if tipo_modulo not in CONFIG_MODULOS:
            raise ValueError(f"Módulo desconocido: {tipo_modulo}")

        self.tipo_modulo = tipo_modulo
        self.config = CONFIG_MODULOS[tipo_modulo]
        self.base_dir = Path(base_dir)
        self.largo = int(self.config["largo_codigo"])
        self.es_comercio = bool(self.config["es_comercio"])

        prefijo = self.config["prefijo"]
        ruta_df = self.base_dir / f"df_{prefijo}.parquet"
        ruta_vec = self.base_dir / f"vec_{prefijo}.npz"
        ruta_cpc = self.base_dir / "cpc.xlsx"

        faltantes = [str(r.name) for r in (ruta_df, ruta_vec, ruta_cpc) if not r.exists()]
        if faltantes:
            raise FileNotFoundError("Faltan archivos requeridos: " + ", ".join(faltantes))

        self.df_ref = pd.read_parquet(ruta_df, engine="pyarrow").fillna("").reset_index(drop=True)
        self.vectores = _normalizar_filas(np.load(ruta_vec)["vectores"])
        if len(self.df_ref) != len(self.vectores):
            raise ValueError("El Parquet y el archivo NPZ no contienen el mismo número de registros.")

        requeridas = {
            "CODIGO_CPC",
            "GLOSA_NORMALIZADA",
            "GLOSA_INDIVIDUAL",
            "CIIU_ASOCIADO",
            "EJEMPLOS_REALES_LIMPIOS",
        }
        if self.es_comercio:
            requeridas.add("TIPO_COMERCIO")
        faltan_columnas = requeridas.difference(self.df_ref.columns)
        if faltan_columnas:
            raise ValueError("El Parquet no contiene: " + ", ".join(sorted(faltan_columnas)))

        self.df_ref["CODIGO_CPC_LIMPIO"] = self.df_ref["CODIGO_CPC"].map(
            lambda x: limpiar_codigo(x, self.largo)[0]
        )
        self.df_ref["CIIU_LIMPIO"] = self.df_ref["CIIU_ASOCIADO"].map(limpiar_ciiu)
        if self.es_comercio:
            self.df_ref["TIPO_COMERCIO_LIMPIO"] = self.df_ref["TIPO_COMERCIO"].map(
                normalizar_tipo_comercio
            )

        self.codigos = self.df_ref["CODIGO_CPC_LIMPIO"].to_numpy(dtype=str)
        self.codigos_unicos, self.id_codigo = np.unique(self.codigos, return_inverse=True)
        self.indices_por_codigo = {
            indice: np.where(self.id_codigo == indice)[0]
            for indice in range(len(self.codigos_unicos))
        }
        self.exactos = self._construir_indice_exacto()
        self.mapa_oficial = self._cargar_catalogo(ruta_cpc)
        self.longitudes_catalogo = sorted({len(x) for x in self.mapa_oficial}, reverse=True)
        self.modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def _construir_indice_exacto(self):
        indice = defaultdict(Counter)
        for fila, row in self.df_ref.iterrows():
            glosa = normalizar_glosa(row["GLOSA_NORMALIZADA"] or row["GLOSA_INDIVIDUAL"])
            if not glosa:
                continue
            clave_tipo = (
                normalizar_tipo_comercio(row.get("TIPO_COMERCIO_LIMPIO", ""))
                if self.es_comercio
                else ""
            )
            indice[(glosa, clave_tipo)][self.codigos[fila]] += 1
        return indice

    def _cargar_catalogo(self, ruta: Path):
        cabecera = pd.read_excel(ruta, nrows=0)
        columnas = list(cabecera.columns)
        candidatas_codigo = [c for c in columnas if "CODIGO" in str(c).upper()]
        candidatas_desc = [
            c for c in columnas if "DESCRIP" in str(c).upper() or "CPC" in str(c).upper()
        ]
        if not candidatas_codigo or not candidatas_desc:
            raise ValueError("No se identificaron las columnas de código y descripción en cpc.xlsx.")

        col_codigo = candidatas_codigo[0]
        col_descripcion = candidatas_desc[-1]
        catalogo = pd.read_excel(ruta, dtype={col_codigo: str})
        mapa = {}
        for _, row in catalogo.iterrows():
            codigo = re.sub(r"\D", "", str(row[col_codigo]).replace(".0", ""))
            descripcion = limpiar_texto_visual(row[col_descripcion])
            if codigo and descripcion and descripcion.casefold() != "nan":
                mapa[codigo] = descripcion
                if len(codigo) < self.largo:
                    mapa[codigo.zfill(self.largo)] = descripcion
        return mapa

    def descripcion_oficial(self, codigo: str) -> str:
        if not codigo:
            return ""
        if codigo in self.mapa_oficial:
            return self.mapa_oficial[codigo]
        for largo in self.longitudes_catalogo:
            if largo < len(codigo) and codigo[:largo] in self.mapa_oficial:
                return self.mapa_oficial[codigo[:largo]] + " (nivel agrupado)"
        return "No localizada en el catálogo cargado"

    def existe_catalogo(self, codigo: str) -> bool:
        return bool(codigo and codigo in self.mapa_oficial)

    def _indices_alcance(self, ciiu: str, tipo_comercio: str, usar_ciiu: bool):
        mascara = np.ones(len(self.df_ref), dtype=bool)

        if self.es_comercio and tipo_comercio:
            mascara &= self.df_ref["TIPO_COMERCIO_LIMPIO"].to_numpy(dtype=str) == tipo_comercio

        indices_base = np.where(mascara)[0]
        if usar_ciiu and ciiu:
            mascara_ciiu = mascara & (
                self.df_ref["CIIU_LIMPIO"].to_numpy(dtype=str) == ciiu
            )
            indices_ciiu = np.where(mascara_ciiu)[0]
        else:
            indices_ciiu = np.array([], dtype=int)
        return indices_base, indices_ciiu

    def _top_distintos(self, scores: np.ndarray, indices: np.ndarray, top_k: int = 3):
        maximos = np.full(len(self.codigos_unicos), -np.inf, dtype=np.float32)
        if len(indices):
            np.maximum.at(maximos, self.id_codigo[indices], scores[indices])
        cantidad = min(top_k, int(np.isfinite(maximos).sum()))
        if cantidad == 0:
            return []
        ids_top = np.argpartition(maximos, -cantidad)[-cantidad:]
        ids_top = ids_top[np.argsort(maximos[ids_top])[::-1]]

        salida = []
        conjunto_indices = set(indices.tolist())
        for id_codigo in ids_top:
            disponibles = [
                i for i in self.indices_por_codigo[int(id_codigo)] if i in conjunto_indices
            ]
            if not disponibles:
                continue
            mejor_indice = disponibles[int(np.argmax(scores[disponibles]))]
            row = self.df_ref.iloc[mejor_indice]
            salida.append(
                Candidato(
                    codigo=str(self.codigos_unicos[id_codigo]),
                    similitud=float(maximos[id_codigo]),
                    glosa_match=limpiar_texto_visual(row["GLOSA_INDIVIDUAL"]),
                    ciiu_asociado=limpiar_ciiu(row["CIIU_ASOCIADO"]),
                    historial=limpiar_texto_visual(row["EJEMPLOS_REALES_LIMPIOS"]),
                    tipo_match="SEMÁNTICO",
                )
            )
        return salida

    def _insertar_exactos(self, candidatos, glosa_norm, tipo_comercio):
        clave = (glosa_norm, tipo_comercio if self.es_comercio else "")
        conteo = self.exactos.get(clave)
        if not conteo:
            return candidatos, set()

        codigos_exactos = {codigo for codigo, _ in conteo.most_common()}
        nuevos = []
        for codigo, _ in conteo.most_common(3):
            filas = self.indices_por_codigo[np.where(self.codigos_unicos == codigo)[0][0]]
            row = self.df_ref.iloc[filas[0]]
            nuevos.append(
                Candidato(
                    codigo=codigo,
                    similitud=1.0,
                    glosa_match=limpiar_texto_visual(row["GLOSA_INDIVIDUAL"]),
                    ciiu_asociado=limpiar_ciiu(row["CIIU_ASOCIADO"]),
                    historial=limpiar_texto_visual(row["EJEMPLOS_REALES_LIMPIOS"]),
                    tipo_match="EXACTO HISTÓRICO",
                )
            )
        nuevos.extend(c for c in candidatos if c.codigo not in codigos_exactos)
        return nuevos[:3], codigos_exactos

    def _puntaje_codigo(self, scores, codigo, indices, codigos_exactos):
        if not codigo:
            return np.nan
        if codigo in codigos_exactos:
            return 1.0
        posiciones_codigo = np.where(self.codigos == codigo)[0]
        if not len(posiciones_codigo):
            return np.nan
        posiciones = np.intersect1d(posiciones_codigo, indices, assume_unique=False)
        return float(np.max(scores[posiciones])) if len(posiciones) else np.nan

    def _dictaminar(self, codigo_registrado, alerta, candidatos, puntaje_registrado, exactos):
        if alerta:
            return "REVISAR ESTRUCTURA DEL CPC", "ALTA", alerta
        if not candidatos:
            return "SIN EVIDENCIA", "MEDIA", "No se generaron alternativas CPC."

        codigos_top = [c.codigo for c in candidatos]
        mejor = candidatos[0]

        if codigo_registrado in exactos:
            return (
                "CONSISTENTE - MATCH EXACTO",
                "BAJA",
                "El CPC registrado coincide con el historial exacto de la descripción.",
            )
        if codigo_registrado == mejor.codigo and mejor.similitud >= 0.85:
            return (
                "CONSISTENTE",
                "BAJA",
                "El CPC registrado coincide con la primera alternativa de alta similitud.",
            )
        if codigo_registrado in codigos_top:
            posicion = codigos_top.index(codigo_registrado) + 1
            return (
                "REVISAR - CPC ENTRE LAS 3 ALTERNATIVAS",
                "MEDIA",
                f"El CPC registrado aparece en la posición {posicion} del ranking.",
            )

        diferencia = (
            mejor.similitud - puntaje_registrado
            if not pd.isna(puntaje_registrado)
            else np.nan
        )
        if mejor.similitud >= 0.85 and (pd.isna(diferencia) or diferencia >= 0.10):
            return (
                "POSIBLE INCONSISTENCIA",
                "ALTA",
                "El CPC registrado no aparece entre las alternativas y existe una sugerencia fuerte.",
            )
        if mejor.similitud >= 0.50:
            return (
                "REVISIÓN MANUAL",
                "MEDIA",
                "El CPC registrado no aparece entre las tres alternativas, pero la evidencia no es concluyente.",
            )
        return (
            "SIN EVIDENCIA SUFICIENTE",
            "BAJA",
            "Ninguna alternativa alcanza el umbral mínimo de similitud.",
        )

    def analizar_base(
        self,
        base: pd.DataFrame,
        columna_nombre: str,
        columna_cpc: str,
        columna_ciiu: str | None = None,
        columna_tipo_comercio: str | None = None,
        usar_ciiu: bool = True,
        umbral_ciiu: float = 0.50,
        tamano_lote: int = 64,
        callback_progreso: Callable[[int, int], None] | None = None,
    ):
        salida = base.copy().reset_index(drop=True)
        salida.insert(0, "cpc_fila_origen", np.arange(2, len(salida) + 2))
        salida["cpc_nombre_normalizado"] = salida[columna_nombre].map(normalizar_glosa)

        codigos_limpios = salida[columna_cpc].map(lambda x: limpiar_codigo(x, self.largo))
        salida["cpc_registrado_limpio"] = codigos_limpios.map(lambda x: x[0])
        salida["cpc_alerta_estructura"] = codigos_limpios.map(lambda x: x[1])
        salida["cpc_existe_catalogo"] = salida["cpc_registrado_limpio"].map(
            self.existe_catalogo
        )
        salida["cpc_descripcion_registrada"] = salida["cpc_registrado_limpio"].map(
            self.descripcion_oficial
        )

        glosas_unicas = salida["cpc_nombre_normalizado"].drop_duplicates().tolist()
        vectores_por_glosa = {}
        glosas_validas = [g for g in glosas_unicas if g]
        if glosas_validas:
            embeddings = self.modelo.encode(
                glosas_validas,
                batch_size=tamano_lote,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            vectores_por_glosa = dict(zip(glosas_validas, embeddings))

        resultados = []
        total = len(salida)
        for inicio in range(0, total, tamano_lote):
            fin = min(inicio + tamano_lote, total)
            bloque = salida.iloc[inicio:fin]

            for _, row in bloque.iterrows():
                glosa = row["cpc_nombre_normalizado"]
                codigo_registrado = row["cpc_registrado_limpio"]
                alerta = row["cpc_alerta_estructura"]
                ciiu = limpiar_ciiu(row[columna_ciiu]) if columna_ciiu else ""
                tipo_comercio = (
                    normalizar_tipo_comercio(row[columna_tipo_comercio])
                    if columna_tipo_comercio
                    else ""
                )

                if not glosa:
                    resultados.append(
                        self._fila_resultado_vacia("Descripción vacía", codigo_registrado)
                    )
                    continue

                vector = np.asarray(vectores_por_glosa[glosa], dtype=np.float32)
                scores = self.vectores @ vector
                indices_base, indices_ciiu = self._indices_alcance(
                    ciiu=ciiu,
                    tipo_comercio=tipo_comercio,
                    usar_ciiu=usar_ciiu and self.config["usar_ciiu"],
                )

                uso_filtro_ciiu = False
                indices = indices_base
                if len(indices_ciiu) and float(np.max(scores[indices_ciiu])) >= umbral_ciiu:
                    indices = indices_ciiu
                    uso_filtro_ciiu = True

                candidatos = self._top_distintos(scores, indices, top_k=3)
                candidatos, exactos = self._insertar_exactos(
                    candidatos, glosa, tipo_comercio
                )
                puntaje_registrado = self._puntaje_codigo(
                    scores, codigo_registrado, indices, exactos
                )
                dictamen, prioridad, detalle = self._dictaminar(
                    codigo_registrado,
                    alerta,
                    candidatos,
                    puntaje_registrado,
                    exactos,
                )
                resultados.append(
                    self._armar_fila_resultado(
                        candidatos=candidatos,
                        codigo_registrado=codigo_registrado,
                        puntaje_registrado=puntaje_registrado,
                        filtro_ciiu=uso_filtro_ciiu,
                        dictamen=dictamen,
                        prioridad=prioridad,
                        detalle=detalle,
                    )
                )

            if callback_progreso:
                callback_progreso(fin, total)

        calculados = pd.DataFrame(resultados)
        enriquecida = pd.concat([salida, calculados], axis=1)
        observaciones = enriquecida[
            ~enriquecida["cpc_dictamen"].str.startswith("CONSISTENTE", na=False)
        ].copy()
        resumen = self._resumir(enriquecida)
        return {
            "base_enriquecida": enriquecida,
            "observaciones": observaciones,
            "resumen": resumen,
        }

    def _fila_resultado_vacia(self, detalle, codigo_registrado):
        return self._armar_fila_resultado(
            candidatos=[],
            codigo_registrado=codigo_registrado,
            puntaje_registrado=np.nan,
            filtro_ciiu=False,
            dictamen="REVISIÓN MANUAL",
            prioridad="ALTA",
            detalle=detalle,
        )

    def _armar_fila_resultado(
        self,
        candidatos,
        codigo_registrado,
        puntaje_registrado,
        filtro_ciiu,
        dictamen,
        prioridad,
        detalle,
    ):
        fila = {
            "cpc_confianza_registrado": (
                round(float(puntaje_registrado) * 100, 2)
                if not pd.isna(puntaje_registrado)
                else np.nan
            ),
            "cpc_posicion_registrado": (
                next(
                    (i + 1 for i, c in enumerate(candidatos) if c.codigo == codigo_registrado),
                    np.nan,
                )
            ),
            "cpc_filtro_ciiu_aplicado": filtro_ciiu,
            "cpc_dictamen": dictamen,
            "cpc_prioridad": prioridad,
            "cpc_detalle": detalle,
        }
        for posicion in range(3):
            prefijo = f"cpc_sugerido_{posicion + 1}"
            if posicion < len(candidatos):
                candidato = candidatos[posicion]
                fila[prefijo] = candidato.codigo
                fila[f"cpc_confianza_{posicion + 1}"] = round(
                    candidato.similitud * 100, 2
                )
                fila[f"cpc_descripcion_{posicion + 1}"] = self.descripcion_oficial(
                    candidato.codigo
                )
                fila[f"cpc_glosa_match_{posicion + 1}"] = candidato.glosa_match
                fila[f"cpc_ciiu_asociado_{posicion + 1}"] = candidato.ciiu_asociado
                fila[f"cpc_tipo_match_{posicion + 1}"] = candidato.tipo_match
                fila[f"cpc_historial_{posicion + 1}"] = candidato.historial
            else:
                fila[prefijo] = ""
                fila[f"cpc_confianza_{posicion + 1}"] = np.nan
                fila[f"cpc_descripcion_{posicion + 1}"] = ""
                fila[f"cpc_glosa_match_{posicion + 1}"] = ""
                fila[f"cpc_ciiu_asociado_{posicion + 1}"] = ""
                fila[f"cpc_tipo_match_{posicion + 1}"] = ""
                fila[f"cpc_historial_{posicion + 1}"] = ""
        return fila

    @staticmethod
    def _resumir(df):
        dictamen = df["cpc_dictamen"].fillna("")
        return {
            "total_registros": len(df),
            "consistentes": int(dictamen.str.startswith("CONSISTENTE").sum()),
            "revision": int(dictamen.str.startswith("REVIS").sum()),
            "posibles_inconsistencias": int(
                dictamen.str.startswith("POSIBLE INCONSISTENCIA").sum()
            ),
            "sin_evidencia": int(dictamen.str.startswith("SIN EVIDENCIA").sum()),
        }


def construir_excel_resultados(resultados: dict) -> bytes:
    buffer = io.BytesIO()
    resumen = pd.DataFrame(
        {
            "Indicador": [
                "Total de registros",
                "Consistentes",
                "Revisión",
                "Posibles inconsistencias",
                "Sin evidencia suficiente",
            ],
            "Valor": [
                resultados["resumen"]["total_registros"],
                resultados["resumen"]["consistentes"],
                resultados["resumen"]["revision"],
                resultados["resumen"]["posibles_inconsistencias"],
                resultados["resumen"]["sin_evidencia"],
            ],
        }
    )
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, sheet_name="RESUMEN", index=False)
        resultados["base_enriquecida"].to_excel(
            writer, sheet_name="BASE_ENRIQUECIDA", index=False
        )
        resultados["observaciones"].to_excel(
            writer, sheet_name="OBSERVACIONES", index=False
        )

        for nombre_hoja, frame in {
            "RESUMEN": resumen,
            "BASE_ENRIQUECIDA": resultados["base_enriquecida"],
            "OBSERVACIONES": resultados["observaciones"],
        }.items():
            hoja = writer.sheets[nombre_hoja]
            hoja.freeze_panes(1, 0)
            hoja.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            hoja.set_column(0, max(len(frame.columns) - 1, 0), 18)
    return buffer.getvalue()

