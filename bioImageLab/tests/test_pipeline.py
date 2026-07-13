#!/usr/bin/env python3
# === tests/test_pipeline.py ===
"""
Tests de integración del pipeline BioImageLab.

Ejecutar desde la raíz del proyecto:
    pytest tests/test_pipeline.py -v --tb=short

O directamente:
    python tests/test_pipeline.py

Requisitos:
  - Estructura de paquete completa (nucleo/)
  - Imagen de prueba .ids/.ics en TEST_DATA_DIR
  - pip install pytest numpy pandas matplotlib scipy scikit-image

Cobertura:
  1. Importaciones — verifica que todos los módulos cargan sin error
  2. Resultado_Either — Ok/Err/log/bind funcionan correctamente
  3. Validar_Flujo_Trabajo — siempre Ok, log detallado, nunca bloquea
  4. Grafo DAG — orden topológico, detección de ciclos
  5. FlujoTrabajo lineal — ejecución con datos sintéticos
  6. FlujoTrabajo con merge — bifurcación + fusión
  7. ConstructorFlujoTrabajo — construcción desde config dict
  8. YAML parsing — test.yaml correcto se parsea sin error
  9. GestorLab — registro, validación, ejecución con imagen real (si existe)
  10. Modo Batch — ejecución sobre lista de rutas
  11. Controladores — normalizador y segmentador sobre arrays sintéticos
  12. Cuantificador — retorna DataFrame no vacío
  13. Exportadores — CSV y TSV se escriben a disco
  14. Plots — figuras se generan sin excepciones
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# ── Ajustar path para importaciones relativas ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Rutas de prueba ────────────────────────────────────────────────────────
TEST_DATA_DIR = ROOT / "bioImageLab" / "test_data"
RUTA_IDS      = TEST_DATA_DIR / "glp1_1.ids"
RUTA_ICS      = TEST_DATA_DIR / "glp1_1.ics"
RUTA_YAML     = ROOT / "test.yaml"

_IMAGEN_REAL  = RUTA_IDS if RUTA_IDS.exists() else (RUTA_ICS if RUTA_ICS.exists() else None)


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def bio_imagen_sintetica():
    """BioImagenData sintética: T=1, Z=3, C=1, Y=64, X=64, uint16."""
    from nucleo.controlador.Controlador_BioImagen import BioImagenData, Dimensiones
    datos = (np.random.randint(100, 3000, (1, 3, 1, 64, 64), dtype=np.uint16))
    dims  = Dimensiones(T=1, Z=3, C=1, Y=64, X=64)
    return BioImagenData(
        datos=datos, dims=dims,
        canales=("canal_0",), ruta_origen=Path("sintetico.ids"),
    )


@pytest.fixture(scope="session")
def grafo_lineal_simple():
    """Grafo lineal: input → max_norm → resultado."""
    from nucleo.gestorLab.Flujo_Trabajo import (
        GrafoPipeline, NodoPipeline, AristaOperacion
    )
    from nucleo.gestorLab.Operacion import Operacion
    from nucleo.gestorLab.Categoria_Operacion import CategoriaOperacion, TipoDato
    from nucleo.controlador.Resultado_Either import Ok

    grafo = GrafoPipeline()
    grafo.agregar_nodo(NodoPipeline(id="input",   tipo_dato=TipoDato.IMAGEN))
    grafo.agregar_nodo(NodoPipeline(id="salida",  tipo_dato=TipoDato.IMAGEN))
    op = Operacion(
        nombre="identidad",
        categoria=CategoriaOperacion.PREPROCESAMIENTO,
        instancia_callable=lambda x: Ok(x),
        tipo_dato_salida=TipoDato.IMAGEN,
    )
    grafo.agregar_arista(AristaOperacion("input", "salida", op))
    return grafo


@pytest.fixture(scope="session")
def flujo_lineal(grafo_lineal_simple):
    from nucleo.gestorLab.Flujo_Trabajo import FlujoTrabajo
    flujo = FlujoTrabajo(grafo_lineal_simple)
    flujo.nombre = "test_lineal"
    return flujo


# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTACIONES
# ──────────────────────────────────────────────────────────────────────────────

class TestImportaciones:
    def test_resultado_either(self):
        from nucleo.controlador.Resultado_Either import Ok, Err, LogEvento, NivelLog
        assert Ok and Err and LogEvento and NivelLog

    def test_controlador_bioimagen(self):
        from nucleo.controlador.Controlador_BioImagen import (
            BioImagenData, Dimensiones, ErrorBioImagen, ModoImagen
        )
        assert BioImagenData

    def test_flujo_trabajo(self):
        from nucleo.gestorLab.Flujo_Trabajo import (
            FlujoTrabajo, GrafoPipeline, NodoPipeline, AristaOperacion
        )
        assert FlujoTrabajo

    def test_validar_flujo(self):
        from nucleo.gestorLab.Validar_Flujo_Trabajo import (
            validar_pipeline, DiagnosticoPipeline
        )
        assert validar_pipeline

    def test_gestor_lab(self):
        from nucleo.gestorLab.Gestor_Lab import GestorLab
        assert GestorLab

    def test_constructor(self):
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        assert ConstructorFlujoTrabajo

    def test_estetica(self):
        from nucleo.analizador.plots.Estetica import Estetica
        assert Estetica

    def test_exportadores(self):
        from nucleo.analizador.exportacion.csv     import ExportadorCSV
        from nucleo.analizador.exportacion.tsv     import ExportadorTSV
        from nucleo.analizador.exportacion.parquet import ExportadorParquet
        from nucleo.analizador.exportacion.figures import ExportadorFiguras
        assert ExportadorCSV and ExportadorTSV


# ──────────────────────────────────────────────────────────────────────────────
# 2. RESULTADO EITHER
# ──────────────────────────────────────────────────────────────────────────────

class TestResultadoEither:
    def test_ok_unwrap(self):
        from nucleo.controlador.Resultado_Either import Ok
        assert Ok(42).unwrap() == 42

    def test_err_no_unwrap(self):
        from nucleo.controlador.Resultado_Either import Err
        with pytest.raises(Exception):
            Err("fallo").unwrap()

    def test_ok_es_ok(self):
        from nucleo.controlador.Resultado_Either import Ok
        assert Ok(1).es_ok() and not Ok(1).es_err()

    def test_err_es_err(self):
        from nucleo.controlador.Resultado_Either import Err
        assert Err("x").es_err() and not Err("x").es_ok()

    def test_map_ok(self):
        from nucleo.controlador.Resultado_Either import Ok
        r = Ok(3).map(lambda x: x * 2)
        assert r.es_ok() and r.unwrap() == 6

    def test_map_err_no_aplica(self):
        from nucleo.controlador.Resultado_Either import Err
        r = Err("e").map(lambda x: x * 2)
        assert r.es_err()

    def test_bind_encadenamiento(self):
        from nucleo.controlador.Resultado_Either import Ok, Err
        r = Ok(5).bind(lambda x: Ok(x + 1)).bind(lambda x: Ok(x * 2))
        assert r.unwrap() == 12

    def test_bind_cortocircuito_en_err(self):
        from nucleo.controlador.Resultado_Either import Ok, Err
        r = Ok(5).bind(lambda x: Err("fallo")).bind(lambda x: Ok(x * 999))
        assert r.es_err()

    def test_log_acumulado_en_ok(self):
        from nucleo.controlador.Resultado_Either import Ok, LogEvento, NivelLog
        ev = LogEvento(etapa="t", mensaje="msg", nivel=NivelLog.INFO)
        r  = Ok(1).log(ev)
        assert len(r._log) == 1

    def test_unwrap_or(self):
        from nucleo.controlador.Resultado_Either import Err
        assert Err("x").unwrap_or(99) == 99


# ──────────────────────────────────────────────────────────────────────────────
# 3. VALIDACIÓN — nunca bloquea, siempre Ok
# ──────────────────────────────────────────────────────────────────────────────

class TestValidarFlujoTrabajo:
    def test_validacion_siempre_retorna_ok(self, grafo_lineal_simple):
        from nucleo.gestorLab.Validar_Flujo_Trabajo import validar_pipeline
        res = validar_pipeline(grafo_lineal_simple)
        assert res.es_ok(), "validar_pipeline debe retornar Ok siempre"

    def test_diagnostico_tiene_eventos(self, grafo_lineal_simple):
        from nucleo.gestorLab.Validar_Flujo_Trabajo import validar_pipeline
        diag = validar_pipeline(grafo_lineal_simple).unwrap()
        assert len(diag.eventos) > 0

    def test_pipeline_valido_sin_errores_duros(self, grafo_lineal_simple):
        from nucleo.gestorLab.Validar_Flujo_Trabajo import validar_pipeline
        diag = validar_pipeline(grafo_lineal_simple).unwrap()
        assert diag.es_valido, f"Pipeline simple debe ser válido. Errores: {diag.errores}"

    def test_grafo_con_ciclo_reporta_error_no_excepcion(self):
        from nucleo.gestorLab.Validar_Flujo_Trabajo import validar_pipeline
        from nucleo.gestorLab.Flujo_Trabajo import GrafoPipeline, NodoPipeline, AristaOperacion
        from nucleo.gestorLab.Operacion import Operacion
        from nucleo.gestorLab.Categoria_Operacion import CategoriaOperacion, TipoDato
        from nucleo.controlador.Resultado_Either import Ok

        grafo = GrafoPipeline()
        grafo.agregar_nodo(NodoPipeline(id="A", tipo_dato=TipoDato.IMAGEN))
        grafo.agregar_nodo(NodoPipeline(id="B", tipo_dato=TipoDato.IMAGEN))
        op = Operacion("op", CategoriaOperacion.PREPROCESAMIENTO, lambda x: Ok(x))
        # Forzar ciclo saltando la validación del grafo
        grafo.aristas.append(AristaOperacion("A", "B", op))
        grafo.aristas.append(AristaOperacion("B", "A", op))

        # No debe lanzar excepción
        res = validar_pipeline(grafo)
        assert res.es_ok()
        diag = res.unwrap()
        assert diag.tiene_errores_duros, "Ciclo debe reportarse como error duro"

    def test_resumen_legible(self, grafo_lineal_simple):
        from nucleo.gestorLab.Validar_Flujo_Trabajo import validar_pipeline
        diag = validar_pipeline(grafo_lineal_simple).unwrap()
        resumen = diag.resumen()
        assert "VÁLIDO" in resumen or "ERRORES" in resumen


# ──────────────────────────────────────────────────────────────────────────────
# 4. GRAFO DAG
# ──────────────────────────────────────────────────────────────────────────────

class TestGrafoPipeline:
    def test_orden_topologico_lineal(self, grafo_lineal_simple):
        orden = grafo_lineal_simple.orden_topologico()
        assert orden[0] == "input"
        assert len(orden) == 2

    def test_nodos_iniciales_y_finales(self, grafo_lineal_simple):
        iniciales = [n.id for n in grafo_lineal_simple.nodos_iniciales()]
        finales   = [n.id for n in grafo_lineal_simple.nodos_finales()]
        assert "input"  in iniciales
        assert "salida" in finales

    def test_ciclo_lanza_valor_error(self):
        from nucleo.gestorLab.Flujo_Trabajo import GrafoPipeline, NodoPipeline, AristaOperacion
        from nucleo.gestorLab.Operacion import Operacion
        from nucleo.gestorLab.Categoria_Operacion import CategoriaOperacion, TipoDato
        from nucleo.controlador.Resultado_Either import Ok

        g = GrafoPipeline()
        g.agregar_nodo(NodoPipeline(id="X", tipo_dato=TipoDato.IMAGEN))
        g.agregar_nodo(NodoPipeline(id="Y", tipo_dato=TipoDato.IMAGEN))
        op = Operacion("op", CategoriaOperacion.PREPROCESAMIENTO, lambda x: Ok(x))
        g.aristas.append(AristaOperacion("X", "Y", op))
        g.aristas.append(AristaOperacion("Y", "X", op))
        with pytest.raises(ValueError):
            g.orden_topologico()


# ──────────────────────────────────────────────────────────────────────────────
# 5. FLUJO DE TRABAJO LINEAL
# ──────────────────────────────────────────────────────────────────────────────

class TestFlujoTrabajoLineal:
    def test_ejecucion_pasa_dato(self, flujo_lineal, bio_imagen_sintetica):
        flujo_lineal.reset_datos()
        res = flujo_lineal.ejecutar(bio_imagen_sintetica)
        assert res.es_ok()
        salida, logs = res.unwrap()
        assert len(salida) == 1
        val = list(salida.values())[0]
        assert val is bio_imagen_sintetica  # identidad

    def test_logs_generados(self, flujo_lineal, bio_imagen_sintetica):
        flujo_lineal.reset_datos()
        res = flujo_lineal.ejecutar(bio_imagen_sintetica)
        _, logs = res.unwrap()
        assert len(logs) >= 1

    def test_reset_limpia_datos(self, flujo_lineal, bio_imagen_sintetica):
        flujo_lineal.ejecutar(bio_imagen_sintetica)
        flujo_lineal.reset_datos()
        for nodo in flujo_lineal.grafo.nodos.values():
            assert nodo.data == []

    def test_doble_ejecucion_consistente(self, flujo_lineal, bio_imagen_sintetica):
        flujo_lineal.reset_datos()
        r1 = flujo_lineal.ejecutar(bio_imagen_sintetica)
        flujo_lineal.reset_datos()
        r2 = flujo_lineal.ejecutar(bio_imagen_sintetica)
        assert r1.es_ok() and r2.es_ok()


# ──────────────────────────────────────────────────────────────────────────────
# 6. FLUJO CON MERGE
# ──────────────────────────────────────────────────────────────────────────────

class TestFlujoMerge:
    def _construir_grafo_merge(self):
        from nucleo.gestorLab.Flujo_Trabajo import (
            GrafoPipeline, NodoPipeline, AristaOperacion
        )
        from nucleo.gestorLab.Operacion import Operacion
        from nucleo.gestorLab.Categoria_Operacion import CategoriaOperacion, TipoDato
        from nucleo.controlador.Resultado_Either import Ok

        op_id = lambda nombre, cat=CategoriaOperacion.PREPROCESAMIENTO: Operacion(
            nombre=nombre, categoria=cat,
            instancia_callable=lambda x: Ok(x),
            tipo_dato_salida=TipoDato.IMAGEN,
        )

        def merge_fn(datos):
            return {"rama_a": datos[0], "rama_b": datos[1]}

        g = GrafoPipeline()
        g.agregar_nodo(NodoPipeline(id="input",   tipo_dato=TipoDato.IMAGEN))
        g.agregar_nodo(NodoPipeline(id="rama_a",  tipo_dato=TipoDato.IMAGEN))
        g.agregar_nodo(NodoPipeline(id="rama_b",  tipo_dato=TipoDato.MASCARA))
        g.agregar_nodo(NodoPipeline(
            id="merge", tipo_dato=TipoDato.IMAGEN,
            es_merge=True, merge_fn=merge_fn,
        ))
        g.agregar_nodo(NodoPipeline(id="salida",  tipo_dato=TipoDato.IMAGEN))

        g.agregar_arista(AristaOperacion("input",  "rama_a", op_id("op_a")))
        g.agregar_arista(AristaOperacion("input",  "rama_b", op_id("op_b", CategoriaOperacion.SEGMENTADOR)))
        g.agregar_arista(AristaOperacion("rama_a", "merge",  op_id("pass_a")))
        g.agregar_arista(AristaOperacion("rama_b", "merge",  op_id("pass_b", CategoriaOperacion.SEGMENTADOR)))
        g.agregar_arista(AristaOperacion("merge",  "salida", op_id("post")))
        return g

    def test_merge_combina_dos_ramas(self, bio_imagen_sintetica):
        from nucleo.gestorLab.Flujo_Trabajo import FlujoTrabajo
        flujo = FlujoTrabajo(self._construir_grafo_merge())
        flujo.nombre = "test_merge"
        res = flujo.ejecutar(bio_imagen_sintetica)
        assert res.es_ok(), f"Merge falló: {res.error if res.es_err() else ''}"

    def test_merge_orden_topologico_correcto(self):
        g = self._construir_grafo_merge()
        orden = g.orden_topologico()
        assert orden.index("merge") > orden.index("rama_a")
        assert orden.index("merge") > orden.index("rama_b")

    def test_merge_log_contiene_evento_merge(self, bio_imagen_sintetica):
        from nucleo.gestorLab.Flujo_Trabajo import FlujoTrabajo
        from nucleo.controlador.Resultado_Either import NivelLog
        flujo = FlujoTrabajo(self._construir_grafo_merge())
        flujo.nombre = "test_merge_log"
        res = flujo.ejecutar(bio_imagen_sintetica)
        assert res.es_ok()
        _, logs = res.unwrap()
        merge_logs = [l for l in logs if l.etapa == "merge"]
        assert len(merge_logs) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 7. CONSTRUCTOR DESDE CONFIG DICT
# ──────────────────────────────────────────────────────────────────────────────

class TestConstructorFlujoTrabajo:
    def _config_minima(self):
        return {
            "nombre_pipeline": "test_construccion",
            "etapas": [
                {"preprocesamiento": [
                    {"metodo": "max_norm", "dominio": "normalizacion",
                     "canal": 0, "tipo_aplicacion": "global"}
                ]},
            ]
        }

    def test_construye_flujo_desde_config(self):
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        cfg   = self._config_minima()
        flujo = ConstructorFlujoTrabajo().construir(cfg)
        assert flujo.nombre == "test_construccion"
        assert len(flujo.grafo.nodos) >= 2

    def test_flujo_construido_es_dag_valido(self):
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        flujo = ConstructorFlujoTrabajo().construir(self._config_minima())
        orden = flujo.grafo.orden_topologico()
        assert len(orden) == len(flujo.grafo.nodos)

    def test_config_con_split_y_merge(self):
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        cfg = {
            "nombre_pipeline": "test_split_merge",
            "etapas": [
                {"preprocesamiento": [
                    {"metodo": "max_norm", "dominio": "normalizacion",
                     "canal": 0, "tipo_aplicacion": "global",
                     "anchor": "post_norm"}
                ]},
                {"split": {
                    "nombre": "rama_seg",
                    "desde":  "post_norm",
                    "etapas": [
                        {"preprocesamiento": [
                            {"metodo": "to_uint8", "dominio": "normalizacion",
                             "canal": 0, "tipo_aplicacion": "global"}
                        ]},
                        {"segmentacion": [
                            {"metodo": "otsu", "dominio": "segmentacion",
                             "canal": 0, "tipo_aplicacion": "por_corte_espaciotemporal"}
                        ]},
                    ],
                }},
                {"merge": {
                    "nombre":     "fusion",
                    "estrategia": "imagen_mascara",
                    "imagen":     "post_norm",
                    "mascara":    "rama_seg",
                }},
            ],
        }
        flujo = ConstructorFlujoTrabajo().construir(cfg)
        assert any(n.es_merge for n in flujo.grafo.nodos.values()), \
            "Debe haber al menos un nodo merge"


# ──────────────────────────────────────────────────────────────────────────────
# 8. YAML PARSING
# ──────────────────────────────────────────────────────────────────────────────

class TestYamlParsing:
    def test_yaml_corregido_se_parsea(self):
        import yaml
        ruta = Path(__file__).parent.parent / "test.yaml"
        if not ruta.exists():
            pytest.skip("test.yaml no encontrado")
        config = yaml.safe_load(ruta.read_text(encoding="utf-8"))
        assert "nombre_pipeline" in config
        assert "etapas" in config

    def test_yaml_split_tiene_estructura_correcta(self):
        import yaml
        ruta = Path(__file__).parent.parent / "test.yaml"
        if not ruta.exists():
            pytest.skip("test.yaml no encontrado")
        config = yaml.safe_load(ruta.read_text(encoding="utf-8"))
        etapas = config.get("etapas", [])
        claves = [list(e.keys())[0] for e in etapas if isinstance(e, dict)]
        assert "split" in claves, f"Debe haber una etapa 'split'. Claves: {claves}"
        assert "merge" in claves, f"Debe haber una etapa 'merge'. Claves: {claves}"

    def test_yaml_construye_pipeline(self):
        import yaml
        ruta = Path(__file__).parent.parent / "test.yaml"
        if not ruta.exists():
            pytest.skip("test.yaml no encontrado")
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        config = yaml.safe_load(ruta.read_text(encoding="utf-8"))
        flujo  = ConstructorFlujoTrabajo().construir(config)
        assert flujo.nombre == config["nombre_pipeline"]


# ──────────────────────────────────────────────────────────────────────────────
# 9. GESTOR LAB — con imagen real (skip si no existe)
# ──────────────────────────────────────────────────────────────────────────────

class TestGestorLab:
    def test_registrar_y_listar(self):
        from nucleo.gestorLab.Gestor_Lab import GestorLab
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        gestor = GestorLab()
        flujo  = ConstructorFlujoTrabajo().construir({
            "nombre_pipeline": "prueba",
            "etapas": [
                {"preprocesamiento": [
                    {"metodo": "max_norm", "dominio": "normalizacion",
                     "canal": 0, "tipo_aplicacion": "global"}
                ]}
            ]
        })
        gestor.registrar(flujo)
        assert "prueba" in gestor.listar()

    def test_diagnostico_no_lanza_excepcion(self):
        from nucleo.gestorLab.Gestor_Lab import GestorLab
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        gestor = GestorLab()
        flujo  = ConstructorFlujoTrabajo().construir({
            "nombre_pipeline": "test_diag",
            "etapas": [
                {"preprocesamiento": [
                    {"metodo": "max_norm", "dominio": "normalizacion",
                     "canal": 0, "tipo_aplicacion": "global"}
                ]}
            ]
        })
        gestor.registrar(flujo)
        # No debe lanzar excepción
        gestor.mostrar_diagnostico("test_diag")

    @pytest.mark.skipif(_IMAGEN_REAL is None, reason="Sin imagen real de prueba")
    def test_ejecucion_imagen_real(self):
        from nucleo.gestorLab.Gestor_Lab import GestorLab
        gestor = GestorLab()
        gestor.registrar_desde_yaml(RUTA_YAML)
        import yaml
        config = yaml.safe_load(RUTA_YAML.read_text())
        nombre = config["nombre_pipeline"]
        res    = gestor.ejecutar_desde_ruta(nombre, _IMAGEN_REAL, debug=False)
        # Debe retornar Ok o Err — nunca lanzar excepción
        assert res.es_ok() or res.es_err()


# ──────────────────────────────────────────────────────────────────────────────
# 10. MODO BATCH
# ──────────────────────────────────────────────────────────────────────────────

class TestModoBatch:
    def test_batch_con_lista_vacia(self):
        from nucleo.gestorLab.Gestor_Lab import GestorLab
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        gestor = GestorLab()
        flujo  = ConstructorFlujoTrabajo().construir({
            "nombre_pipeline": "batch_test",
            "etapas": [{"preprocesamiento": [
                {"metodo": "max_norm", "dominio": "normalizacion",
                 "canal": 0, "tipo_aplicacion": "global"}
            ]}]
        })
        gestor.registrar(flujo)
        res = gestor.ejecutar_batch("batch_test", [])
        assert isinstance(res, dict)
        assert len(res) == 0

    def test_batch_con_rutas_inexistentes_no_lanza(self):
        from nucleo.gestorLab.Gestor_Lab import GestorLab
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        gestor = GestorLab()
        flujo  = ConstructorFlujoTrabajo().construir({
            "nombre_pipeline": "batch_rutas",
            "etapas": [{"preprocesamiento": [
                {"metodo": "max_norm", "dominio": "normalizacion",
                 "canal": 0, "tipo_aplicacion": "global"}
            ]}]
        })
        gestor.registrar(flujo)
        rutas = [Path("/no/existe/img1.ids"), Path("/no/existe/img2.ids")]
        # No debe lanzar excepción — debe retornar Err por cada ruta
        res = gestor.ejecutar_batch("batch_rutas", rutas)
        assert len(res) == 2
        for r in res.values():
            assert r.es_err()  # imágenes inexistentes → Err de carga

    def test_batch_escribe_tsv(self, tmp_path):
        from nucleo.gestorLab.Gestor_Lab import GestorLab
        from nucleo.gestorLab.Constructor_Flujo_Trabajo import ConstructorFlujoTrabajo
        gestor = GestorLab()
        flujo  = ConstructorFlujoTrabajo().construir({
            "nombre_pipeline": "batch_tsv",
            "etapas": [{"preprocesamiento": [
                {"metodo": "max_norm", "dominio": "normalizacion",
                 "canal": 0, "tipo_aplicacion": "global"}
            ]}]
        })
        gestor.registrar(flujo)
        ruta_tsv = tmp_path / "batch_resumen.tsv"
        gestor.ejecutar_batch(
            "batch_tsv", [],
            ruta_log_batch=ruta_tsv,
        )
        assert ruta_tsv.exists()
        contenido = ruta_tsv.read_text()
        assert "archivo" in contenido  # cabecera TSV


# ──────────────────────────────────────────────────────────────────────────────
# 11. CONTROLADORES SOBRE ARRAYS SINTÉTICOS
# ──────────────────────────────────────────────────────────────────────────────

class TestControladores:
    def test_normalizador_max_norm(self, bio_imagen_sintetica):
        from nucleo.controlador.Controlador_Normalizador import Controlador_Normalizador
        from nucleo.controlador.Estrategias_Aplicacion import Global
        ctrl = Controlador_Normalizador()
        op   = ctrl.crear_operacion_max_norm(tipo=Global(), canal=0)  # imperativo
        # Usar via crear_operacion
        from nucleo.gestorLab.Categoria_Operacion import CategoriaOperacion
        operacion = ctrl.crear_operacion(
            nombre_metodo="max_norm",
            categoria=CategoriaOperacion.PREPROCESAMIENTO,
            tipo_aplicacion=Global(),
            canal=0,
        )
        res = operacion.ejecutar(bio_imagen_sintetica)
        assert res.es_ok() or res.es_err()  # no lanza excepción

    def test_segmentador_otsu_requiere_uint8(self, bio_imagen_sintetica):
        """Otsu requiere uint8/uint16 — con datos float debe retornar Err."""
        from nucleo.controlador.Controlador_Segmentador import Controlador_Segmentador
        from nucleo.controlador.Estrategias_Aplicacion import PorCorteEspaciotemporal
        from nucleo.gestorLab.Categoria_Operacion import CategoriaOperacion
        import numpy as np
        from nucleo.controlador.Controlador_BioImagen import BioImagenData, Dimensiones

        # Datos float — el segmentador debe rechazarlos con Err, no excepción
        datos_float = np.random.rand(1, 1, 1, 32, 32).astype(np.float64)
        data_float  = BioImagenData(
            datos=datos_float,
            dims=Dimensiones(T=1, Z=1, C=1, Y=32, X=32),
            canales=("c0",), ruta_origen=Path("x.ids"),
        )
        ctrl = Controlador_Segmentador()
        op   = ctrl.crear_operacion(
            nombre_metodo="otsu",
            categoria=CategoriaOperacion.SEGMENTADOR,
            tipo_aplicacion=PorCorteEspaciotemporal(),
            canal=0,
            tipo_salida=__import__(
                "nucleo.gestorLab.Operacion", fromlist=["TipoDato"]
            ).TipoDato.MASCARA,
        )
        res = op.ejecutar(data_float)
        # Debe ser Err (dtype inválido) sin lanzar excepción
        assert res.es_err()


# ──────────────────────────────────────────────────────────────────────────────
# 12. CUANTIFICADOR
# ──────────────────────────────────────────────────────────────────────────────

class TestCuantificador:
    def test_media_intensidad_retorna_float(self):
        """MediaIntensidad recibe (img_segmentada, img_procesada) y retorna un float."""
        import numpy as np
        from nucleo.cuantificador.intensidad.Cuantificadores_Intensidad import (
            MediaIntensidad,
        )

        rng = np.random.default_rng(0)
        mask = (rng.integers(0, 2, (64, 64)) * 255).astype(np.uint8)
        img = rng.integers(0, 4096, (64, 64)).astype(np.uint16)

        met = MediaIntensidad()
        res = met(mask, img)

        assert isinstance(res, float)
        assert np.isfinite(res)
        assert 0.0 <= res <= 4095.0


# ──────────────────────────────────────────────────────────────────────────────
# 13. EXPORTADORES
# ──────────────────────────────────────────────────────────────────────────────

class TestExportadores:
    def _df_prueba(self):
        import pandas as pd
        return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    def test_exportador_csv(self, tmp_path):
        from nucleo.analizador.exportacion.csv import ExportadorCSV
        ruta = tmp_path / "salida.csv"
        exp  = ExportadorCSV()
        exp(self._df_prueba(), ruta)
        assert ruta.exists()
        assert ruta.stat().st_size > 0

    def test_exportador_tsv(self, tmp_path):
        from nucleo.analizador.exportacion.tsv import ExportadorTSV
        ruta = tmp_path / "salida.tsv"
        exp  = ExportadorTSV()
        exp(self._df_prueba(), ruta)
        assert ruta.exists()
        contenido = ruta.read_text()
        assert "\t" in contenido

    def test_exportador_figura(self, tmp_path):
        import matplotlib.pyplot as plt
        from nucleo.analizador.exportacion.figures import ExportadorFiguras
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ruta = tmp_path / "figura.png"
        exp  = ExportadorFiguras(formato="png", dpi=72)
        exp(fig, ruta)
        plt.close(fig)
        assert ruta.exists()
        assert ruta.stat().st_size > 0


# ──────────────────────────────────────────────────────────────────────────────
# 14. PLOTS — generan figura sin excepción
# ──────────────────────────────────────────────────────────────────────────────

class TestPlots:
    def _df_numerica(self):
        import pandas as pd
        import numpy as np
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "media":    rng.normal(100, 20, 50),
            "std":      rng.normal(30,  5, 50),
            "area":     rng.normal(500, 100, 50),
        })

    def test_histograma_intensidad(self):
        import matplotlib
        matplotlib.use("Agg")
        from nucleo.analizador.plots.Plots_Estadisticos import HistogramaIntensidad
        from nucleo.analizador.plots.Estetica import Estetica
        fig = HistogramaIntensidad()(self._df_numerica(), Estetica())
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")

    def test_heatmap_correlacion(self):
        import matplotlib
        matplotlib.use("Agg")
        from nucleo.analizador.plots.Plots_Estadisticos import HeatmapCorrelacion
        from nucleo.analizador.plots.Estetica import Estetica
        fig = HeatmapCorrelacion()(self._df_numerica(), Estetica())
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")

    def test_boxplot_canales(self):
        import matplotlib
        matplotlib.use("Agg")
        from nucleo.analizador.plots.Plots_Estadisticos import BoxplotCanales
        from nucleo.analizador.plots.Estetica import Estetica
        fig = BoxplotCanales()(self._df_numerica(), Estetica())
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")

    def test_comparacion_canales_imagen(self, bio_imagen_sintetica):
        import matplotlib
        matplotlib.use("Agg")
        from nucleo.analizador.plots.Plots_Imagen import ComparacionCanales
        from nucleo.analizador.plots.Estetica import Estetica
        fig = ComparacionCanales()(bio_imagen_sintetica, Estetica())
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")

    def test_stack_viewer(self, bio_imagen_sintetica):
        import matplotlib
        matplotlib.use("Agg")
        from nucleo.analizador.plots.Plots_Imagen import StackViewer
        from nucleo.analizador.plots.Estetica import Estetica
        fig = StackViewer(canal=0, t=0)(bio_imagen_sintetica, Estetica())
        import matplotlib.pyplot as plt
        assert fig is not None
        plt.close("all")


# ──────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA DIRECTO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
