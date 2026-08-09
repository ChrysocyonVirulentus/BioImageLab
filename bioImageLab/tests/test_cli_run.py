#!/usr/bin/env python3
# === tests/test_cli_run.py ===
"""
Tests del comando `run` de la CLI.

Ejecutar:
    pytest bioImageLab/tests/test_cli_run.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parent.parent  # .../bioImageLab
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from bioImageLab.cli.main import cli  # ajustá el import si tu grupo `cli` vive en otro lado

IMAGEN     = ROOT / "test_data" / "glp1_1.ids"
YAML_VALIDO = ROOT / "test_cli.yaml"


@pytest.fixture
def yaml_en_tmp(tmp_path):
    """Copia test_cli.yaml pero con log/qc apuntando a tmp_path (no ensucia el repo)."""
    import yaml
    config = yaml.safe_load(YAML_VALIDO.read_text())
    config["ruta_log"] = str(tmp_path / "log.txt")
    config["ruta_qc"]  = str(tmp_path / "qc.png")
    destino = tmp_path / "pipeline.yaml"
    destino.write_text(yaml.safe_dump(config))
    return destino


def test_run_ok(yaml_en_tmp):
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--yaml", str(yaml_en_tmp), "--imagen", str(IMAGEN)])
    assert result.exit_code == 0, result.output
    assert "Ejecución completada" in result.output


def test_run_yaml_inexistente(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", "--yaml", str(tmp_path / "no_existe.yaml"), "--imagen", str(IMAGEN)]
    )
    assert result.exit_code != 0


def test_run_imagen_inexistente(yaml_en_tmp, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", "--yaml", str(yaml_en_tmp), "--imagen", str(tmp_path / "no_existe.tiff")]
    )
    assert result.exit_code != 0