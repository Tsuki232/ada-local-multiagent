"""Paridad entre las dos superficies de herramientas (CLI y MCP): lo que el CLI ofrece a Ada tiene que existir tambien en el MCP."""

from __future__ import annotations

import importlib.util

import pytest

from ada.config import PROJECT_ROOT


@pytest.fixture(scope="module")
def inv():
    spec = importlib.util.spec_from_file_location("tool_inventory_test", PROJECT_ROOT / "scripts" / "tool_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.inventory()


def test_every_cli_tool_exists_in_the_mcp_server(inv):
    assert not inv["cli_not_in_mcp"], f"herramientas que el CLI ofrece y el MCP no tiene: {inv['cli_not_in_mcp']}"


def test_the_new_statistics_tools_are_visible_on_both_surfaces(inv):
    for name in ("ada_inferential_stats", "ada_probability_bayes"):
        assert name in inv["cli"] and name in inv["mcp"], name


def test_routing_packs_only_name_real_tools(inv):
    for pack, names in inv["packs"].items():
        assert set(names) <= set(inv["cli"]), f"{pack}: {set(names) - set(inv['cli'])}"


def test_cli_and_mcp_expose_the_same_arguments_for_the_statistics_tools():
    """Los dos caminos deben aceptar la misma forma de datos: se comprueba ejecutando AMBOS con la misma entrada."""
    import asyncio
    import json

    from ada.stats.inference import compact_result, run_inference
    from ada.tools import mcp_server

    data = {"x": [1, 2, 3, 4, 5], "y": [3, 4, 5, 6, 7]}
    via_mcp = json.loads(mcp_server.ada_inferential_stats("two_groups", json.dumps(data)))
    direct = compact_result(run_inference("two_groups", data))
    assert via_mcp["p_value"] == pytest.approx(direct["p_value"]) and via_mcp["test"] == direct["test"]
    prob = json.loads(mcp_server.ada_probability_bayes("bayes_priors", json.dumps({"params": {"prior": "0.01", "sensitivity": "0.99", "false_positive_rate": "0.05"}})))
    assert prob["posterior_fraction"] == "1/6"
    err = json.loads(mcp_server.ada_probability_bayes("event", json.dumps({"path": "C:/Windows/win.ini", "a": {"column": "x", "op": ">", "value": 1}})))
    assert "error" in err, "una ruta fuera de data/ debe devolver un error claro, no una excepcion"
    _ = asyncio
