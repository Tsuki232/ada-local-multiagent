"""Inventario de herramientas de ADA: que ve el CLI, que expone el MCP, diferencias y COSTO EN TOKENS de cada esquema (Fase 0 del plan).

  python scripts/tool_inventory.py
Sirve para vigilar el presupuesto de contexto: cada esquema se reenvia en cada peticion y le quita espacio a Ada para pensar y responder.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Nombres distintos para la MISMA herramienta en CLI y MCP (deuda conocida; la Fase 1 del plan la elimina con un registro unico).
KNOWN_ALIASES = {"ada_math_textbook_studio": "ada_build_math_textbook"}


def _load_cli():
    spec = importlib.util.spec_from_file_location("ada_cli_inventory", PROJECT_ROOT / "scripts" / "ada_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inventory() -> dict:
    from ada.tools import budget, mcp_server
    from ada.tools.routing import PACKS

    cli = _load_cli()
    cli_tools = {t["function"]["name"]: t for t in cli.TOOLS_SCHEMA}
    mcp_tools = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    have_tok = budget.available()
    cli_tokens = {n: budget.schema_tokens(t) for n, t in cli_tools.items()} if have_tok else {}
    mcp_tokens = ({n: budget.count(json.dumps({"name": n, "description": t.description, "inputSchema": t.input_schema}, ensure_ascii=False)) for n, t in mcp_tools.items()} if have_tok else {})
    resolved = {KNOWN_ALIASES.get(n, n) for n in cli_tools}
    return {
        "cli": sorted(cli_tools), "mcp": sorted(mcp_tools), "cli_not_in_mcp": sorted(n for n in cli_tools if KNOWN_ALIASES.get(n, n) not in mcp_tools),
        "mcp_not_in_cli": sorted(set(mcp_tools) - resolved), "cli_tokens": cli_tokens, "mcp_tokens": mcp_tokens, "packs": PACKS,
        "cli_fixed_overhead": budget.rendered_overhead(cli.SYSTEM_PROMPT, cli.TOOLS_SCHEMA) if have_tok else None,
        "mcp_total": (sum(mcp_tokens.values()) if have_tok else None),
    }


def main() -> int:
    inv = inventory()
    print(f"CLI: {len(inv['cli'])} herramientas | MCP: {len(inv['mcp'])} herramientas")
    if inv["cli_fixed_overhead"] is not None:
        print(f"\nCLI  arranque fijo con TODAS las herramientas: {inv['cli_fixed_overhead']} tokens")
        for n, t in sorted(inv["cli_tokens"].items(), key=lambda kv: -kv[1]):
            print(f"       {n:<32} {t:>5}")
        print(f"\nMCP  esquemas de {len(inv['mcp'])} herramientas: {inv['mcp_total']} tokens (+ instrucciones del servidor)")
        for n, t in sorted(inv["mcp_tokens"].items(), key=lambda kv: -kv[1])[:8]:
            print(f"       {n:<32} {t:>5}")
    else:
        print("(sin tokenizador: no se calculan tokens)")
    print("\nSolo en el CLI :", inv["cli_not_in_mcp"] or "ninguna")
    print("Solo en el MCP :", len(inv["mcp_not_in_cli"]), "herramientas (el CLI envia solo un subconjunto; es lo esperado)")
    print("\nPaquetes de enrutamiento:", json.dumps(inv["packs"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
