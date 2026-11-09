"""Scraping verificado de tablas HTML: valores calculados a mano sobre una pagina de ejemplo, sin tocar la red."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone

import pandas as pd
import pytest

from ada.config import DATA_WORK_DIR
from ada.scraping import web_tables as wt
from ada.scraping.web_tables import Fetched, clean_table, compare_parsers, decode_html, http_fetch, protect_cells, robots_decision, safe_url, scrape_tables

URL = "https://datos.example.org/paises"
HTML = """<html><body>
<table><tr><th>País</th><th>Población</th><th>Cambio (%)</th><th>Código</th></tr>
<tr><td>Alfa[1]</td><td>1&nbsp;400&nbsp;000</td><td>1,5 %</td><td>007</td></tr>
<tr><td>Beta</td><td>250 000</td><td>-0,3 %</td><td>012</td></tr>
<tr><td>Gamma[a]</td><td>3 000</td><td>0,0 %</td><td>101</td></tr></table>
<table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2.5</td></tr><tr><td></td><td>4.5</td></tr></table>
</body></html>"""
BODY = HTML.encode("utf-8")
WHEN = datetime(2026, 11, 7, 12, 0, 0, tzinfo=timezone.utc)


def make_fetch(pages=None, robots=("User-agent: *\nAllow: /\n", 200)):
    pages = pages if pages is not None else {URL: (BODY, "text/html; charset=utf-8", 200)}

    def fetch(url):
        if url.endswith("/robots.txt"):
            return Fetched(url, robots[1], "text/plain", robots[0].encode())
        body, ctype, status = pages[url]
        return Fetched(url, status, ctype, body)

    return fetch


@pytest.fixture()
def out_dir():
    d = DATA_WORK_DIR / "_pytest_tmp" / f"scrape_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def run(out_dir, **kw):
    return scrape_tables(URL, fetch=kw.pop("fetch", make_fetch()), out_dir=out_dir, now=lambda: WHEN, **kw)


# ----------------------------------------------------------------------------- listar y extraer


def test_listing_shows_both_tables_with_hand_counted_shapes(out_dir):
    res = run(out_dir)
    assert res["mode"] == "lista" and res["n_tables"] == 2 and res["verified"] is True and res["discrepancies"] == []
    assert (res["tables"][0]["rows"], res["tables"][0]["cols"]) == (3, 4) and (res["tables"][1]["rows"], res["tables"][1]["cols"]) == (2, 2)
    assert res["tables"][0]["columns"] == ["País", "Población", "Cambio (%)", "Código"]
    assert res["http_status"] == 200 and res["html_sha256"] == hashlib.sha256(BODY).hexdigest()
    assert not list(out_dir.iterdir()), "listar NO escribe nada"


def test_extraction_cleans_footnotes_converts_only_unambiguous_numbers_and_keeps_leading_zeros(out_dir):
    res = run(out_dir, table=0)
    assert res["mode"] == "extraccion" and res["verified"] is True and res["discrepancies"] == []
    assert (res["rows"], res["cols"]) == (3, 4)
    df = pd.read_csv(res["output_path"], dtype=str, keep_default_na=False)
    assert df["País"].tolist() == ["Alfa", "Beta", "Gamma"], "las notas [1] y [a] se quitan"
    assert df["Código"].tolist() == ["007", "012", "101"], "los ceros a la izquierda NO se pierden (pandas por defecto los volvia 7, 12, 101)"
    num = pd.read_csv(res["output_path"])
    assert num["Población"].tolist() == [1400000, 250000, 3000] and int(num["Población"].sum()) == 1653000
    assert res["cleaning"]["numeric_columns"] == ["Población", "Cambio (%)"] and res["cleaning"]["percent_stripped"] == ["Cambio (%)"]
    assert num["Cambio (%)"].tolist() == [1.5, -0.3, 0.0], "coma decimal con evidencia ('1,5'): 1,5 % -> 1.5 (en unidades de porcentaje)"
    assert res["cleaning"]["looks_numeric_not_converted"] == [] and res["warnings"] == []


def test_provenance_hashes_and_methods_are_recorded(out_dir):
    res = run(out_dir, table=1)
    csv = res["output_path"]
    assert res["csv_sha256"] == hashlib.sha256(open(csv, "rb").read()).hexdigest()
    prov = json.loads(open(res["provenance_path"], encoding="utf-8").read())
    assert prov["url"] == URL and prov["fetched_at_utc"] == "2026-11-07T12:00:00+00:00" and prov["html_sha256"] == hashlib.sha256(BODY).hexdigest()
    assert prov["csv_sha256"] == res["csv_sha256"] and prov["table_index"] == 1 and prov["selected_by"] == "table=1" and prov["tool"].startswith("ada_scrape_tables")
    assert any("html5lib" in m for m in res["verification_methods"]) and any("DuckDB" in m for m in res["verification_methods"])


def test_second_table_with_an_empty_cell_sums_to_hand_values(out_dir):
    res = run(out_dir, table=1)
    df = pd.read_csv(res["output_path"])
    assert (res["rows"], res["cols"]) == (2, 2) and df["x"].sum() == 1.0 and df["y"].sum() == 7.0 and int(df["x"].isna().sum()) == 1
    assert res["verified"] is True


def test_match_picks_the_largest_table_containing_the_text_and_errors_are_clear(out_dir):
    assert run(out_dir, match="Gamma")["table_index"] == 0
    assert "ninguna tabla contiene" in run(out_dir, match="zzz")["error"]
    assert "fuera de rango" in run(out_dir, table=5)["error"]


def test_existing_files_are_never_overwritten(out_dir):
    a = run(out_dir, table=0, output_filename="mi tabla")
    b = run(out_dir, table=0, output_filename="mi tabla")
    assert a["output_path"].endswith("mi_tabla.csv") and b["output_path"].endswith("mi_tabla_2.csv")


def test_writing_outside_the_allowed_folders_is_refused(tmp_path):
    from ada.config import SecurityViolationError

    with pytest.raises(SecurityViolationError):
        scrape_tables(URL, table=0, fetch=make_fetch(), out_dir=tmp_path, now=lambda: WHEN)


# ----------------------------------------------------------------------------- errores honestos


def test_http_errors_wrong_content_and_pages_without_tables_explain_themselves(out_dir):
    assert "403" in run(out_dir, fetch=make_fetch({URL: (b"no", "text/html", 403)}))["error"]
    assert "no HTML" in run(out_dir, fetch=make_fetch({URL: (b"{}", "application/json", 200)}))["error"]
    res = run(out_dir, fetch=make_fetch({URL: (b"<html><body><p>hola</p></body></html>", "text/html", 200)}))
    assert "No se encontraron tablas" in res["error"] and "JavaScript" in res["hint"]


def test_a_second_parser_disagreement_makes_the_result_unverified(out_dir, monkeypatch):
    monkeypatch.setattr(wt, "compare_parsers", lambda a, b: ["tabla 0: 1 celdas distintas (simulado)"])
    res = run(out_dir, table=0)
    assert res["verified"] is False and res["discrepancies"] == ["tabla 0: 1 celdas distintas (simulado)"]


# ----------------------------------------------------------------------------- robots.txt


@pytest.mark.parametrize(
    "robots,status,allowed",
    [("User-agent: *\nDisallow: /", 200, False), ("User-agent: *\nDisallow: /privado", 200, True), ("", 404, True), ("", 503, False), ("User-agent: AdaResearchBot\nDisallow: /", 200, False)],
)
def test_robots_rules_follow_rfc_9309(robots, status, allowed):
    ok, _note = robots_decision(URL, make_fetch(robots=(robots, status)))
    assert ok is allowed


def test_a_page_forbidden_by_robots_is_not_downloaded(out_dir):
    res = run(out_dir, fetch=make_fetch(robots=("User-agent: *\nDisallow: /", 200)))
    assert "PROHIBE" in res["error"]
    assert "robots.txt NO consultado" in run(out_dir, fetch=make_fetch(robots=("User-agent: *\nDisallow: /", 200)), respect_robots=False)["robots"]


# ----------------------------------------------------------------------------- seguridad de URL


@pytest.mark.parametrize("addr", ["127.0.0.1", "10.0.0.5", "192.168.1.20", "169.254.169.254", "::1", "0.0.0.0"])
def test_local_and_private_addresses_are_blocked(addr):
    with pytest.raises(ValueError, match="local o privada"):
        safe_url("http://host.example/x", resolve=lambda h: [addr])


def test_unsafe_schemes_and_credentials_are_rejected_and_public_hosts_pass():
    pub = lambda h: ["93.184.216.34"]  # noqa: E731
    for bad in ("ftp://x.example/a", "file:///etc/passwd", "javascript:alert(1)", "http://user:pw@x.example/", "http:///sinhost"):
        with pytest.raises(ValueError):
            safe_url(bad, resolve=pub)
    assert safe_url("https://x.example/a?b=1", resolve=pub) == "https://x.example/a?b=1"


class FakeResp:
    def __init__(self, status=200, headers=None, chunks=(b"<html></html>",)):
        self.status_code, self.headers, self._chunks, self.closed = status, headers or {"Content-Type": "text/html"}, chunks, False

    def iter_content(self, _n):
        yield from self._chunks

    def close(self):
        self.closed = True


def test_redirects_are_validated_hop_by_hop_and_size_is_capped(monkeypatch):
    table = {"ok.example": ["93.184.216.34"], "evil.example": ["127.0.0.1"]}
    resolve = lambda h: table[h]  # noqa: E731
    redirect = FakeResp(302, {"Location": "http://evil.example/admin"})
    with pytest.raises(ValueError, match="local o privada"):
        http_fetch("http://ok.example/start", resolve=resolve, get=lambda *a, **k: redirect)
    assert redirect.closed
    with pytest.raises(ValueError, match="demasiadas redirecciones"):
        http_fetch("http://ok.example/loop", resolve=resolve, get=lambda *a, **k: FakeResp(302, {"Location": "http://ok.example/loop"}))
    with monkeypatch.context() as small:                       # el tope pequeno SOLO dentro de este bloque
        small.setattr(wt, "MAX_BYTES", 10)
        with pytest.raises(ValueError, match="limite"):
            http_fetch("http://ok.example/big", resolve=resolve, get=lambda *a, **k: FakeResp(chunks=(b"x" * 8, b"y" * 8)))
    seen = {}

    def get(url, **kw):
        seen.update(kw["headers"])
        return FakeResp(chunks=(b"<html>ok</html>",))

    got = http_fetch("http://ok.example/p", resolve=resolve, get=get)
    assert got.body == b"<html>ok</html>" and got.status == 200 and seen["User-Agent"].startswith("AdaResearchBot/")


# ----------------------------------------------------------------------------- piezas sueltas


def test_clean_table_flattens_headers_dedups_and_respects_codes():
    raw = pd.DataFrame([["\u200b1\u00a0234", "\u200b2.5", "\u200b01234", "\u200bx[3]"], [None, None, None, None], ["\u200b5", "\u200b3.5", "\u200b00987", "\u200by"]],
                       columns=pd.MultiIndex.from_tuples([("Población", "2020"), ("Valor", "Valor"), ("Valor", "Valor"), ("Unnamed: 3_level_0", "Etiqueta")]))
    df, rep = clean_table(raw)
    assert list(df.columns) == ["Población 2020", "Valor", "Valor_2", "Etiqueta"]
    assert df["Población 2020"].tolist() == [1234, 5] and df["Valor"].tolist() == [2.5, 3.5], "NBSP como separador de miles; fila vacia fuera"
    assert df["Valor_2"].tolist() == ["01234", "00987"] and df["Etiqueta"].tolist() == ["x", "y"]
    assert rep["dropped_empty_rows"] == 1 and rep["numeric_columns"] == ["Población 2020", "Valor"]


@pytest.mark.parametrize(
    "cells,expected,ambiguous",
    [
        (["1435447000", "1401499000"], [1435447000.0, 1401499000.0], False),                  # miles con espacio ya quitados
        (["17,50", "0,84", "4,20"], [17.5, 0.84, 4.2], False),                                 # coma decimal con evidencia
        (["2.5", "-1.25"], [2.5, -1.25], False),
        (["1.234", "2.5"], [1.234, 2.5], False),                                               # '2.5' prueba que el punto es decimal
        (["1.234,5", "2.000"], [1234.5, 2000.0], False),                                        # formato europeo
        (["1,234,567", "89"], [1234567.0, 89.0], False),                                        # formato americano (dos grupos de miles)
        (["12", "3,5"], [12.0, 3.5], False),
        (["1", "-", "3"], [1.0, None, 3.0], False),                                             # '-' = sin dato
        (["1.234", "5.678"], None, True),                                                       # ¿1234 o 1.234? no se adivina
        (["1,234", "5,678"], None, True),
        (["007", "012"], None, False),                                                          # codigos: texto
        (["a", "1"], None, False),
        (["-", "—"], None, False),                                                              # solo marcadores: no hay numeros
    ],
)
def test_convert_numeric_only_converts_unambiguous_formats(cells, expected, ambiguous):
    values, amb = wt.convert_numeric(cells)
    assert values == expected and amb is ambiguous


def test_clean_table_handles_percent_unicode_minus_and_placeholders():
    raw = pd.DataFrame({"v": ["​12,5 %", "​−3 %", "​-"], "t": ["​ok", "​no", "​si"]})
    df, rep = clean_table(raw)
    assert df["v"].tolist()[:2] == [12.5, -3.0] and pd.isna(df["v"].iloc[2]) and df["t"].tolist() == ["ok", "no", "si"]
    assert rep["percent_stripped"] == ["v"] and rep["placeholders_as_missing"] == {"v": 1} and rep["numeric_columns"] == ["v"]
    amb, rep2 = clean_table(pd.DataFrame({"a": ["​1.234", "​5.678"]}))
    assert amb["a"].tolist() == ["1.234", "5.678"] and rep2["looks_numeric_not_converted"] == ["a"], "ambiguo: se reporta y no se convierte"


def test_repeated_header_rows_are_dropped_but_similar_looking_data_is_kept():
    """Hallazgo con la tabla real de Wikipedia: el encabezado se repetia como fila de datos y bloqueaba la conversion de TODAS las columnas."""
    hdr = ["​Nombre", "​Valor"]
    raw = pd.DataFrame([hdr, ["​A", "​1"], hdr, ["​B", "​2"], ["​País Vasco", "​3"]], columns=hdr)
    df, rep = clean_table(raw)
    assert df["Nombre"].tolist() == ["A", "B", "País Vasco"] and df["Valor"].tolist() == [1, 2, 3] and rep["repeated_header_rows_dropped"] == 2
    assert rep["numeric_columns"] == ["Valor"]


def test_end_to_end_a_repeated_header_row_does_not_stop_numeric_conversion(out_dir):
    page = ("<table><tr><th>Nombre</th><th>Valor</th></tr><tr><td>Nombre</td><td>Valor</td></tr>"
            "<tr><td>A</td><td>1</td></tr><tr><td>Nombre</td><td>Valor</td></tr><tr><td>B</td><td>2</td></tr><tr><td>C</td><td>3</td></tr></table>").encode()
    res = run(out_dir, table=0, fetch=make_fetch({URL: (page, "text/html", 200)}))
    assert res["rows"] == 3 and res["cleaning"]["repeated_header_rows_dropped"] == 2 and res["cleaning"]["numeric_columns"] == ["Valor"]
    assert int(pd.read_csv(res["output_path"])["Valor"].sum()) == 6 and res["verified"] is True


def test_protect_cells_marks_every_td_and_th_only():
    out = protect_cells('<TABLE><tr><th class="a">H</th><td>1</td></tr></TABLE>')
    assert out.count("\u200b") == 2 and "<th class=\"a\">\u200bH" in out and "<td>\u200b1" in out


def test_decode_html_uses_header_then_meta_then_utf8():
    assert decode_html("café".encode("latin-1"), "text/html; charset=ISO-8859-1") == "café"
    assert decode_html(b'<meta charset="iso-8859-1">caf\xe9') .endswith("café")
    assert decode_html("café".encode("utf-8")) == "café"


def test_compare_parsers_reports_count_shape_and_cell_differences():
    a = pd.DataFrame({"c": ["1", "2"]})
    assert compare_parsers([a], [a.copy()]) == []
    assert "1 tablas y html5lib 0" in compare_parsers([a], [])[0]
    assert "forma" in compare_parsers([a], [pd.DataFrame({"c": ["1"]})])[0]
    assert "1 celdas distintas" in compare_parsers([a], [pd.DataFrame({"c": ["1", "9"]})])[0]
    nan = pd.DataFrame({"c": ["1", None]})
    assert compare_parsers([nan], [nan.copy()]) == [], "NaN contra NaN no es una diferencia (era un falso positivo medido)"


def test_duckdb_roundtrip_detects_a_csv_that_does_not_match_the_dataframe(out_dir):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = out_dir / "t.csv"
    df.to_csv(path, index=False)
    assert wt.duckdb_roundtrip(path, df) == []
    problems = wt.duckdb_roundtrip(path, pd.DataFrame({"a": [1, 2, 4], "b": ["x", "y", "z"]}))
    assert problems and "suma de 'a': DuckDB 6" in problems[0]


def test_the_mcp_tool_returns_a_clear_error_for_local_urls_without_touching_the_network():
    from ada.tools import mcp_server

    res = json.loads(mcp_server.ada_scrape_tables("http://127.0.0.1:9/x"))
    assert "local o privada" in res["error"]
    res = json.loads(mcp_server.ada_scrape_tables("file:///C:/Windows/win.ini"))
    assert "http" in res["error"]
