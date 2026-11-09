"""Extraccion VERIFICADA de tablas HTML: descarga una pagina, lista sus tablas o guarda una como CSV trazable en data/work/.

Por que existe: los scripts que Ada Q escribia para raspar fallaban 0/3 (403 por falta de User-Agent propio y `pd.read_html(<str>)`, que
pandas 3 ya no acepta). Aqui eso queda resuelto en codigo y con las mismas garantias que el resto de ADA:

- Doble metodo: la tabla se lee con DOS analizadores independientes (`lxml` y `html5lib`) y deben coincidir celda por celda; el CSV guardado
  se relee con DuckDB (filas y sumas de columnas numericas).
- Procedencia: URL final, fecha UTC, estado HTTP, SHA-256 del HTML y del CSV en un `.provenance.json` junto al CSV.
- Seguridad: solo http(s) sin credenciales, bloquea direcciones locales/privadas (tambien tras redirecciones), respeta robots.txt,
  tope de 15 MB, escribe solo en data/work (nunca data/raw) y no pisa archivos existentes.
- Honestidad: no ejecuta JavaScript; si la pagina carga sus datos dinamicamente lo dice. Los numeros se convierten SOLO si el formato es
  inequivoco (miles con espacio, coma o punto decimal con evidencia en la columna); un '1.234' sin pista se queda como texto y se reporta.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import ipaddress
import json
import re
import socket
import unicodedata
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ada.config import DATA_WORK_DIR, assert_safe_write_path

USER_AGENT = "AdaResearchBot/1.0 (uso local academico; respeta robots.txt)"
MAX_BYTES = 15 * 1024 * 1024
MAX_REDIRECTS = 5
TIMEOUT_S = 30
MAX_ROWS = 100_000
MAX_TABLES_LISTED = 15
TOOL_VERSION = "ada_scrape_tables v1"


# ----------------------------------------------------------------------------- red y seguridad


@dataclass
class Fetched:
    url: str
    status: int
    content_type: str
    body: bytes


def _resolve(host: str) -> list[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})


def safe_url(url: str, resolve: Callable[[str], list[str]] = _resolve) -> str:
    """Valida la URL y devuelve la URL normalizada; ValueError con un mensaje claro si no es segura."""
    p = urllib.parse.urlsplit(str(url).strip())
    if p.scheme not in ("http", "https"):
        raise ValueError("solo se permiten URLs http o https")
    if not p.hostname:
        raise ValueError("la URL no tiene host")
    if p.username or p.password:
        raise ValueError("no se permiten credenciales dentro de la URL")
    try:
        addresses = resolve(p.hostname)
    except OSError as exc:
        raise ValueError(f"no se pudo resolver el host '{p.hostname}': {exc}") from exc
    for addr in addresses:
        ip = ipaddress.ip_address(addr.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValueError(f"el host resuelve a una direccion local o privada ({addr}): bloqueado por seguridad")
    return p.geturl()


def decode_html(body: bytes, content_type: str = "") -> str:
    """Decodifica con el charset del encabezado, si no el de <meta>, si no UTF-8."""
    m = re.search(r"charset=([\w.-]+)", content_type or "", re.I) or re.search(rb"<meta[^>]+charset=[\"']?([\w.-]+)", body[:4096], re.I)
    enc = (m.group(1).decode("ascii", "ignore") if m and isinstance(m.group(1), bytes) else (m.group(1) if m else "utf-8"))
    try:
        return body.decode(enc, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def http_fetch(url: str, *, resolve: Callable[[str], list[str]] = _resolve, get: Callable[..., Any] | None = None) -> Fetched:
    """GET con User-Agent propio, redirecciones validadas una a una y tope de tamano."""
    if get is None:
        import requests

        get = requests.get
    current = safe_url(url, resolve)
    for _ in range(MAX_REDIRECTS + 1):
        resp = get(current, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}, timeout=TIMEOUT_S, stream=True, allow_redirects=False)
        try:
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                current = safe_url(urllib.parse.urljoin(current, resp.headers["Location"]), resolve)
                continue
            body = bytearray()
            for chunk in resp.iter_content(65536):
                body += chunk
                if len(body) > MAX_BYTES:
                    raise ValueError(f"la pagina supera el limite de {MAX_BYTES // (1024 * 1024)} MB")
            return Fetched(current, int(resp.status_code), str(resp.headers.get("Content-Type", "")), bytes(body))
        finally:
            resp.close()
    raise ValueError(f"demasiadas redirecciones (mas de {MAX_REDIRECTS})")


def robots_decision(url: str, fetch: Callable[[str], Fetched]) -> tuple[bool, str]:
    """(permitido, nota). RFC 9309: 4xx = sin restricciones; 5xx = se asume prohibido; sin respuesta = permitido (se avisa)."""
    p = urllib.parse.urlsplit(url)
    try:
        r = fetch(f"{p.scheme}://{p.netloc}/robots.txt")
    except Exception as exc:  # noqa: BLE001 - un robots.txt inaccesible no debe tumbar la herramienta
        return True, f"robots.txt no consultable ({type(exc).__name__}); se continua"
    if 400 <= r.status < 500:
        return True, f"sin robots.txt (HTTP {r.status})"
    if r.status >= 500:
        return False, f"robots.txt devolvio HTTP {r.status}: se asume prohibido (RFC 9309)"
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(decode_html(r.body, r.content_type).splitlines())
    ok = rp.can_fetch(USER_AGENT, url)
    return ok, "robots.txt permite la ruta" if ok else "robots.txt PROHIBE esta ruta para ADA"


# ----------------------------------------------------------------------------- limpieza


_MARKER = re.compile(r"\[[^\[\]]{1,8}\]")           # notas al pie tipo [7], [a], [nota 1]
_SPACES = re.compile(r"[\s ​   ]+")
_CELL_OPEN = re.compile(r"(<t[dh]\b[^>]*>)", re.I)


def protect_cells(html: str) -> str:
    """Pone un espacio de ancho cero al inicio de cada celda para que pandas NO infiera tipos: sin esto '007' pasa a 7 y se pierde el cero
    (medido). Todo llega como texto y la conversion a numero la decide `clean_table`, que si respeta codigos con ceros a la izquierda."""
    return _CELL_OPEN.sub(r"\1​", html)
_PLACEHOLDERS = {"-", "–", "—", "n/d", "nd", "s/d", "n/a", "na", "?", "…", "..."}      # "no hay dato": cuentan como vacio
_P_INT = re.compile(r"^[+-]?\d+$")
_P_DOT = re.compile(r"^[+-]?\d+\.\d+$")
_P_COMMA = re.compile(r"^[+-]?\d+,\d+$")
_P_EU = re.compile(r"^[+-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?$")        # 1.234.567,89
_P_US = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")        # 1,234,567.89
_AMBIGUOUS_DOT = re.compile(r"^[+-]?\d{1,3}\.\d{3}$")             # 1.234: ¿miles europeos o decimal?
_AMBIGUOUS_COMMA = re.compile(r"^[+-]?\d{1,3},\d{3}$")            # 1,234: ¿miles americanos o decimal?


def convert_numeric(cells: list[str]) -> tuple[list[float | None] | None, bool]:
    """Convierte una columna de texto SOLO si su formato es inequivoco. Devuelve (valores, parece_numerica_pero_ambigua).

    Miles con espacios ya vienen sin ellos. 'valores' usa None donde habia un marcador de 'sin dato' ('-', '—', 'n/d'...). Casos ambiguos
    ('1.234' o '1,234' sin ninguna otra celda que aclare si el punto o la coma son decimales) NO se convierten: se reportan. Codigos con
    ceros a la izquierda ('007') tampoco: son texto.
    """
    vals = [None if c.lower() in _PLACEHOLDERS else c for c in cells]
    real = [v for v in vals if v is not None]
    if not real:
        return None, False
    if any(_P_INT.match(v) and len(v.lstrip("+-")) > 1 and v.lstrip("+-")[0] == "0" for v in real):
        return None, False

    def every(*pats: re.Pattern) -> bool:
        return all(any(p.match(v) for p in pats) for v in real)

    def out(fn: Callable[[str], float]) -> list[float | None]:
        return [None if v is None else fn(v) for v in vals]

    if every(_P_INT):
        return out(float), False
    if every(_P_INT, _P_DOT):
        if not any(re.search(r"\.\d{1,2}$|\.\d{4,}$", v) for v in real) and any(_AMBIGUOUS_DOT.match(v) for v in real):
            return None, True
        return out(float), False
    if every(_P_INT, _P_COMMA):
        if not any(re.search(r",\d{1,2}$|,\d{4,}$", v) for v in real) and any(_AMBIGUOUS_COMMA.match(v) for v in real):
            return None, True
        return out(lambda v: float(v.replace(",", "."))), False
    if every(_P_INT, _P_EU) and any(("," in v) or v.count(".") >= 2 for v in real):
        return out(lambda v: float(v.replace(".", "").replace(",", "."))), False
    if every(_P_INT, _P_US) and any(v.count(",") >= 2 or "." in v for v in real):
        return out(lambda v: float(v.replace(",", ""))), False
    return None, every(_P_INT, _P_DOT, _P_COMMA, _P_EU, _P_US)


def _text(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    out = _SPACES.sub(" ", _MARKER.sub("", v)).strip()
    return out if out else pd.NA


def _flat(col: Any) -> str:
    parts = [str(p) for p in col] if isinstance(col, tuple) else [str(col)]
    seen: list[str] = []
    for p in parts:
        p = _SPACES.sub(" ", _MARKER.sub("", p)).strip()
        if p and not p.lower().startswith("unnamed") and p not in seen:
            seen.append(p)
    return " ".join(seen)


def _label(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(text).lower()))


def _is_repeated_header(row: tuple, names: list[str]) -> bool:
    """Una fila de datos cuyas celdas de texto coinciden (>= 75 % de parecido) con sus nombres de columna en al menos el 60 % de los casos."""
    pairs = [(_label(v), _label(n)) for v, n in zip(row, names) if isinstance(v, str) and len(_label(v)) >= 2 and len(_label(n)) >= 2]
    if len(pairs) < 2:
        return False
    hits = sum(1 for a, b in pairs if a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.75)
    return hits / len(pairs) >= 0.6


def clean_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nombres planos y unicos, texto sin notas al pie, vacios fuera y numeros SOLO si son inequivocos (enteros o decimales con punto)."""
    df = raw.copy()
    names, used = [], set()
    for i, c in enumerate(df.columns):
        base = _flat(c) or f"col_{i + 1}"
        name, k = base, 2
        while name in used:
            name, k = f"{base}_{k}", k + 1
        used.add(name)
        names.append(name)
    df.columns = names
    df = df.astype(object).apply(lambda s: s.map(_text))
    is_header = [_is_repeated_header(row, names) for row in df.itertuples(index=False, name=None)]
    n_headers = sum(is_header)
    if n_headers:                                              # encabezados repetidos como filas de datos (muy comun en Wikipedia)
        df = df[[not h for h in is_header]].reset_index(drop=True)
    n0r, n0c = len(df), df.shape[1]
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    converted, looks, percent, placeholders = [], [], [], {}
    for c in df.columns:
        tokens = [(_SPACES.sub("", v).replace("−", "-") if isinstance(v, str) else None) for v in df[c].tolist()]
        if not any(t is not None for t in tokens):
            continue
        pct = any(t is not None and t.endswith("%") for t in tokens)
        tokens = [t[:-1] if (t is not None and t.endswith("%")) else t for t in tokens]
        present = [t for t in tokens if t is not None]
        values, ambiguous = convert_numeric(present)
        if values is None:
            if ambiguous:
                looks.append(c)                                    # parece numerica pero no se puede saber el formato: NO se adivina
            continue
        it = iter(values)
        full = [float("nan") if t is None else (float("nan") if (v := next(it)) is None else v) for t in tokens]
        ser = pd.Series(full, dtype="float64")
        if ser.dropna().map(float.is_integer).all():
            ser = ser.astype("Int64")
        df[c] = ser
        converted.append(c)
        percent += [c] if pct else []
        n_ph = sum(1 for t in present if t.lower() in _PLACEHOLDERS)
        if n_ph:
            placeholders[c] = n_ph
    report = {"repeated_header_rows_dropped": n_headers, "dropped_empty_rows": n0r - len(df), "dropped_empty_cols": n0c - df.shape[1], "numeric_columns": converted, "percent_stripped": percent,
              "placeholders_as_missing": placeholders, "looks_numeric_not_converted": looks}
    return df, report


# ----------------------------------------------------------------------------- doble metodo


def _canon(df: pd.DataFrame) -> list[list[str]]:
    rows = [[_flat(c) for c in df.columns]]
    rows += [["" if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False, name=None)]
    return rows


def compare_parsers(a: list[pd.DataFrame], b: list[pd.DataFrame]) -> list[str]:
    """Discrepancias entre lxml y html5lib (lista vacia = coinciden en numero de tablas, formas y TODAS las celdas)."""
    if len(a) != len(b):
        return [f"lxml encontro {len(a)} tablas y html5lib {len(b)}"]
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape:
            out.append(f"tabla {i}: forma {x.shape} (lxml) vs {y.shape} (html5lib)")
            continue
        cx, cy = _canon(x), _canon(y)
        diff = [(r, c) for r in range(len(cx)) for c in range(len(cx[r])) if cx[r][c] != cy[r][c]]
        if diff:
            r, c = diff[0]
            out.append(f"tabla {i}: {len(diff)} celdas distintas (p. ej. fila {r}, col {c}: {cx[r][c][:30]!r} vs {cy[r][c][:30]!r})")
    return out


def duckdb_roundtrip(path: Path, df: pd.DataFrame) -> list[str]:
    """Relee el CSV guardado con DuckDB: mismo numero de filas y misma suma en cada columna numerica."""
    import duckdb

    problems: list[str] = []
    con = duckdb.connect()
    try:
        n = con.execute("SELECT count(*) FROM read_csv_auto(?, header=true)", [str(path)]).fetchone()[0]
        if n != len(df):
            problems.append(f"DuckDB lee {n} filas y el DataFrame tiene {len(df)}")
        for c in df.select_dtypes("number").columns:
            q = '"' + str(c).replace('"', '""') + '"'
            s = con.execute(f"SELECT sum({q}) FROM read_csv_auto(?, header=true)", [str(path)]).fetchone()[0]
            want = float(df[c].sum())
            if s is None or abs(float(s) - want) > 1e-9 * max(1.0, abs(want)):
                problems.append(f"suma de '{c}': DuckDB {s} vs pandas {want}")
    finally:
        con.close()
    return problems


# ----------------------------------------------------------------------------- API


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "tabla"


def _short(v: Any, n: int = 40) -> str:
    s = "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _pick(tables: list[pd.DataFrame], table: int, match: str) -> tuple[int | None, str | None]:
    if table >= 0:
        return (table, None) if table < len(tables) else (None, f"table={table} fuera de rango (hay {len(tables)} tablas: 0..{len(tables) - 1})")
    needle = match.lower()
    hits = []
    for i, t in enumerate(tables):
        text = " ".join(_flat(c) for c in t.columns).lower() + " " + " ".join(t.astype(str).fillna("").to_numpy().ravel().tolist()).lower()
        if needle in text:
            hits.append((t.shape[0] * t.shape[1], i))
    return (max(hits)[1], None) if hits else (None, f"ninguna tabla contiene '{match}'")


def scrape_tables(
    url: str,
    table: int = -1,
    match: str = "",
    output_filename: str = "",
    *,
    fetch: Callable[[str], Fetched] | None = None,
    out_dir: Path | None = None,
    now: Callable[[], datetime] | None = None,
    respect_robots: bool = True,
) -> dict[str, Any]:
    """Lista las tablas de `url` (sin table ni match) o extrae una y la guarda como CSV verificado. Nunca lanza: devuelve {"error": ...}."""
    fetch = fetch or http_fetch
    stamp = (now or (lambda: datetime.now(timezone.utc)))()
    try:
        page = fetch(url)
        if respect_robots:
            allowed, robots_note = robots_decision(page.url, fetch)
            if not allowed:
                return {"error": f"No se descarga: {robots_note}.", "url": page.url, "robots": robots_note}
        else:
            robots_note = "robots.txt NO consultado (respect_robots=False)"
    except (ValueError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "hint": "Revisa la URL; solo http(s) publicas."}
    except Exception as exc:  # noqa: BLE001 - errores de requests/red con mensaje claro, sin traza
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "hint": "No se pudo descargar la pagina."}
    if page.status != 200:
        hint = "El sitio bloquea clientes automaticos o la pagina no existe." if page.status in (401, 403, 404, 429) else "Respuesta inesperada del servidor."
        return {"error": f"HTTP {page.status} al descargar la pagina.", "url": page.url, "hint": hint}
    if "html" not in page.content_type.lower() and "xml" not in page.content_type.lower():
        return {"error": f"el contenido es '{page.content_type or 'desconocido'}', no HTML.", "url": page.url}
    html = protect_cells(decode_html(page.body, page.content_type))
    try:
        primary = pd.read_html(io.StringIO(html), flavor="lxml")
    except ValueError:
        return {"error": "No se encontraron tablas HTML en la pagina.", "url": page.url,
                "hint": "Si los datos se cargan con JavaScript esta herramienta no puede verlos (no ejecuta un navegador)."}
    discrepancies: list[str] = []
    try:
        secondary = pd.read_html(io.StringIO(html), flavor="bs4")
        discrepancies = compare_parsers(primary, secondary)
    except Exception as exc:  # noqa: BLE001
        discrepancies = [f"el segundo analizador (html5lib) fallo: {type(exc).__name__}"]
    methods = ["pandas.read_html(lxml)", "pandas.read_html(html5lib): mismas tablas, formas y celdas"]
    base = {"url": page.url, "http_status": page.status, "robots": robots_note, "html_sha256": _sha256(page.body), "verification_methods": methods}

    if table < 0 and not match:
        listing = [{"i": i, "rows": int(t.shape[0]), "cols": int(t.shape[1]), "columns": [_short(_flat(c), 24) for c in t.columns[:6]]} for i, t in enumerate(primary[:MAX_TABLES_LISTED])]
        return {**base, "mode": "lista", "n_tables": len(primary), "tables": listing, "verified": not discrepancies, "discrepancies": discrepancies,
                "next": "Llama de nuevo con table=<i> (o match='texto de la tabla') para extraerla y guardarla como CSV."}

    idx, err = _pick(primary, table, match)
    if idx is None:
        return {**base, "error": err, "n_tables": len(primary), "verified": False}
    cleaned, report = clean_table(primary[idx])
    if len(cleaned) > MAX_ROWS:
        return {**base, "error": f"la tabla tiene {len(cleaned)} filas (maximo {MAX_ROWS}).", "verified": False}
    target_dir = Path(out_dir) if out_dir else DATA_WORK_DIR
    host = urllib.parse.urlsplit(page.url).hostname or "web"
    name = _slug(output_filename) if output_filename else f"web_{_slug(host)}_t{idx}_{stamp:%Y%m%d}"
    name = name if name.lower().endswith(".csv") else name + ".csv"
    path = assert_safe_write_path(target_dir / name)
    k = 2
    while path.exists():                                         # nunca pisa un archivo existente
        path = assert_safe_write_path(target_dir / f"{Path(name).stem}_{k}.csv")
        k += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path, index=False, encoding="utf-8")
    discrepancies += duckdb_roundtrip(path, cleaned)
    csv_sha = _sha256(path.read_bytes())
    prov_path = path.with_name(path.stem + ".provenance.json")
    provenance = {**base, "tool": TOOL_VERSION, "user_agent": USER_AGENT, "fetched_at_utc": stamp.isoformat(timespec="seconds"), "content_type": page.content_type,
                  "table_index": idx, "selected_by": f"match={match!r}" if table < 0 else f"table={table}", "rows": len(cleaned), "cols": cleaned.shape[1],
                  "csv_path": str(path), "csv_sha256": csv_sha, "cleaning": report, "verified": not discrepancies, "discrepancies": discrepancies}
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=1), encoding="utf-8")
    warnings = []
    if report["looks_numeric_not_converted"]:
        warnings.append(f"columnas con numeros de formato AMBIGUO (1.234 / 1,234 sin pista de si el separador es decimal o de miles) que se dejaron como TEXTO: {report['looks_numeric_not_converted']}")
    if cleaned.isna().mean().mean() > 0.5:
        warnings.append("mas de la mitad de las celdas estan vacias: revisa que sea la tabla correcta")
    methods = methods + ["DuckDB read_csv_auto: filas y suma de cada columna numerica"]
    return {**base, "mode": "extraccion", "verification_methods": methods, "verified": not discrepancies, "discrepancies": discrepancies,
            "table_index": idx, "output_path": str(path), "csv_sha256": csv_sha, "provenance_path": str(prov_path), "rows": len(cleaned), "cols": cleaned.shape[1],
            "columns": [_short(c, 40) for c in cleaned.columns[:20]], "preview": [[_short(v) for v in r] for r in cleaned.head(3).itertuples(index=False, name=None)],
            "cleaning": report, "warnings": warnings}
