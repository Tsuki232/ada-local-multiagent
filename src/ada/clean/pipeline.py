"""Limpieza, filtrado, ordenamiento y adaptacion trazable de datos (datos crudos inmutables)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ada.audit.ledger import compute_sha256, record_audit_event
from ada.config import DATA_WORK_DIR, assert_safe_write_path
from ada.ingest.loader import load_dataset_frame


def _slugify_column(name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "col"


def clean_and_adapt_dataset(
    input_path: str | Path,
    *,
    output_filename: str = "",
    table_or_sheet: str = "",
    where_sql: str = "",
    order_by_sql: str = "",
    select_columns: list[str] | None = None,
    drop_duplicates: bool = True,
    normalize_column_names: bool = False,
    missing_strategy: str = "keep",
) -> dict[str, Any]:
    """Limpia, filtra, ordena y adapta un dataset guardando SIEMPRE una copia trazable en data/work/.

    Parametros:
        missing_strategy: 'keep' (recomendado antes de ML para evitar fuga train/test),
                          'drop_rows', 'median_mode' (solo analisis descriptivo).
    """
    src_path = Path(input_path).resolve()
    df_orig, meta = load_dataset_frame(src_path, table_or_sheet=table_or_sheet)
    rows_before, cols_before = df_orig.shape
    input_hash = meta["sha256"]

    work_df = df_orig.copy()
    steps_applied: list[str] = []

    if normalize_column_names:
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        for c in work_df.columns:
            base = _slugify_column(str(c))
            idx = seen.get(base, 0)
            seen[base] = idx + 1
            new_cols.append(base if idx == 0 else f"{base}_{idx}")
        work_df.columns = new_cols
        steps_applied.append("normalize_column_names")

    if select_columns:
        missing_cols = [c for c in select_columns if c not in work_df.columns]
        if missing_cols:
            raise ValueError(f"Columnas inexistentes para seleccionar: {missing_cols}. Disponibles: {list(work_df.columns)}")
        work_df = work_df[list(select_columns)].copy()
        steps_applied.append(f"select_columns({len(select_columns)})")

    if where_sql.strip() or order_by_sql.strip():
        conn = duckdb.connect(":memory:")
        try:
            conn.register("src_tbl", work_df)
            query = "SELECT * FROM src_tbl"
            if where_sql.strip():
                query += f" WHERE {where_sql.strip()}"
                steps_applied.append(f"where_sql({where_sql.strip()})")
            if order_by_sql.strip():
                query += f" ORDER BY {order_by_sql.strip()}"
                steps_applied.append(f"order_by_sql({order_by_sql.strip()})")
            work_df = conn.execute(query).df()
        finally:
            conn.close()

    for col in work_df.select_dtypes(include=["object", "string"]).columns:
        non_null_mask = work_df[col].notna()
        work_df.loc[non_null_mask, col] = work_df.loc[non_null_mask, col].astype(str).str.strip()
        work_df[col] = work_df[col].replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA, "None": pd.NA})
    steps_applied.append("trim_strings_and_standardize_null_tokens")

    dups_removed = 0
    if drop_duplicates:
        before_dup = len(work_df)
        work_df = work_df.drop_duplicates().reset_index(drop=True)
        dups_removed = before_dup - len(work_df)
        steps_applied.append(f"drop_duplicates(removed={dups_removed})")

    strategy = missing_strategy.strip().lower()
    leakage_warning = ""
    if strategy == "drop_rows":
        before_na = len(work_df)
        work_df = work_df.dropna().reset_index(drop=True)
        steps_applied.append(f"dropna_rows(removed={before_na - len(work_df)})")
    elif strategy == "median_mode":
        leakage_warning = (
            "AVISO METODOLOGICO: La imputacion global median_mode es apta para reportes descriptivos, "
            "pero para prediccion ML usa missing_strategy='keep' y deja que evaluate_and_predict_database "
            "impute DENTRO del split train/test para evitar data leakage."
        )
        for col in work_df.columns:
            if work_df[col].isna().any():
                if pd.api.types.is_numeric_dtype(work_df[col]) and not pd.api.types.is_bool_dtype(work_df[col]):
                    work_df[col] = work_df[col].fillna(work_df[col].median())
                else:
                    modes = work_df[col].mode(dropna=True)
                    if len(modes) > 0:
                        work_df[col] = work_df[col].fillna(modes.iloc[0])
        steps_applied.append("impute_global_median_mode")
    elif strategy != "keep":
        raise ValueError(f"missing_strategy invalido '{missing_strategy}'. Usa 'keep', 'drop_rows' o 'median_mode'.")

    DATA_WORK_DIR.mkdir(parents=True, exist_ok=True)
    default_out_name = f"{src_path.stem}_cleaned.parquet"
    chosen_name = output_filename.strip() if output_filename.strip() else default_out_name
    out_path = assert_safe_write_path(DATA_WORK_DIR / chosen_name)

    if out_path.suffix.lower() == ".csv":
        work_df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        if out_path.suffix.lower() not in {".parquet", ".pq"}:
            out_path = assert_safe_write_path(out_path.with_suffix(".parquet"))
        work_df.to_parquet(out_path, index=False)

    # Verificacion cruzada independiente: releer archivo escrito con DuckDB y verificar inmutabilidad del original
    source_after_hash = compute_sha256(src_path)
    source_untouched = input_hash == source_after_hash

    conn = duckdb.connect(":memory:")
    try:
        safe_uri = out_path.as_posix().replace("'", "''")
        if out_path.suffix.lower() == ".csv":
            duck_count = int(conn.execute(f"SELECT COUNT(*) FROM read_csv_auto('{safe_uri}')").fetchall()[0][0])
        else:
            duck_count = int(conn.execute(f"SELECT COUNT(*) FROM read_parquet('{safe_uri}')").fetchall()[0][0])
    finally:
        conn.close()

    rows_after, cols_after = work_df.shape
    verified = source_untouched and (rows_after == duck_count)

    summary = {
        "verified": verified,
        "source_immutable_verified": source_untouched,
        "input_path": str(src_path),
        "input_sha256": input_hash,
        "output_path": str(out_path),
        "output_sha256": compute_sha256(out_path),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duckdb_recheck_rows": duck_count,
        "cols_before": cols_before,
        "cols_after": cols_after,
        "duplicates_removed": dups_removed,
        "remaining_missing_cells": int(work_df.isna().sum().sum()),
        "steps_applied": steps_applied,
        "leakage_warning": leakage_warning,
        "columns": list(work_df.columns),
    }
    record_audit_event("clean_and_adapt_dataset", summary, input_path=src_path, output_path=out_path)
    return summary
