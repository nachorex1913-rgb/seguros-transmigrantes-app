"""
Migrate the original SQLite DB into Google Sheets.

Usage (Windows):
  py tools/migrate_sqlite_to_gsheets.py ^
    --sqlite_path "C:\\ruta\\db.sqlite" ^
    --spreadsheet_id "..." ^
    --service_account_json "C:\\ruta\\service_account.json"

Usage (Mac/Linux):
  python tools/migrate_sqlite_to_gsheets.py \
    --sqlite_path "/path/to/db.sqlite" \
    --spreadsheet_id "..." \
    --service_account_json "/path/to/service_account.json"

This script:
  - ensures worksheets exist: providers, offices, agents, products, coverage_alias, policies
  - writes a header row EXACTLY as expected by the Streamlit app
  - writes all rows (reordered to match the app schema)

Note: share the Google Sheet with the service account email (Editor).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

TABLE_COLUMNS: dict[str, list[str]] = {
    "providers": ["id", "code", "name", "currency", "active"],
    "offices": ["id", "code", "name"],
    "agents": ["id", "name", "default_commission_pct", "active"],
    "products": ["id", "provider_id", "coverage_key", "days", "cost", "agent_profit", "price", "active"],
    "coverage_alias": ["id", "provider_id", "raw_coverage_text", "normalized_coverage_key", "active"],
    "policies": [
        "id",
        "policy_code",
        "provider_id",
        "office_id",
        "agent_id",
        "sale_date",
        "client_name",
        "raw_coverage_text",
        "coverage_key",
        "days",
        "price",
        "cost",
        "agent_profit",
        "agent_commission_pct",
        "agent_commission_amount",
        "your_net_profit",
        "status",
        "import_source",
        "period_label",
        "created_at",
    ],
}

TABLES = list(TABLE_COLUMNS.keys())


def read_sqlite_table(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {name}", conn)


def normalize_df_to_schema(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to match TABLE_COLUMNS[table]. Add missing columns as blank.
    Ignore extra columns from SQLite (keeps only the expected ones)."""
    expected = TABLE_COLUMNS[table]
    out = df.copy()

    # Add missing columns
    for c in expected:
        if c not in out.columns:
            out[c] = pd.NA

    # Keep only expected columns (drop extras)
    out = out[expected]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite_path", required=True)
    ap.add_argument("--spreadsheet_id", required=True)
    ap.add_argument("--service_account_json", required=True)
    args = ap.parse_args()

    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite no existe: {sqlite_path}")

    sa_path = Path(args.service_account_json)
    if not sa_path.exists():
        raise SystemExit(f"Service account JSON no existe: {sa_path}")

    sa = json.loads(sa_path.read_text(encoding="utf-8"))

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(args.spreadsheet_id)

    with sqlite3.connect(str(sqlite_path)) as conn:
        for t in TABLES:
            df_raw = read_sqlite_table(conn, t)
            df = normalize_df_to_schema(t, df_raw)

            # Ensure worksheet (NO falla si existe)
            try:
                ws = sh.worksheet(t)
            except Exception:
                ws = sh.add_worksheet(
                    title=t,
                    rows=max(2000, len(df) + 10),
                    cols=max(10, len(df.columns) + 5),
                )

            # Build values: header + rows
            df2 = df.where(pd.notnull(df), "")
            values = [TABLE_COLUMNS[t]] + [list(r) for r in df2.itertuples(index=False)]

            # Clear and write everything starting at A1
            ws.clear()
            ws.update(values)

            print(f"✅ {t}: {len(df)} filas")

    print("\nListo. Abre tu Google Sheet y verifica las pestañas.")


if __name__ == "__main__":
    main()
