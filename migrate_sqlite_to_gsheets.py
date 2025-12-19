"""
Migrate the original SQLite DB into Google Sheets.

The script will:
  - ensure worksheets exist: providers, offices, agents, products, coverage_alias, policies
  - write headers and all rows (in the exact schema expected by the app)

Rule:
  - NEVER attempts to create a worksheet if it already exists.
"""

from _future_ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound, APIError


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
    expected = TABLE_COLUMNS[table]
    out = df.copy()

    for c in expected:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[expected]
    return out


def get_or_create_worksheet(sh: gspread.Spreadsheet, title: str, nrows: int, ncols: int):
    """
    Only creates when worksheet truly does not exist.
    If API says 'already exists', fallback to open it.
    """
    # Fast check to avoid repeated calls
    existing_titles = {ws.title for ws in sh.worksheets()}
    if title in existing_titles:
        return sh.worksheet(title)

    # Missing -> create once
    try:
        return sh.add_worksheet(title=title, rows=nrows, cols=ncols)
    except APIError as e:
        msg = str(e)
        if "already exists" in msg or "A sheet with the name" in msg:
            return sh.worksheet(title)
        raise


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

            ws = get_or_create_worksheet(
                sh,
                title=t,
                nrows=max(2000, len(df) + 10),
                ncols=max(10, len(df.columns) + 5),
            )

            df2 = df.where(pd.notnull(df), "")
            values = [TABLE_COLUMNS[t]] + [list(r) for r in df2.itertuples(index=False)]

            ws.clear()
            ws.update(values)

            print(f"✅ {t}: {len(df)} filas")

    print("\nListo. Abre tu Google Sheet y verifica las pestañas.")


if _name_ == "_main_":
    main()
