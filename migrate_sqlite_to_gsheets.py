"""Migrate the original SQLite DB into Google Sheets.

Usage:
  python tools/migrate_sqlite_to_gsheets.py \
    --sqlite_path "/path/to/db.sqlite" \
    --spreadsheet_id "..." \
    --service_account_json "/path/to/service_account.json"

The script will:
  - ensure worksheets exist: providers, offices, agents, products, coverage_alias, policies
  - write headers and all rows

Note: share the Google Sheet with the service account email (Editor).
"""

from __future__ import annotations

import argparse
import json
import sqlite3

import pandas as pd

from pathlib import Path

from google.oauth2.service_account import Credentials
import gspread

TABLES = ["providers", "offices", "agents", "products", "coverage_alias", "policies"]


def read_sqlite_table(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {name}", conn)


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
            df = read_sqlite_table(conn, t)
            # Ensure worksheet
            try:
                ws = sh.worksheet(t)
            except Exception:
                ws = sh.add_worksheet(title=t, rows=2000, cols=max(10, len(df.columns) + 5))

            ws.clear()
            ws.append_row(list(df.columns))
            if not df.empty:
                # Convert NaN -> ""
                df2 = df.where(pd.notnull(df), "")
                ws.update([list(r) for r in df2.itertuples(index=False)])
            print(f"✅ {t}: {len(df)} filas")

    print("\nListo. Abre tu Google Sheet y verifica las pestañas.")


if __name__ == "__main__":
    main()
