from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

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


def _coerce_types(table: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    int_cols = {
        "providers": ["id", "active"],
        "offices": ["id"],
        "agents": ["id", "active"],
        "products": ["id", "provider_id", "days", "active"],
        "coverage_alias": ["id", "provider_id", "active"],
        "policies": ["id", "provider_id"],
    }.get(table, [])

    float_cols = {
        "agents": ["default_commission_pct"],
        "products": ["cost", "agent_profit", "price"],
        "policies": [
            "price",
            "cost",
            "agent_profit",
            "agent_commission_pct",
            "agent_commission_amount",
            "your_net_profit",
        ],
    }.get(table, [])

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["office_id", "agent_id", "days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df


@dataclass
class SheetConfig:
    spreadsheet_id: str
    credentials_json: dict


class SheetDB:
    def __init__(self, cfg: SheetConfig):
        self.cfg = cfg
        self._client = None
        self._ss = None

    @staticmethod
    def from_streamlit_secrets(secrets: dict) -> "SheetDB":
        sid = secrets.get("GSHEETS_SPREADSHEET_ID")
        sa = secrets.get("gcp_service_account")
        if not sid or not sa:
            raise RuntimeError(
                "Faltan secretos. Agrega GSHEETS_SPREADSHEET_ID y gcp_service_account en .streamlit/secrets.toml"
            )
        return SheetDB(SheetConfig(spreadsheet_id=str(sid), credentials_json=dict(sa)))

    def _connect(self):
        if self._client is not None and self._ss is not None:
            return

        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(self.cfg.credentials_json, scopes=scopes)
        self._client = gspread.authorize(creds)
        self._ss = self._client.open_by_key(self.cfg.spreadsheet_id)

    def _ws(self, name: str):
        if name not in TABLE_COLUMNS:
            raise KeyError(f"Tabla desconocida: {name}")

        self._connect()

        # ✅ SOLO crear si de verdad no existe
        try:
            return self._ss.worksheet(name)
        except WorksheetNotFound:
            try:
                ws = self._ss.add_worksheet(title=name, rows=1000, cols=len(TABLE_COLUMNS[name]) + 5)
                ws.update([TABLE_COLUMNS[name]])
                return ws
            except APIError as e:
                msg = str(e)
                # ✅ Si la API dice "ya existe", abrirla
                if "already exists" in msg or "A sheet with the name" in msg:
                    return self._ss.worksheet(name)
                raise
        except Exception:
            # ✅ Cualquier otro error NO es "missing sheet"
            raise

    def read_table(self, name: str) -> pd.DataFrame:
        ws = self._ws(name)
        values = ws.get_all_values()
        expected = TABLE_COLUMNS[name]

        if not values:
            ws.update([expected])
            return pd.DataFrame(columns=expected)

        header = values[0]
        got = [h.strip() for h in header]

        if got != expected:
            if len(values) <= 1:
                ws.clear()
                ws.update([expected])
                return pd.DataFrame(columns=expected)

            rows = values[1:]
            df = pd.DataFrame(rows, columns=got)
            for col in expected:
                if col not in df.columns:
                    df[col] = ""
            df = df[expected].replace({"": pd.NA})
            df = _coerce_types(name, df)
            return df.reset_index(drop=True)

        rows = values[1:]
        df = pd.DataFrame(rows, columns=expected).replace({"": pd.NA})
        df = _coerce_types(name, df)
        return df.reset_index(drop=True)

    def write_table(self, name: str, df: pd.DataFrame) -> None:
        if name not in TABLE_COLUMNS:
            raise KeyError(f"Tabla desconocida: {name}")

        ws = self._ws(name)

        out = df.copy()
        for c in TABLE_COLUMNS[name]:
            if c not in out.columns:
                out[c] = pd.NA
        out = out[TABLE_COLUMNS[name]]

        existing = ws.get_all_values()
        if out.empty and len(existing) > 1:
            return

        def _to_cell(x: Any) -> str:
            if x is None:
                return ""
            try:
                if pd.isna(x):
                    return ""
            except Exception:
                pass
            return str(x)

        values = [TABLE_COLUMNS[name]] + [[_to_cell(v) for v in row] for row in out.itertuples(index=False)]

        ws.clear()
        ws.update(values)

    def next_id(self, name: str) -> int:
        df = self.read_table(name)
        if df.empty or "id" not in df.columns or df["id"].dropna().empty:
            return 1
        return int(df["id"].dropna().astype(int).max()) + 1
