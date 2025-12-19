from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


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
    """Best-effort type coercion so comparisons work reliably."""
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

    # Optional integer columns that may contain nulls
    for c in ["office_id", "agent_id", "days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df


@dataclass
class SheetConfig:
    spreadsheet_id: str
    credentials_json: dict


class SheetDB:
    """Google Sheets-backed storage.

    Expects worksheets named exactly:
    providers, offices, agents, products, coverage_alias, policies

    Each worksheet should have a header row matching TABLE_COLUMNS.
    If header differs, we DO NOT wipe data; we map columns safely.
    """

    def __init__(self, cfg: SheetConfig):
        self.cfg = cfg
        self._client = None
        self._ss = None

    @staticmethod
    def from_streamlit_secrets(secrets: dict) -> "SheetDB":
        """Build from st.secrets.

        Required:
          - secrets["GSHEETS_SPREADSHEET_ID"]
          - secrets["gcp_service_account"] (dict)
        """
        sid = secrets.get("GSHEETS_SPREADSHEET_ID")
        sa = secrets.get("gcp_service_account")
        if not sid or not sa:
            raise RuntimeError(
                "Faltan secretos. Agrega GSHEETS_SPREADSHEET_ID y gcp_service_account en secrets."
            )
        return SheetDB(SheetConfig(spreadsheet_id=sid, credentials_json=dict(sa)))

    def _connect(self) -> None:
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
        self._connect()
        try:
            return self._ss.worksheet(name)
        except Exception:
            # Create worksheet if missing
            if name not in TABLE_COLUMNS:
                raise KeyError(f"Tabla desconocida: {name}")
            ws = self._ss.add_worksheet(title=name, rows=2000, cols=max(10, len(TABLE_COLUMNS[name]) + 5))
            ws.append_row(TABLE_COLUMNS[name])
            return ws

    def read_table(self, name: str) -> pd.DataFrame:
        if name not in TABLE_COLUMNS:
            raise KeyError(f"Tabla desconocida: {name}")

        ws = self._ws(name)
        values = ws.get_all_values()

        # Sheet completamente vacío: crea header
        if not values:
            ws.append_row(TABLE_COLUMNS[name])
            return pd.DataFrame(columns=TABLE_COLUMNS[name])

        header = values[0]
        expected = TABLE_COLUMNS[name]
        got = [h.strip() for h in header]

        # Header distinto: NO borramos data si ya hay filas
        if got != expected:
            # Si solo hay header, podemos corregir (aquí sí es seguro)
            if len(values) <= 1:
                ws.clear()
                ws.append_row(expected)
                return pd.DataFrame(columns=expected)

            # Si hay data, mapeamos columnas sin borrar
            rows = values[1:]
            df = pd.DataFrame(rows, columns=got)

            # Asegura columnas esperadas
            for col in expected:
                if col not in df.columns:
                    df[col] = ""

            df = df[expected]
        else:
            rows = values[1:]
            df = pd.DataFrame(rows, columns=expected)

        # Limpieza + tipos
        df = df.replace({"": pd.NA})
        df = _coerce_types(name, df)
        return df

    def write_table(self, name: str, df: pd.DataFrame) -> None:
        if name not in TABLE_COLUMNS:
            raise KeyError(f"Tabla desconocida: {name}")

        ws = self._ws(name)

        out = df.copy()

        # Ensure all columns exist
        for c in TABLE_COLUMNS[name]:
            if c not in out.columns:
                out[c] = pd.NA
        out = out[TABLE_COLUMNS[name]]

        # Si me intentan guardar vacío pero ya hay data, NO borro nada
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

        values = [TABLE_COLUMNS[name]]
        for row in out.itertuples(index=False):
            values.append([_to_cell(v) for v in row])

        ws.clear()
        ws.update(values)

    def next_id(self, name: str) -> int:
        df = self.read_table(name)
        if df.empty or "id" not in df.columns or df["id"].dropna().empty:
            return 1
        return int(df["id"].dropna().astype(int).max()) + 1
