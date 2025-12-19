from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import time
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


def _to_cell(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _chunked(seq: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _is_quota_429(err: Exception) -> bool:
    s = str(err)
    return ("[429]" in s) or ("Quota exceeded" in s) or ("Read requests" in s)


def _with_retry(fn, *, max_tries: int = 6, base_sleep: float = 1.2):
    """
    Retry wrapper focused on Google Sheets quota spikes (429).
    Exponential backoff: base_sleep * (2^(try-1)).
    """
    last = None
    for i in range(1, max_tries + 1):
        try:
            return fn()
        except APIError as e:
            last = e
            if not _is_quota_429(e) or i == max_tries:
                raise
            time.sleep(base_sleep * (2 ** (i - 1)))
        except Exception as e:
            last = e
            if not _is_quota_429(e) or i == max_tries:
                raise
            time.sleep(base_sleep * (2 ** (i - 1)))
    if last:
        raise last
    raise RuntimeError("Unexpected retry wrapper state")


@dataclass
class SheetConfig:
    spreadsheet_id: str
    credentials_json: dict


class SheetDB:
    """
    SAFE BY DEFAULT:
    - NUNCA crea worksheets.
    - Si una hoja no existe, levanta error claro.
    - Para import masivo: append_rows_batch() (sin reescribir toda la hoja).
    """

    def __init__(self, cfg: SheetConfig):
        self.cfg = cfg
        self._client = None
        self._ss = None
        self._ws_cache: dict[str, Any] = {}

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

        if name in self._ws_cache:
            return self._ws_cache[name]

        try:
            ws = _with_retry(lambda: self._ss.worksheet(name))
        except WorksheetNotFound as e:
            raise RuntimeError(
                f"No existe la hoja (worksheet) '{name}' en Google Sheets. "
                f"Créala manualmente con este header exacto: {TABLE_COLUMNS[name]}"
            ) from e

        self._ws_cache[name] = ws
        return ws

    # ----------------------------
    # Headers / schema
    # ----------------------------
    def ensure_header(self, name: str) -> None:
        ws = self._ws(name)
        expected = TABLE_COLUMNS[name]
        values = _with_retry(lambda: ws.get_all_values())
        if not values:
            _with_retry(lambda: ws.update([expected]))
            return
        header = [h.strip() for h in values[0]]
        if header != expected and len(values) <= 1:
            _with_retry(lambda: ws.clear())
            _with_retry(lambda: ws.update([expected]))
            return

    def read_table(self, name: str) -> pd.DataFrame:
        self.ensure_header(name)
        ws = self._ws(name)
        values = _with_retry(lambda: ws.get_all_values())
        expected = TABLE_COLUMNS[name]

        if not values:
            return pd.DataFrame(columns=expected)

        header = values[0]
        got = [h.strip() for h in header]

        if got != expected:
            rows = values[1:]
            if not rows:
                return pd.DataFrame(columns=expected)
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
        """
        Reescribe toda la hoja (bien para CRUD pequeño).
        Import masivo usa append_rows_batch().
        """
        if name not in TABLE_COLUMNS:
            raise KeyError(f"Tabla desconocida: {name}")

        self.ensure_header(name)
        ws = self._ws(name)

        out = df.copy()
        for c in TABLE_COLUMNS[name]:
            if c not in out.columns:
                out[c] = pd.NA
        out = out[TABLE_COLUMNS[name]]

        existing = _with_retry(lambda: ws.get_all_values())
        if out.empty and len(existing) > 1:
            return

        values = [TABLE_COLUMNS[name]] + [[_to_cell(v) for v in row] for row in out.itertuples(index=False)]

        _with_retry(lambda: ws.clear())
        _with_retry(lambda: ws.update(values))

    # ----------------------------
    # Fast utilities for import
    # ----------------------------
    def get_header(self, name: str) -> list[str]:
        self.ensure_header(name)
        ws = self._ws(name)
        row1 = _with_retry(lambda: ws.row_values(1))
        return [h.strip() for h in row1]

    def col_index(self, name: str, col_name: str) -> int:
        header = self.get_header(name)
        try:
            return header.index(col_name) + 1  # 1-based
        except ValueError:
            raise RuntimeError(f"Columna '{col_name}' no existe en '{name}'. Header actual: {header}")

    def get_col_values(self, name: str, col_name: str) -> list[str]:
        ws = self._ws(name)
        idx = self.col_index(name, col_name)
        col = _with_retry(lambda: ws.col_values(idx))
        return col[1:] if len(col) > 1 else []

    def max_int_in_col(self, name: str, col_name: str) -> int:
        vals = self.get_col_values(name, col_name)
        mx = 0
        for v in vals:
            try:
                n = int(str(v).strip())
                if n > mx:
                    mx = n
            except Exception:
                continue
        return mx

    def append_rows_batch(self, name: str, rows: list[list[Any]], *, chunk_size: int = 250) -> None:
        """
        Appends rows to an existing worksheet in chunks.
        - No full-table reads/writes.
        - Ideal for PDF import.
        """
        if not rows:
            return
        self.ensure_header(name)
        ws = self._ws(name)

        def _append(chunk: list[list[Any]]):
            chunk2 = [[_to_cell(x) for x in r] for r in chunk]
            return ws.append_rows(chunk2, value_input_option="RAW")

        for chunk in _chunked(rows, chunk_size):
            _with_retry(lambda c=chunk: _append(c))

    # ----------------------------
    # ID helper (kept for small CRUD)
    # ----------------------------
    def next_id(self, name: str) -> int:
        mx = self.max_int_in_col(name, "id")
        return mx + 1
