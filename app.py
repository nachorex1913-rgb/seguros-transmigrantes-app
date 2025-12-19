import re
import uuid
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import streamlit as st
import pdfplumber

from db_gsheets import SheetDB

# ======================
# CONFIGURACIÓN LOGIN PIN
# ======================
APP_PIN = "9999"  # 👈 CAMBIA ESTO POR TU PIN REAL

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def pin_login():
    st.set_page_config(page_title="Seguros Transmigrantes", layout="wide")
    st.markdown("## 🔐 Acceso protegido")
    st.markdown("Ingresa tu PIN para continuar")
    pin_input = st.text_input("PIN", type="password", placeholder="••••", max_chars=6)
    if st.button("Ingresar"):
        if pin_input == APP_PIN:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("PIN incorrecto")


def logout_button():
    if st.sidebar.button("🔒 Cerrar sesión"):
        st.session_state.authenticated = False
        st.rerun()


# ======================
# BLOQUEO DE LA APP
# ======================
if not st.session_state.authenticated:
    pin_login()
    st.stop()

# =============================
# PATHS / ASSETS
# =============================
ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

st.set_page_config(page_title="Seguros Transmigrantes", page_icon="🛡️", layout="wide")

# =============================
# DB (Google Sheets)
# =============================
@st.cache_resource
def get_db() -> SheetDB:
    return SheetDB.from_streamlit_secrets(st.secrets)


def _load_table(name: str) -> pd.DataFrame:
    # cache per-table for snappy UI; cleared on every write
    return get_db().read_table(name)


def _save_table(name: str, df: pd.DataFrame) -> None:
    get_db().write_table(name, df)
    # Clear cache so UI refreshes
    st.cache_data.clear()


# =============================
# NORMALIZATION / FINANCIALS
# =============================
def normalize_spaces(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_coverage_key(s: str) -> str:
    return normalize_spaces(s).upper()


def money(x):
    if x is None or pd.isna(x):
        return "$0.00"
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def compute_financials(agent_profit: float, commission_pct: float):
    commission_amount = round(agent_profit * (commission_pct / 100.0), 2)
    your_net = round(agent_profit - commission_amount, 2)
    return commission_amount, your_net


# =============================
# DATE UTILS (Dom–Sáb)
# =============================
def week_bounds_from_any_day(d: date):
    delta = (d.weekday() + 1) % 7
    ws = (pd.Timestamp(d) - pd.Timedelta(days=delta)).date()  # Sunday
    we = (pd.Timestamp(ws) + pd.Timedelta(days=6)).date()  # Saturday
    return ws, we


# =============================
# DB-LIKE HELPERS (Sheets-backed)
# We keep the rest of the app almost identical by routing the limited
# SQL strings used in the original version.
# =============================

def scalar(query: str, params: dict | None = None):
    df = fetch_df(query, params)
    if df.empty:
        return None
    return df.iloc[0, 0]


def fetch_df(query: str, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    q = " ".join(query.strip().split())

    # --- providers ---
    if q == "SELECT code FROM providers":
        df = _load_table("providers")
        return df[["code"]].dropna()

    if q.startswith("SELECT id, code, name, currency, active FROM providers"):
        df = _load_table("providers")
        if "WHERE active=1" in q:
            df = df[df["active"].fillna(0).astype(int) == 1]
        df = df[["id", "code", "name", "currency", "active"]].sort_values(by=["code"])
        return df.reset_index(drop=True)

    # --- agents ---
    if q.startswith("SELECT id, name, default_commission_pct, active FROM agents"):
        df = _load_table("agents")
        if "WHERE active=1" in q:
            df = df[df["active"].fillna(0).astype(int) == 1]
        df = df[["id", "name", "default_commission_pct", "active"]].sort_values(by=["name"])
        return df.reset_index(drop=True)

    if q == "SELECT default_commission_pct FROM agents WHERE id=:id":
        df = _load_table("agents")
        df = df[df["id"].astype("Int64") == int(params["id"])][["default_commission_pct"]]
        return df.reset_index(drop=True)

    # --- products ---
    if "FROM products" in q and "WHERE provider_id=:pid" in q and "JOIN" not in q:
        df = _load_table("products")
        df = df[df["provider_id"].astype("Int64") == int(params["pid"])]
        if "AND active=1" in q:
            df = df[df["active"].fillna(0).astype(int) == 1]
        df = df[["id", "coverage_key", "days", "cost", "agent_profit", "price", "active"]]
        df = df.sort_values(by=["coverage_key", "days"]).reset_index(drop=True)
        return df

    if q.startswith("SELECT id FROM products WHERE provider_id=:pid AND coverage_key=:ck AND days=:d"):
        df = _load_table("products")
        m = (
            (df["provider_id"].astype("Int64") == int(params["pid"]))
            & (df["coverage_key"].astype(str) == str(params["ck"]))
            & (df["days"].astype("Int64") == int(params["d"]))
        )
        return df.loc[m, ["id"]].reset_index(drop=True)

    # --- offices ---
    if q == "SELECT id FROM offices WHERE code=:c":
        df = _load_table("offices")
        return df[df["code"].astype(str) == str(params["c"])][["id"]].reset_index(drop=True)

    # --- coverage_alias ---
    if "FROM coverage_alias" in q and "provider_id=:pid" in q and "raw_coverage_text=:raw" in q:
        df = _load_table("coverage_alias")
        m = (
            (df["provider_id"].astype("Int64") == int(params["pid"]))
            & (df["raw_coverage_text"].astype(str) == str(params["raw"]))
            & (df["active"].fillna(0).astype(int) == 1)
        )
        return df.loc[m, ["normalized_coverage_key"]].reset_index(drop=True)

    # --- policies simple selects ---
    if q.startswith("SELECT status, price, cost, agent_profit, agent_commission_amount, coverage_key, days FROM policies"):
        df = _load_table("policies")
        a = str(params["a"])
        b = str(params["b"])
        df = df[(df["sale_date"].astype(str) >= a) & (df["sale_date"].astype(str) <= b)]
        cols = ["status", "price", "cost", "agent_profit", "agent_commission_amount", "coverage_key", "days"]
        return df[cols].reset_index(drop=True)

    if q.startswith(
        "SELECT COALESCE(a.name,'(sin gestor)') AS gestor, SUM(p.price) AS vendido FROM policies p"
    ):
        # Sales by agent for month
        pol = _load_table("policies")
        ag = _load_table("agents")
        a = str(params["a"])
        b = str(params["b"])
        pol = pol[(pol["status"].astype(str) == "ACTIVE") & (pol["sale_date"].astype(str) >= a) & (pol["sale_date"].astype(str) <= b)]
        merged = pol.merge(ag[["id", "name"]], left_on="agent_id", right_on="id", how="left", suffixes=("", "_agent"))
        merged["gestor"] = merged["name"].fillna("(sin gestor)")
        out = merged.groupby("gestor", dropna=False)["price"].sum().reset_index().rename(columns={"price": "vendido"})
        out = out.sort_values(by=["vendido"], ascending=False).reset_index(drop=True)
        return out

    if q.startswith("SELECT COUNT(*) FROM "):
        # Counts for dashboard
        table = q.replace("SELECT COUNT(*) FROM ", "").strip()
        if " WHERE " in table:
            table, where = table.split(" WHERE ", 1)
        else:
            where = None
        table = table.strip()
        df = _load_table(table)
        if where:
            if where.strip() == "status='PENDING'":
                df = df[df["status"].astype(str) == "PENDING"]
        return pd.DataFrame([[len(df)]])

    if q.startswith("SELECT id, policy_code, sale_date, client_name, raw_coverage_text, price, agent_id FROM policies"):
        pol = _load_table("policies")
        pol = pol[(pol["status"].astype(str) == "PENDING") & (pol["provider_id"].astype("Int64") == int(params["pid"]))]
        pol = pol.sort_values(by=["sale_date"], ascending=False)
        cols = ["id", "policy_code", "sale_date", "client_name", "raw_coverage_text", "price", "agent_id"]
        return pol[cols].reset_index(drop=True)

    if q == "SELECT COUNT(*) FROM policies WHERE policy_code=:pc":
        pol = _load_table("policies")
        n = int((pol["policy_code"].astype(str) == str(params["pc"])).sum())
        return pd.DataFrame([[n]])

    if "FROM products p" in q and "JOIN providers" in q:
        # Tarifas page joined view
        pid = int(params["pid"])
        prod = _load_table("products")
        prov = _load_table("providers")
        prod = prod[prod["provider_id"].astype("Int64") == pid]
        merged = prod.merge(prov[["id", "code"]], left_on="provider_id", right_on="id", how="left")
        out = pd.DataFrame(
            {
                "proveedor": merged["code"],
                "coverage_key": merged["coverage_key"],
                "days": merged["days"],
                "cost": merged["cost"],
                "agent_profit": merged["agent_profit"],
                "price": merged["price"],
                "activo": merged["active"],
            }
        )
        return out.sort_values(by=["coverage_key", "days"]).reset_index(drop=True)

    raise RuntimeError(f"Query no soportada en versión Google Sheets: {q}")


def run_sql(query: str, params: dict | None = None) -> None:
    params = params or {}
    q = " ".join(query.strip().split())

    # --- providers ---
    if q.startswith("INSERT INTO providers"):
        df = _load_table("providers")
        new_id = get_db().next_id("providers")
        row = {
            "id": new_id,
            "code": str(params["code"]).upper(),
            "name": str(params["name"]),
            "currency": str(params["currency"]),
            "active": int(params.get("active", 1)),
        }
        # uniqueness by code
        if not df.empty and (df["code"].astype(str) == row["code"]).any():
            raise RuntimeError("Provider code ya existe")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _save_table("providers", df)
        return

    if q.startswith("UPDATE providers SET"):
        df = _load_table("providers")
        pid = int(params["id"])
        mask = df["id"].astype("Int64") == pid
        if not mask.any():
            raise RuntimeError("Provider no encontrado")
        df.loc[mask, "name"] = params["name"]
        df.loc[mask, "currency"] = params["currency"]
        df.loc[mask, "active"] = int(params["active"])
        _save_table("providers", df)
        return

    # --- agents ---
    if q.startswith("INSERT INTO agents"):
        df = _load_table("agents")
        new_id = get_db().next_id("agents")
        name = str(params["n"]).strip()
        if not name:
            raise RuntimeError("Nombre vacío")
        if not df.empty and (df["name"].astype(str).str.lower() == name.lower()).any():
            raise RuntimeError("Gestor ya existe")
        row = {
            "id": new_id,
            "name": name,
            "default_commission_pct": float(params["p"]),
            "active": int(params["a"]),
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _save_table("agents", df)
        return

    # --- offices ---
    if q.startswith("INSERT INTO offices"):
        df = _load_table("offices")
        new_id = get_db().next_id("offices")
        row = {"id": new_id, "code": str(params["c"]), "name": str(params["n"])}
        if not df.empty and (df["code"].astype(str) == row["code"]).any():
            return
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _save_table("offices", df)
        return

    # --- products ---
    if q.startswith("INSERT INTO products"):
        df = _load_table("products")
        new_id = get_db().next_id("products")
        row = {
            "id": new_id,
            "provider_id": int(params["pid"]),
            "coverage_key": str(params["ck"]),
            "days": int(params["d"]),
            "cost": float(params["c"]),
            "agent_profit": float(params["ap"]),
            "price": float(params["p"]),
            "active": int(params.get("active", 1)),
        }
        # uniqueness by (provider_id, coverage_key, days)
        if not df.empty:
            m = (
                (df["provider_id"].astype("Int64") == row["provider_id"])
                & (df["coverage_key"].astype(str) == row["coverage_key"])
                & (df["days"].astype("Int64") == row["days"])
            )
            if m.any():
                raise RuntimeError("Tarifa ya existe")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _save_table("products", df)
        return

    if q.startswith("UPDATE products SET"):
        df = _load_table("products")
        pid = int(params["pid"])
        ck = str(params["ck"])
        d = int(params["d"])
        mask = (
            (df["provider_id"].astype("Int64") == pid)
            & (df["coverage_key"].astype(str) == ck)
            & (df["days"].astype("Int64") == d)
        )
        if not mask.any():
            raise RuntimeError("Tarifa no encontrada")
        df.loc[mask, "cost"] = float(params["c"])
        df.loc[mask, "agent_profit"] = float(params["ap"])
        df.loc[mask, "price"] = float(params["p"])
        df.loc[mask, "active"] = 1
        _save_table("products", df)
        return

    # --- policies ---
    if q.startswith("INSERT INTO policies"):
        df = _load_table("policies")
        new_id = get_db().next_id("policies")
        # minimal created_at
        created_at = datetime.now().isoformat(timespec="seconds")
        row = {
            "id": new_id,
            "policy_code": str(params["pc"]),
            "provider_id": int(params["pid"]),
            "office_id": params.get("oid"),
            "agent_id": int(params["aid"]) if params.get("aid") is not None else pd.NA,
            "sale_date": str(params["sd"]),
            "client_name": params.get("cn"),
            "raw_coverage_text": str(params["raw"]),
            "coverage_key": params.get("ck"),
            "days": params.get("days"),
            "price": params.get("price"),
            "cost": params.get("cost"),
            "agent_profit": params.get("ap"),
            "agent_commission_pct": params.get("pct"),
            "agent_commission_amount": params.get("cam"),
            "your_net_profit": params.get("net"),
            "status": str(params.get("status", "ACTIVE")),
            "import_source": "PDF" if "'PDF'" in q or params.get("import_source") == "PDF" else "MANUAL",
            "period_label": params.get("period"),
            "created_at": created_at,
        }
        if not df.empty and (df["policy_code"].astype(str) == row["policy_code"]).any():
            raise RuntimeError("Policy code ya existe")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _save_table("policies", df)
        return

    if q.startswith("UPDATE policies SET"):
        df = _load_table("policies")
        rid = int(params["id"])
        mask = df["id"].astype("Int64") == rid
        if not mask.any():
            raise RuntimeError("Póliza no encontrada")
        # Update known fields used in the app
        for col, key in [
            ("coverage_key", "ck"),
            ("days", "d"),
            ("cost", "c"),
            ("agent_profit", "ap"),
            ("price", "p"),
            ("agent_commission_pct", "pct"),
            ("agent_commission_amount", "cam"),
            ("your_net_profit", "net"),
        ]:
            if key in params:
                df.loc[mask, col] = params[key]
        # status forced to ACTIVE by query
        df.loc[mask, "status"] = "ACTIVE"
        _save_table("policies", df)
        return

    # --- coverage_alias ---
    if q.startswith("INSERT OR IGNORE INTO coverage_alias"):
        df = _load_table("coverage_alias")
        pid = int(params["pid"])
        raw = str(params["raw"])
        ck = str(params["ck"])
        if not df.empty:
            m = (df["provider_id"].astype("Int64") == pid) & (df["raw_coverage_text"].astype(str) == raw)
            if m.any():
                return
        new_id = get_db().next_id("coverage_alias")
        row = {
            "id": new_id,
            "provider_id": pid,
            "raw_coverage_text": raw,
            "normalized_coverage_key": ck,
            "active": 1,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _save_table("coverage_alias", df)
        return

    raise RuntimeError(f"run_sql no soportado en versión Google Sheets: {q}")


# =============================
# PROVIDERS (SEED)
# =============================
def ensure_seed_providers():
    existing = fetch_df("SELECT code FROM providers")
    existing_codes = set(existing["code"].tolist()) if not existing.empty else set()

    seeds = [
        {"code": "SPEED_USA", "name": "Speed (USA)", "currency": "USD"},
        {"code": "QUALITAS_MX", "name": "Qualitas (MX)", "currency": "MXN"},
    ]
    for s in seeds:
        if s["code"] not in existing_codes:
            run_sql(
                """
                INSERT INTO providers (code, name, currency, active)
                VALUES (:code, :name, :currency, 1)
                """,
                s,
            )


# =============================
# FETCHERS
# =============================
def get_providers(active_only=False):
    q = "SELECT id, code, name, currency, active FROM providers"
    if active_only:
        q += " WHERE active=1"
    q += " ORDER BY code"
    return fetch_df(q)


def get_agents(active_only=True):
    q = "SELECT id, name, default_commission_pct, active FROM agents"
    if active_only:
        q += " WHERE active=1"
    q += " ORDER BY name"
    return fetch_df(q)


def get_products(provider_id: int, active_only=True):
    q = """
    SELECT id, coverage_key, days, cost, agent_profit, price, active
    FROM products
    WHERE provider_id=:pid
    """
    if active_only:
        q += " AND active=1"
    q += " ORDER BY coverage_key, days"
    return fetch_df(q, {"pid": provider_id})


def upsert_office(code: str):
    code = normalize_coverage_key(code)
    if not code:
        return None
    existing = fetch_df("SELECT id FROM offices WHERE code=:c", {"c": code})
    if existing.empty:
        run_sql("INSERT INTO offices (code, name) VALUES (:c, :n)", {"c": code, "n": code})
    return int(scalar("SELECT id FROM offices WHERE code=:c", {"c": code}))


# =============================
# PDF PARSING (SPEED WEEKLY)
# =============================
MONTHS_MAP = {
    "JAN": 1,
    "JANUARY": 1,
    "FEB": 2,
    "FEBRUARY": 2,
    "MAR": 3,
    "MARCH": 3,
    "APR": 4,
    "APRIL": 4,
    "MAY": 5,
    "JUN": 6,
    "JUNE": 6,
    "JUL": 7,
    "JULY": 7,
    "AUG": 8,
    "AUGUST": 8,
    "SEP": 9,
    "SEPT": 9,
    "SEPTEMBER": 9,
    "OCT": 10,
    "OCTOBER": 10,
    "NOV": 11,
    "NOVEMBER": 11,
    "DEC": 12,
    "DECEMBER": 12,
    "ENERO": 1,
    "FEBRERO": 2,
    "MARZO": 3,
    "ABRIL": 4,
    "MAYO": 5,
    "JUNIO": 6,
    "JULIO": 7,
    "AGOSTO": 8,
    "SEPTIEMBRE": 9,
    "OCTUBRE": 10,
    "NOVIEMBRE": 11,
    "DICIEMBRE": 12,
}


def parse_sale_date(s: str) -> str:
    s = normalize_spaces(s)
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
    if not m:
        return s
    mon_txt, day, year = m.group(1).upper(), int(m.group(2)), int(m.group(3))
    mon = MONTHS_MAP.get(mon_txt, None)
    if not mon:
        return s
    return date(year, mon, day).isoformat()


def extract_days_and_key(coverage: str):
    cov = normalize_spaces(coverage).upper()
    m = re.search(r"(\d+)\s+DIAS?\b", cov)
    days = int(m.group(1)) if m else None
    if days is not None:
        key = re.sub(r"\s+\d+\s+DIAS?\b", "", cov).strip()
    else:
        key = cov.strip()
    return key, days, cov


ROW_RE = re.compile(
    r"^(F[V|T]\d+)\s+([A-Za-z]{3}\s+\d{1,2},\s+\d{4})\s+([A-Z0-9_]+)\s+(.+?)\s+([A-Z].+?)\s+\$(\d+(?:\.\d+)?)\s+(.+)$"
)


def read_speed_weekly_pdf(file) -> tuple[str | None, pd.DataFrame]:
    with pdfplumber.open(file) as pdf:
        text_all = []
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_all.append(t)

    full = "\n".join(text_all)
    lines = [normalize_spaces(x) for x in full.splitlines() if normalize_spaces(x)]

    period = None
    if lines and re.match(r"^\d{1,2}\s*-\s*\d{1,2}\s+[A-ZÁÉÍÓÚÑ]+", lines[0].upper()):
        period = lines[0].upper()

    rows = []
    for ln in lines:
        up = ln.upper()
        if up.startswith("ID DATE USER") or up.startswith("CHECK/") or up == "1 DE 1":
            continue
        m = ROW_RE.match(ln)
        if not m:
            continue

        policy_code = m.group(1)
        sale_date = parse_sale_date(m.group(2))
        office_code = m.group(3)
        pay = m.group(4)
        client_name = m.group(5)
        total = float(m.group(6))
        coverage = m.group(7)

        cov_key, days, raw_cov = extract_days_and_key(coverage)

        rows.append(
            {
                "policy_code": policy_code,
                "sale_date": sale_date,
                "office_code": office_code,
                "pay": pay,
                "client_name": client_name,
                "price": total,
                "raw_coverage_text": normalize_coverage_key(raw_cov),
                "coverage_key_guess": normalize_coverage_key(cov_key),
                "days_guess": days,
            }
        )

    return period, pd.DataFrame(rows)


# =============================
# UI COMPONENTS
# =============================

def render_logo():
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)


def kpi_cards(items):
    cards = ""
    for t, v, s in items:
        cards += f"""
        <div class="kpi-card">
          <div class="kpi-title">{t}</div>
          <div class="kpi-value">{v}</div>
          <div class="kpi-sub">{s}</div>
        </div>
        """
    st.markdown(
        f"""
        <style>
        .kpi-grid {{
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 12px;
          margin-top: 8px;
          margin-bottom: 6px;
        }}
        .kpi-card {{
          padding: 14px 14px;
          border-radius: 16px;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.10);
          box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }}
        .kpi-title {{
          font-size: 12px;
          opacity: 0.85;
          letter-spacing: .3px;
          text-transform: uppercase;
        }}
        .kpi-value {{
          font-size: 26px;
          font-weight: 800;
          margin-top: 6px;
        }}
        .kpi-sub {{
          font-size: 12px;
          opacity: 0.8;
          margin-top: 4px;
        }}
        @media (max-width: 1100px) {{
          .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        </style>
        <div class="kpi-grid">{cards}</div>
        """,
        unsafe_allow_html=True,
    )


def home_tiles(go):
    st.markdown("### Accesos rápidos")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("📊 Dashboard", use_container_width=True):
        go("Dashboard")
        st.rerun()
    if c2.button("🧾 Registrar póliza", use_container_width=True):
        go("Registrar póliza")
        st.rerun()
    if c3.button("📥 Importar PDF", use_container_width=True):
        go("Importar PDF")
        st.rerun()
    if c4.button("🛠️ Pendientes", use_container_width=True):
        go("Pendientes")
        st.rerun()

    c5, c6, c7, c8 = st.columns(4)
    if c5.button("💲 Tarifas", use_container_width=True):
        go("Tarifas")
        st.rerun()
    if c6.button("👥 Gestores", use_container_width=True):
        go("Gestores")
        st.rerun()
    if c7.button("💼 Proveedores", use_container_width=True):
        go("Proveedores")
        st.rerun()
    if c8.button("🏠 Inicio", use_container_width=True):
        go("Inicio")
        st.rerun()


def chart_top_products(df_active: pd.DataFrame, title: str):
    st.markdown(f"### {title}")
    if df_active.empty:
        st.info("Sin pólizas ACTIVE en este periodo.")
        return
    tmp = df_active.copy()
    tmp["producto"] = tmp["coverage_key"].fillna("") + " - " + tmp["days"].fillna(0).astype(int).astype(str) + " días"
    top = tmp.groupby("producto").size().sort_values(ascending=False).head(10)
    st.bar_chart(top)


def chart_agents_month(month_start: date, month_end: date):
    st.markdown("### Ventas por gestor (mes en curso)")
    df = fetch_df(
        """
        SELECT COALESCE(a.name,'(sin gestor)') AS gestor, SUM(p.price) AS vendido
        FROM policies p
        LEFT JOIN agents a ON a.id = p.agent_id
        WHERE p.status='ACTIVE'
          AND p.sale_date BETWEEN :a AND :b
        GROUP BY COALESCE(a.name,'(sin gestor)')
        ORDER BY vendido DESC
        """,
        {"a": month_start.isoformat(), "b": month_end.isoformat()},
    )
    if df.empty:
        st.info("Sin ventas ACTIVE en el mes.")
        return
    st.bar_chart(df.set_index("gestor")["vendido"])


# =============================
# NAVIGATION
# =============================
PAGES = [
    "Inicio",
    "Dashboard",
    "Registrar póliza",
    "Importar PDF",
    "Pendientes",
    "Tarifas",
    "Gestores",
    "Proveedores",
]

if "page" not in st.session_state:
    st.session_state.page = "Inicio"


def go(page_name: str):
    st.session_state.page = page_name


# =============================
# SIDEBAR
# =============================
render_logo()

with st.sidebar:
    st.subheader("Menú")
    choice = st.radio("Navegar", PAGES, index=PAGES.index(st.session_state.page))
    st.session_state.page = choice
logout_button()

# =============================
# HEADER
# =============================
st.title("Seguros Transmigrantes")
page = st.session_state.page

# =============================
# PAGES
# =============================
if page == "Inicio":
    home_tiles(go)
    st.stop()

elif page == "Dashboard":
    today = date.today()

    prov = int(scalar("SELECT COUNT(*) FROM providers") or 0)
    ag = int(scalar("SELECT COUNT(*) FROM agents") or 0)
    prod = int(scalar("SELECT COUNT(*) FROM products") or 0)
    pol = int(scalar("SELECT COUNT(*) FROM policies") or 0)
    pend = int(scalar("SELECT COUNT(*) FROM policies WHERE status='PENDING'") or 0)

    st.markdown("### Resumen del sistema")
    kpi_cards(
        [
            ("Proveedores", f"{prov}", "Registrados"),
            ("Gestores", f"{ag}", "Registrados"),
            ("Tarifas", f"{prod}", "Cargadas"),
            ("Pólizas", f"{pol}", "Total"),
            ("Pendientes", f"{pend}", "Por resolver"),
        ]
    )

    st.divider()

    week_start, week_end = week_bounds_from_any_day(today)

    df_week = fetch_df(
        """
        SELECT status, price, cost, agent_profit, agent_commission_amount, coverage_key, days
        FROM policies
        WHERE sale_date BETWEEN :a AND :b
        """,
        {"a": week_start.isoformat(), "b": week_end.isoformat()},
    )
    wA = df_week[df_week["status"] == "ACTIVE"].copy()

    st.markdown("### Semana (actual)")
    st.caption(f"Semana (Dom–Sáb): {week_start} → {week_end}")
    kpi_cards(
        [
            ("Pólizas semana", f"{len(wA)}", "ACTIVE"),
            ("Vendido semana", money(wA["price"].sum() if not wA.empty else 0), "PRICE"),
            ("Costo semana", money(wA["cost"].sum() if not wA.empty else 0), "Proveedor"),
            ("Agent semana", money(wA["agent_profit"].sum() if not wA.empty else 0), "Utilidad bruta"),
            (
                "Gestor semana",
                money(wA["agent_commission_amount"].sum() if not wA.empty else 0),
                "Comisión",
            ),
        ]
    )
    chart_top_products(wA, "Pólizas más vendidas (semana)")

    st.divider()

    month_start = today.replace(day=1)
    month_end = today

    df_month = fetch_df(
        """
        SELECT status, price, cost, agent_profit, agent_commission_amount, coverage_key, days
        FROM policies
        WHERE sale_date BETWEEN :a AND :b
        """,
        {"a": month_start.isoformat(), "b": month_end.isoformat()},
    )
    mA = df_month[df_month["status"] == "ACTIVE"].copy()

    st.markdown("### Mes (actual)")
    st.caption(f"Mes: {month_start} → {month_end}")
    kpi_cards(
        [
            ("Pólizas mes", f"{len(mA)}", "ACTIVE"),
            ("Vendido mes", money(mA["price"].sum() if not mA.empty else 0), "PRICE"),
            ("Costo mes", money(mA["cost"].sum() if not mA.empty else 0), "Proveedor"),
            ("Agent mes", money(mA["agent_profit"].sum() if not mA.empty else 0), "Utilidad bruta"),
            (
                "Gestor mes",
                money(mA["agent_commission_amount"].sum() if not mA.empty else 0),
                "Comisión",
            ),
        ]
    )
    chart_top_products(mA, "Pólizas más vendidas (mes)")

    st.divider()

    if "period_range" not in st.session_state:
        st.session_state.period_range = ((pd.Timestamp(today) - pd.Timedelta(days=30)).date(), today)

    st.markdown("### Periodo (personalizado)")
    period = st.date_input("Selecciona el periodo (Desde → Hasta)", value=st.session_state.period_range, key="period_range")

    if isinstance(period, (date, datetime)):
        st.warning("Selecciona un rango completo (Desde y Hasta).")
        st.stop()

    p_start, p_end = period
    if p_end < p_start:
        st.error("El 'Hasta' no puede ser menor que el 'Desde'.")
        st.stop()

    st.caption(f"Periodo seleccionado: {p_start} → {p_end}")

    df_period = fetch_df(
        """
        SELECT status, price, cost, agent_profit, agent_commission_amount, coverage_key, days
        FROM policies
        WHERE sale_date BETWEEN :a AND :b
        """,
        {"a": p_start.isoformat(), "b": p_end.isoformat()},
    )
    pA = df_period[df_period["status"] == "ACTIVE"].copy()

    kpi_cards(
        [
            ("Pólizas periodo", f"{len(pA)}", "ACTIVE"),
            ("Vendido periodo", money(pA["price"].sum() if not pA.empty else 0), "PRICE"),
            ("Costo periodo", money(pA["cost"].sum() if not pA.empty else 0), "Proveedor"),
            ("Agent periodo", money(pA["agent_profit"].sum() if not pA.empty else 0), "Utilidad bruta"),
            (
                "Gestor periodo",
                money(pA["agent_commission_amount"].sum() if not pA.empty else 0),
                "Comisión",
            ),
        ]
    )
    chart_top_products(pA, "Pólizas más vendidas (periodo)")

    st.divider()

    chart_agents_month(month_start, month_end)

elif page == "Proveedores":
    st.subheader("Proveedores")

    df = get_providers(active_only=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("✅ Crear proveedores base (Speed / Qualitas)", use_container_width=True):
        ensure_seed_providers()
        st.success("Listo. Proveedores base creados (si no existían).")
        st.rerun()

    st.markdown("### Agregar proveedor")
    with st.form("add_provider"):
        code = st.text_input("Code (ej: SPEED_USA)", value="").strip().upper()
        name = st.text_input("Nombre", value="")
        currency = st.selectbox("Moneda", ["USD", "MXN"], index=0)
        active = st.checkbox("Activo", value=True)
        submitted = st.form_submit_button("Guardar")

    if submitted:
        if not code or not name:
            st.error("Code y Nombre son obligatorios.")
        else:
            try:
                run_sql(
                    """
                    INSERT INTO providers (code, name, currency, active)
                    VALUES (:code, :name, :currency, :active)
                    """,
                    {"code": code, "name": name, "currency": currency, "active": 1 if active else 0},
                )
                st.success("Proveedor agregado.")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo guardar. Error: {e}")

    st.markdown("### Editar / desactivar")
    if not df.empty:
        sel_id = st.selectbox(
            "Selecciona proveedor",
            df["id"].tolist(),
            format_func=lambda pid: f"{df.loc[df['id']==pid,'code'].iloc[0]} — {df.loc[df['id']==pid,'name'].iloc[0]}",
        )
        row = df[df["id"] == sel_id].iloc[0]
        with st.form("edit_provider"):
            new_name = st.text_input("Nombre", value=row["name"])
            new_currency = st.selectbox("Moneda", ["USD", "MXN"], index=0 if row["currency"] == "USD" else 1)
            new_active = st.checkbox("Activo", value=bool(row["active"]))
            save = st.form_submit_button("Actualizar")
        if save:
            try:
                run_sql(
                    """
                    UPDATE providers
                    SET name=:name, currency=:currency, active=:active
                    WHERE id=:id
                    """,
                    {"id": int(sel_id), "name": new_name, "currency": new_currency, "active": 1 if new_active else 0},
                )
                st.success("Proveedor actualizado.")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo actualizar. Error: {e}")

elif page == "Gestores":
    st.subheader("Gestores")
    df = fetch_df("SELECT id, name, default_commission_pct, active FROM agents ORDER BY name")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Agregar gestor")
    with st.form("add_agent"):
        name = st.text_input("Nombre", value="").strip()
        pct = st.number_input("% comisión default", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
        active = st.checkbox("Activo", value=True)
        submitted = st.form_submit_button("Guardar")

    if submitted:
        if not name:
            st.error("El nombre es obligatorio.")
        else:
            try:
                run_sql(
                    "INSERT INTO agents (name, default_commission_pct, active) VALUES (:n,:p,:a)",
                    {"n": name, "p": float(pct), "a": 1 if active else 0},
                )
                st.success("Gestor agregado.")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo guardar. Error: {e}")

elif page == "Tarifas":
    st.subheader("Tarifas")

    providers = get_providers(active_only=True)
    if providers.empty:
        st.warning("No hay proveedores activos.")
        st.stop()

    pid = st.selectbox(
        "Proveedor",
        providers["id"].tolist(),
        format_func=lambda x: f"{providers.loc[providers['id']==x,'code'].iloc[0]} — {providers.loc[providers['id']==x,'name'].iloc[0]}",
    )

    df = fetch_df(
        """
        SELECT
          pr.code AS proveedor,
          p.coverage_key,
          p.days,
          p.cost,
          p.agent_profit,
          p.price,
          p.active AS activo
        FROM products p
        JOIN providers pr ON pr.id = p.provider_id
        WHERE p.provider_id=:pid
        ORDER BY p.coverage_key, p.days
        """,
        {"pid": int(pid)},
    )

    if not df.empty:
        show = df.copy()
        show["cost"] = show["cost"].map(money)
        show["agent_profit"] = show["agent_profit"].map(money)
        show["price"] = show["price"].map(money)
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No hay tarifas aún para este proveedor.")

    st.divider()
    st.markdown("### Carga masiva (pegar CSV)")
    st.caption("Formato: coverage_key,days,cost,agent_profit,price (sin $)")
    csv_text = st.text_area("Pega aquí tu CSV", height=220)

    if st.button("Importar CSV", use_container_width=True):
        if not csv_text.strip():
            st.warning("Pega el CSV primero.")
        else:
            try:
                df_in = pd.read_csv(pd.io.common.StringIO(csv_text))
                required = {"coverage_key", "days", "cost", "agent_profit", "price"}
                if not required.issubset(set(df_in.columns)):
                    st.error(f"Faltan columnas. Requiere: {sorted(required)}")
                else:
                    df_in["coverage_key"] = df_in["coverage_key"].astype(str).map(normalize_coverage_key)
                    df_in["days"] = df_in["days"].astype(int)
                    df_in["cost"] = df_in["cost"].astype(float)
                    df_in["agent_profit"] = df_in["agent_profit"].astype(float)
                    df_in["price"] = df_in["price"].astype(float)

                    inserted, updated = 0, 0
                    for _, r in df_in.iterrows():
                        exists = fetch_df(
                            "SELECT id FROM products WHERE provider_id=:pid AND coverage_key=:ck AND days=:d",
                            {"pid": int(pid), "ck": r["coverage_key"], "d": int(r["days"])},
                        )
                        if exists.empty:
                            run_sql(
                                """
                                INSERT INTO products (provider_id, coverage_key, days, cost, agent_profit, price, active)
                                VALUES (:pid,:ck,:d,:c,:ap,:p,1)
                                """,
                                {
                                    "pid": int(pid),
                                    "ck": r["coverage_key"],
                                    "d": int(r["days"]),
                                    "c": float(r["cost"]),
                                    "ap": float(r["agent_profit"]),
                                    "p": float(r["price"]),
                                },
                            )
                            inserted += 1
                        else:
                            run_sql(
                                """
                                UPDATE products
                                SET cost=:c, agent_profit=:ap, price=:p, active=1
                                WHERE provider_id=:pid AND coverage_key=:ck AND days=:d
                                """,
                                {
                                    "pid": int(pid),
                                    "ck": r["coverage_key"],
                                    "d": int(r["days"]),
                                    "c": float(r["cost"]),
                                    "ap": float(r["agent_profit"]),
                                    "p": float(r["price"]),
                                },
                            )
                            updated += 1

                    st.success(f"Listo. Insertadas: {inserted} | Actualizadas: {updated}")
                    st.rerun()
            except Exception as e:
                st.error(f"No se pudo importar. Error: {e}")

elif page == "Registrar póliza":
    st.subheader("Registrar póliza")

    providers = get_providers(active_only=True)
    agents = get_agents(active_only=True)

    if providers.empty:
        st.warning("Primero crea proveedores.")
        st.stop()
    if agents.empty:
        st.warning("Primero crea al menos 1 gestor.")
        st.stop()

    colA, colB, colC = st.columns(3)
    pid = colA.selectbox(
        "Proveedor",
        providers["id"].tolist(),
        format_func=lambda x: providers.loc[providers["id"] == x, "code"].iloc[0],
    )

    agent_id = colB.selectbox(
        "Gestor",
        agents["id"].tolist(),
        format_func=lambda x: agents.loc[agents["id"] == x, "name"].iloc[0],
    )
    default_pct = float(agents.loc[agents["id"] == agent_id, "default_commission_pct"].iloc[0])
    commission_pct = colC.number_input("% comisión", min_value=0.0, max_value=100.0, value=default_pct, step=0.5)

    products = get_products(int(pid), active_only=True)
    if products.empty:
        st.warning("Este proveedor no tiene tarifas aún. Carga Tarifas primero.")
        st.stop()

    office_code = st.text_input("Office/User", value="PRESIDIO")
    sale_date = st.date_input("Fecha de venta", value=date.today())
    client_name = st.text_input("Nombre del cliente", value="")

    product_choice = st.selectbox(
        "Cobertura",
        products["id"].tolist(),
        format_func=lambda x: f"{products.loc[products['id']==x,'coverage_key'].iloc[0]} — {int(products.loc[products['id']==x,'days'].iloc[0])} días",
    )
    row = products[products["id"] == product_choice].iloc[0]
    st.caption(f"COST: {money(row['cost'])} | AGENT: {money(row['agent_profit'])} | PRICE: {money(row['price'])}")

    policy_code = st.text_input("Policy Code (opcional)", value="").strip()

    if st.button("Guardar", use_container_width=True):
        try:
            office_id = upsert_office(office_code)
            if not policy_code:
                policy_code = "MAN-" + datetime.now().strftime("%Y%m%d-%H%M%S-") + str(uuid.uuid4())[:6].upper()

            ap = float(row["agent_profit"])
            cam, net = compute_financials(ap, float(commission_pct))

            run_sql(
                """
                INSERT INTO policies (
                  policy_code, provider_id, office_id, agent_id,
                  sale_date, client_name,
                  raw_coverage_text, coverage_key, days,
                  price, cost, agent_profit,
                  agent_commission_pct, agent_commission_amount, your_net_profit,
                  status, import_source, period_label
                )
                VALUES (
                  :pc, :pid, :oid, :aid,
                  :sd, :cn,
                  :raw, :ck, :days,
                  :price, :cost, :ap,
                  :pct, :cam, :net,
                  'ACTIVE', 'MANUAL', NULL
                )
                """,
                {
                    "pc": policy_code,
                    "pid": int(pid),
                    "oid": int(office_id) if office_id else None,
                    "aid": int(agent_id),
                    "sd": sale_date.isoformat(),
                    "cn": client_name or None,
                    "raw": f"{row['coverage_key']} {int(row['days'])} DIAS",
                    "ck": row["coverage_key"],
                    "days": int(row["days"]),
                    "price": float(row["price"]),
                    "cost": float(row["cost"]),
                    "ap": float(row["agent_profit"]),
                    "pct": float(commission_pct),
                    "cam": float(cam),
                    "net": float(net),
                    "status": "ACTIVE",
                },
            )
            st.success(f"Póliza guardada: {policy_code}")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo guardar. Error: {e}")

elif page == "Importar PDF":
    st.subheader("Importar PDF (Speed)")

    providers = get_providers(active_only=True)
    speed = providers[providers["code"] == "SPEED_USA"]
    if speed.empty:
        st.warning("No existe SPEED_USA. Créalo en Proveedores.")
        st.stop()
    speed_id = int(speed.iloc[0]["id"])

    agents = get_agents(active_only=True)
    if agents.empty:
        st.warning("Primero crea al menos 1 gestor.")
        st.stop()

    col1, col2 = st.columns(2)
    agent_id = col1.selectbox(
        "Gestor (para asignar pólizas importadas)",
        agents["id"].tolist(),
        format_func=lambda x: agents.loc[agents["id"] == x, "name"].iloc[0],
    )
    default_pct = float(agents.loc[agents["id"] == agent_id, "default_commission_pct"].iloc[0])
    commission_pct = col2.number_input(
        "% comisión para esta importación", min_value=0.0, max_value=100.0, value=default_pct, step=0.5
    )

    up = st.file_uploader("Sube el PDF semanal", type=["pdf"])
    if up is not None:
        try:
            period, df = read_speed_weekly_pdf(up)
            if df.empty:
                st.error("No encontré filas de pólizas en el PDF (formato inesperado).")
                st.stop()

            st.write("Vista previa:")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Periodo detectado: {period or '(no detectado)'}")

            if st.button("Importar", use_container_width=True):
                inserted, skipped = 0, 0
                products = get_products(speed_id, active_only=True)

                for _, r in df.iterrows():
                    exists = scalar("SELECT COUNT(*) FROM policies WHERE policy_code=:pc", {"pc": r["policy_code"]}) or 0
                    if int(exists) > 0:
                        skipped += 1
                        continue

                    office_id = upsert_office(r["office_code"])
                    raw_cov = normalize_coverage_key(r["raw_coverage_text"])

                    alias = fetch_df(
                        """
                        SELECT normalized_coverage_key
                        FROM coverage_alias
                        WHERE provider_id=:pid AND raw_coverage_text=:raw AND active=1
                        """,
                        {"pid": speed_id, "raw": raw_cov},
                    )
                    cov_key = alias.iloc[0]["normalized_coverage_key"] if not alias.empty else r["coverage_key_guess"]
                    days = int(r["days_guess"]) if pd.notna(r["days_guess"]) else None

                    match = pd.DataFrame()
                    if days is not None:
                        match = products[(products["coverage_key"] == cov_key) & (products["days"] == days)]

                    if not match.empty:
                        mrow = match.iloc[0]
                        cost = float(mrow["cost"])
                        ap = float(mrow["agent_profit"])
                        price = float(mrow["price"])
                        cam, net = compute_financials(ap, float(commission_pct))
                        status = "ACTIVE"
                        ck_store = cov_key
                        days_store = days
                        pct_store = float(commission_pct)
                    else:
                        cost = None
                        ap = None
                        price = float(r["price"]) if pd.notna(r["price"]) else None
                        cam, net = None, None
                        status = "PENDING"
                        ck_store = None
                        days_store = days
                        pct_store = None

                    run_sql(
                        """
                        INSERT INTO policies (
                          policy_code, provider_id, office_id, agent_id,
                          sale_date, client_name,
                          raw_coverage_text, coverage_key, days,
                          price, cost, agent_profit,
                          agent_commission_pct, agent_commission_amount, your_net_profit,
                          status, import_source, period_label
                        )
                        VALUES (
                          :pc, :pid, :oid, :aid,
                          :sd, :cn,
                          :raw, :ck, :days,
                          :price, :cost, :ap,
                          :pct, :cam, :net,
                          :status, 'PDF', :period
                        )
                        """,
                        {
                            "pc": r["policy_code"],
                            "pid": speed_id,
                            "oid": int(office_id) if office_id else None,
                            "aid": int(agent_id),
                            "sd": r["sale_date"],
                            "cn": r["client_name"],
                            "raw": raw_cov,
                            "ck": ck_store,
                            "days": int(days_store) if days_store is not None else None,
                            "price": price,
                            "cost": cost,
                            "ap": ap,
                            "pct": pct_store,
                            "cam": cam,
                            "net": net,
                            "status": status,
                            "period": period,
                        },
                    )
                    inserted += 1

                st.success(f"Importación completa. Insertadas: {inserted} | Duplicadas omitidas: {skipped}")
                st.rerun()

        except Exception as e:
            st.error(f"No pude leer/importar el PDF. Error: {e}")

elif page == "Pendientes":
    st.subheader("Pendientes")

    providers = get_providers(active_only=True)
    if providers.empty:
        st.warning("No hay proveedores activos.")
        st.stop()

    pid = st.selectbox(
        "Proveedor",
        providers["id"].tolist(),
        format_func=lambda x: providers.loc[providers["id"] == x, "code"].iloc[0],
    )

    products = get_products(int(pid), active_only=True)
    if products.empty:
        st.warning("Este proveedor no tiene tarifas. Carga Tarifas primero.")
        st.stop()

    pend = fetch_df(
        """
        SELECT id, policy_code, sale_date, client_name, raw_coverage_text, price, agent_id
        FROM policies
        WHERE status='PENDING' AND provider_id=:pid
        ORDER BY sale_date DESC
        """,
        {"pid": int(pid)},
    )

    if pend.empty:
        st.success("No hay pendientes.")
        st.stop()

    if st.button("⚡ Auto-resolver lo que ya matchee tarifa", use_container_width=True):
        fixed = 0
        for _, rr in pend.iterrows():
            rid = int(rr["id"])
            raw = rr["raw_coverage_text"]
            gk, gd, _ = extract_days_and_key(raw)
            gk = normalize_coverage_key(gk)
            if gd is None:
                continue

            m = products[(products["coverage_key"] == gk) & (products["days"] == int(gd))]
            if m.empty:
                continue

            mrow = m.iloc[0]
            cost = float(mrow["cost"])
            ap = float(mrow["agent_profit"])
            price = float(mrow["price"])

            pct = 20.0
            if rr["agent_id"] is not None and pd.notna(rr["agent_id"]):
                a = fetch_df("SELECT default_commission_pct FROM agents WHERE id=:id", {"id": int(rr["agent_id"])})
                if not a.empty:
                    pct = float(a.iloc[0]["default_commission_pct"])

            cam, net = compute_financials(ap, pct)

            run_sql(
                """
                UPDATE policies
                SET coverage_key=:ck, days=:d,
                    cost=:c, agent_profit=:ap, price=:p,
                    agent_commission_pct=:pct, agent_commission_amount=:cam, your_net_profit=:net,
                    status='ACTIVE'
                WHERE id=:id
                """,
                {"ck": gk, "d": int(gd), "c": cost, "ap": ap, "p": price, "pct": pct, "cam": cam, "net": net, "id": rid},
            )

            run_sql(
                """
                INSERT OR IGNORE INTO coverage_alias (provider_id, raw_coverage_text, normalized_coverage_key, active)
                VALUES (:pid, :raw, :ck, 1)
                """,
                {"pid": int(pid), "raw": normalize_coverage_key(raw), "ck": gk},
            )

            fixed += 1

        st.success(f"Auto-resueltas: {fixed}")
        st.rerun()

    st.markdown("### Lista de pendientes")
    show = pend.copy()
    show["price"] = show["price"].map(money)
    st.dataframe(show.drop(columns=["agent_id"]), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Resolver una póliza (con sugerencia automática)")

    row_id = st.selectbox(
        "Selecciona",
        pend["id"].tolist(),
        format_func=lambda x: f"{pend.loc[pend['id']==x,'policy_code'].iloc[0]} — {pend.loc[pend['id']==x,'raw_coverage_text'].iloc[0]}",
    )
    raw_text = pend.loc[pend["id"] == row_id, "raw_coverage_text"].iloc[0]

    guess_key, guess_days, _ = extract_days_and_key(raw_text)
    guess_key = normalize_coverage_key(guess_key)

    ids = products["id"].tolist()
    default_index = 0
    if guess_days is not None:
        match_ids = products[(products["coverage_key"] == guess_key) & (products["days"] == int(guess_days))]["id"].tolist()
        if match_ids:
            default_index = ids.index(match_ids[0])

    product_choice = st.selectbox(
        "Cobertura sugerida (puedes cambiarla si hace falta)",
        ids,
        index=default_index,
        format_func=lambda x: f"{products.loc[products['id']==x,'coverage_key'].iloc[0]} — {int(products.loc[products['id']==x,'days'].iloc[0])} días",
    )
    mrow = products[products["id"] == product_choice].iloc[0]

    pol_agent = pend.loc[pend["id"] == row_id, "agent_id"].iloc[0]
    pct = 20.0
    if pol_agent is not None and pd.notna(pol_agent):
        a = fetch_df("SELECT default_commission_pct FROM agents WHERE id=:id", {"id": int(pol_agent)})
        if not a.empty:
            pct = float(a.iloc[0]["default_commission_pct"])

    commission_pct = st.number_input("% comisión", min_value=0.0, max_value=100.0, value=float(pct), step=0.5)

    st.caption(f"Sugerencia detectada desde RAW: {guess_key} — {guess_days} días")

    if st.button("Activar + crear alias", use_container_width=True):
        try:
            cov_key = mrow["coverage_key"]
            days = int(mrow["days"])
            cost = float(mrow["cost"])
            ap = float(mrow["agent_profit"])
            price = float(mrow["price"])
            cam, net = compute_financials(ap, float(commission_pct))

            run_sql(
                """
                UPDATE policies
                SET coverage_key=:ck, days=:d,
                    cost=:c, agent_profit=:ap, price=:p,
                    agent_commission_pct=:pct, agent_commission_amount=:cam, your_net_profit=:net,
                    status='ACTIVE'
                WHERE id=:id
                """,
                {
                    "ck": cov_key,
                    "d": days,
                    "c": cost,
                    "ap": ap,
                    "p": price,
                    "pct": float(commission_pct),
                    "cam": float(cam),
                    "net": float(net),
                    "id": int(row_id),
                },
            )

            run_sql(
                """
                INSERT OR IGNORE INTO coverage_alias (provider_id, raw_coverage_text, normalized_coverage_key, active)
                VALUES (:pid, :raw, :ck, 1)
                """,
                {"pid": int(pid), "raw": normalize_coverage_key(raw_text), "ck": cov_key},
            )

            st.success("Póliza activada y alias creado.")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo resolver. Error: {e}")

