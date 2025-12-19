# Seguros Transmigrantes — Streamlit + Google Sheets

Esta versión elimina el **SQLite local (C:\\Apps...)** y guarda todo en **Google Sheets**.

## 1) Crea el Google Sheet (estructura)
Crea un Spreadsheet y agrega estas pestañas (o deja que la app las cree automáticamente):
- `providers`
- `offices`
- `agents`
- `products`
- `coverage_alias`
- `policies`

La app creará cada hoja con el encabezado correcto si no existe.

## 2) Crea el Service Account
1. En Google Cloud → crea un **Service Account**
2. Descarga el JSON
3. Comparte el Spreadsheet con el email del service account como **Editor**

## 3) Configura secretos
Copia `.streamlit/secrets.example.toml` a `.streamlit/secrets.toml` y:
- pon tu `GSHEETS_SPREADSHEET_ID`
- pega el JSON del service account dentro de `[gcp_service_account]`

## 4) Instala y corre local
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## 5) Migrar tu SQLite actual (opcional)
Si ya tienes datos en `db.sqlite`, corre:
```bash
python tools/migrate_sqlite_to_gsheets.py \
  --sqlite_path "/ruta/a/db.sqlite" \
  --spreadsheet_id "TU_ID" \
  --service_account_json "/ruta/a/service_account.json"
```

## Nota importante
- Para despliegue en Streamlit Community Cloud, pega los secretos en Settings → Secrets.
- El PIN está hardcodeado en `src/app.py` (variable `APP_PIN`).
