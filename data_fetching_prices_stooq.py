#!/usr/bin/env python3
"""
Stooq CSV Loader – 50 ausgewählte Aktien (nur EOD, finale Speicherung als CSV)

- Quelle: https://stooq.com (kostenfrei, CSV pro Ticker)
- Für jede der 50 vorgegebenen Firmen wird die tägliche Historie geladen
  und als CSV in ./data/prices/ gespeichert.
- Bereits vorhandene Dateien werden übersprungen.
- Robustes Logging + einfache Retry-Logik.

Aufruf:
  python3 stooq_loader.py

Optional:
  - ENV SLEEP_SECONDS (Default 2) zur Drosselung zwischen Anfragen

Hinweis:
  - Nicht alle internationalen Titel sind garantiert auf Stooq verfügbar (z. B. ADX, SSE, NSE).
    Das Skript loggt dafür WARNs, überspringt aber den Rest weiter.
"""

import os
import sys
import time
import csv
from pathlib import Path
from typing import Dict, Optional
import urllib.parse
import urllib.request

import io
import pandas as pd

BASE_URL_STOOQ = "https://stooq.com/q/d/l/"  # CSV per Ticker
OUT_DIR = Path("data")
PRICES_DIR = OUT_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "2"))
RETRIES = 2
TIMEOUT = 25

# Mapping: Firmenname → Stooq-Symbol
SYMBOLS_STOOQ: Dict[str, str] = {
    "Apple Inc.": "AAPL.US",
    "Microsoft Corporation": "MSFT.US",
    "Alphabet Inc. (Google)": "GOOGL.US",
    "NVIDIA Corporation": "NVDA.US",
    "Amazon.com, Inc.": "AMZN.US",
    "Meta Platforms, Inc.": "META.US",
    "TSMC – Taiwan Semiconductor Manufacturing Company": "TSM.US", 
    "Berkshire Hathaway Inc.": "BRK-A.US",
    "Tesla, Inc.": "TSLA.US",
    "Eli Lilly and Company": "LLY.US",
    "Visa Inc.": "V.US",
    "Mastercard Incorporated": "MA.US",
    "Exxon Mobil Corporation": "XOM.US",
    "Johnson & Johnson": "JNJ.US",
    "Samsung Electronics Co., Ltd.": "SMSN.UK",
    "Home Depot, Inc.": "HD.US",
    "ASML Holding N.V.": "ASML.US", 
    "Alibaba Group Holding Ltd.": "BABA.US",
    "Bank of America Corporation": "BAC.US",
    "Advanced Micro Devices, Inc.": "AMD.US",
    "LVMH Moët Hennessy Louis Vuitton SE": "0HAU.UK",
    "Procter & Gamble Co.": "PG.US",
    "UnitedHealth Group Incorporated": "UNH.US",
    "SAP SE": "SAP.DE",
    "Chevron Corporation": "CVX.US",
    "Coca‑Cola Company": "KO.US",
    "Roche Holding AG": "0QOK.UK",
    "Cisco Systems, Inc.": "CSCO.US",
    "Toyota Motor Corporation": "TM.US",
    "Nestlé S.A.": "0QR4.UK",
    "Wells Fargo & Company": "WFC.US",
    "AstraZeneca PLC": "AZN.US",
    "T‑Mobile US, Inc.": "TMUS.US",
    "Novartis AG": "NVS.US",
    "Morgan Stanley": "MS.US",
    "L’Oréal S.A.": "0NZM.UK",
    "Salesforce, Inc.": "CRM.US",
    "Philip Morris International Inc.": "PM.US",
    "Caterpillar Inc.": "CAT.US",
    "China Mobile Limited": "941.HK",
    "RTX Corporation": "RTX.US",
    "Novo Nordisk A/S": "NVO.US",
    "SK Hynix Inc.": "HY9H.DEF",
    "HSBC Holdings plc": "HSBC.US", 
    "PetroChina Company Limited": "857.HK",
    "Reliance Industries Limited": "RS.US", 
    "Micron Technology, Inc.": "MU.US",
    "Abbott Laboratories": "ABT.US",
}


def build_url_stooq(symbol: str) -> str:
    q = urllib.parse.urlencode({"s": symbol.lower(), "i": "d"})
    return f"{BASE_URL_STOOQ}?{q}"


def is_valid_stooq_csv(data: bytes) -> bool:
    if not data:
        return False
    head = data[:128].decode("utf-8", "ignore").lower()
    return ("date,open,high,low,close" in head) or ("data not found" in head)


def fetch_stooq_csv(symbol: str) -> Optional[bytes]:
    url = build_url_stooq(symbol)
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRIES + 2):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                data = resp.read()
            if not is_valid_stooq_csv(data):
                raise ValueError("Unerwartetes CSV-Format oder leerer Inhalt")
            lower = data[:256].decode("utf-8", "ignore").lower()
            if "data not found" in lower or "<html" in lower:
                raise ValueError("Ticker bei Stooq nicht gefunden")
            return data
        except Exception as e:
            last_err = e
            if attempt <= RETRIES:
                wait = 1.5 * attempt
                print(f"[RETRY {attempt}] {symbol}: {e} → warte {wait:.1f}s…")
                time.sleep(wait)
            else:
                break
    print(f"[WARN] {symbol}: Download fehlgeschlagen – {last_err}")
    return None


def save_csv(symbol: str, content: bytes) -> Path:
    out_path = PRICES_DIR / f"{symbol.replace('.', '_').upper()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Read new data into DataFrame
    new_df = pd.read_csv(io.BytesIO(content))
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            # Only keep new rows (by Date)
            if 'Date' in existing_df.columns and 'Date' in new_df.columns:
                existing_dates = set(existing_df['Date'])
                new_rows = new_df[~new_df['Date'].isin(existing_dates)]
                if not new_rows.empty:
                    # Append new rows to file
                    new_rows.to_csv(out_path, mode='a', header=False, index=False)
            else:
                # If Date column missing, just overwrite
                new_df.to_csv(out_path, index=False)
        except Exception as e:
            print(f"[WARN] Fehler beim Lesen/Anhängen an {out_path}: {e}. Überschreibe Datei.")
            new_df.to_csv(out_path, index=False)
    else:
        new_df.to_csv(out_path, index=False)
    return out_path


def main():
    companies = list(SYMBOLS_STOOQ.items())
    print(f"Starte Stooq-Download für {len(companies)} Unternehmen…")


    for name, stq in companies:
        target = PRICES_DIR / f"{stq.replace('.', '_').upper()}.csv"
        print(f"\n=== {name} → {stq} ===")
        data = fetch_stooq_csv(stq)
        if data:
            path = save_csv(stq, data)
            print(f"[OK] Gespeichert/aktualisiert: {path}")
        else:
            print("[WARN] Übersprungen (keine Daten)")
            with open("error.txt", "a") as f:
                f.write(f"=== {name} - {stq} ===\n")
        time.sleep(SLEEP_SECONDS)

    print("\nFertig. CSVs unter ./data/prices/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Abgebrochen.")
        sys.exit(130)
