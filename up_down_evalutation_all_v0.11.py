import pandas as pd
import numpy as np
from pathlib import Path # Pfad Angabe (Linux/Windows)
from glob import glob
from lightgbm import LGBMClassifier # ML-Modell
from sklearn.metrics import ( # ML-Modell Metriken
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt #Visualisierungen
from datetime import datetime

###############################################################################
# Globals
###############################################################################
DEBUG = 1
# Unterscheidung zwischen Mini-Backtest Hold&Sell oder Hold&Sell(bei Signal = Down)
VARIANTE = 2
#from IPython.display import display, HTML

# Pfad zu den Ordnern mit den Daten
BASE_DIR = Path(r"data")
PRICE_DIR     = BASE_DIR / "prices"          # Pfad zu den Aktienkursen
FUND_DIR      = BASE_DIR / "fundamentals"    # Pfad zu den Quartalsberichten und Unternehmenskennzahlen

TARGET_HORIZON_DAYS = 5                # für wie viele Tage in die Zukunft die Vorhersage gelten soll
TEST_SPLIT_RATIO = 0.05                 # Aufteilung der Daten in Trainings- und Testdaten (z.b. 80% Training/20% Test)
PROBA_THRESHOLD = 0.55                 # Schwellwert der Wahrscheinlichkeit, ab wann die Vorhersage "UP" ausgibt und ab wann "DOWN"
VOLATILITY = 20                        # Anzahl der Tage für die Berechnung der Volatilität
MOMENTUM = 5                           # Anzahl der Tage für die Berechnung des Momentums
VOLUME = 5                             # Anzahl der Tage für die Berechnung der Volume-Dynamik

REPORT_FILE_NAME = f"Klassifikationsreport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Mapping zwischen Aktienkurs-Dateinamen und Unternehmenskennzahlen-Dateinamen
# Aktienkurs-Dateinamen ohne Suffix (_US.csv etc.)
# Unternehmenskennzahlen-Dateiname ohne _balance_sheet.csv etc.

TICKER_MAP = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "NVDA": "NVDA",
    "AMZN": "AMZN",
    "META": "META",
    "TSM": "TSM",
    "BRK-A": "BRK-A",
    "TSLA": "TSLA",
    "LLY": "LLY",
    "V": "V",
    "MA": "MA",
    "XOM": "XOM",
    "JNJ": "JNJ",
    #"SMSN": "005930.KS",
    "HD": "HD",
    "ASML": "ASML",
    "BABA": "BABA",
    "BAC": "BAC",
    "AMD": "AMD",
    #"0HAU": "MC.PA",
    "PG": "PG",
    "UNH": "UNH",
    "SAP": "SAP.DE",
    "CVX": "CVX",
    "KO": "KO",
    #"0QOK": "ROG.SW",
    "CSCO": "CSCO",
    "TM": "TM",
    #"0QR4": "NESN.SW",
    "WFC": "WFC",
    "AZN": "AZN",
    "TMUS": "TMUS",
    "NVS": "NVS",
    "MS": "MS",
    #"0NZM": "OR.PA",
    "CRM": "CRM",
    "PM": "PM",
    "CAT": "CAT",
    #"941": "0941.HK",
    "RTX": "RTX",
    "NVO": "NVO",
    #"HY9H": "000660.KS",
    #"HSBC": "HSBC",
    #"857": "0857.HK",
    #"MU": "MU",
    #"ABT": "ABT",
}


###############################################################################
# Funktionen
###############################################################################
def build_price_features(price_df: pd.DataFrame, horizon_days: int, volatility: int, momentum: int, volume : int) -> pd.DataFrame: #Rückgabehinweis -> pd.DataFrame, aber keine Überprüfung/Zwang

    # Kopieren des Dataset
    df = price_df.copy()
    # Sortieren des Dataset nach der Spalte "Date" und Entfernen des zuvorigen Index 
    df = df.sort_values("Date").reset_index(drop=True)

    # Berechnet die tägliche Rendite (Differenz zwischen den Spalten "close" und "close vom Vortag") und speichert die Ergebnisse als eigene Spalte "Return_1d" ab
    df["Return_1d"] = df["Close"].pct_change()

    # Close des Tages, welcher x Tage in der Zukunft liegt(x=Anzahl der Tage, die in die Zukunft geschaut wird [horizon_days])
    future_close = df["Close"].shift(-horizon_days)
    # (Close von x Tage in der Zukunft durch das Close heute -1) und Speicherung des Ergebnisses in der "Return_fwd"-Spalte
    df["Return_fwd"] = future_close / df["Close"] - 1.0

    # Ist der Kurs nach x Tagen höher oder niedriger? -> Speicherung des Ergebnisses in der "Target"-Spalte
    df["Target"] = (df["Return_fwd"] > 0).astype(int)

    # Berechnung des gleitende Durchschnitt für 10 Tage 
    df["SMA10"] = df["Close"].rolling(10).mean() # nimmt die nächsten 10 Werte und ermittelt den Mittelwert davon
    # Berechnung des gleitende Durchschnitt für 50 Tage
    df["SMA50"] = df["Close"].rolling(50).mean()
    # Berechnung des Verhältnis um Trends abzuleiten 
    df["SMA_ratio"] = df["SMA10"] / df["SMA50"]

    # Volatilität - Standardabweichung der täglichen Returns über x Tage
    df["Volatility"] = df["Return_1d"].rolling(volatility).std()

    # Momentum der letzten x Tage (Close von heute durch das Close x Tage in der Vergangenheit -1)
    df["Momentum"] = df["Close"] / df["Close"].shift(momentum) - 1.0

    # Volumen-Dynamik über x Tage (Differenz des Volumens zwischen Volumen des Tages und des Volumens x Tage davor)
    df["VolumeChange"] = df["Volume"].pct_change(volume)

    return df

def load_fundamentals_for_ticker(fund_base: str, fund_dir: Path) -> pd.DataFrame:

    # Pfad zum Balance Sheet
    bal_path = fund_dir / f"{fund_base}_balance_sheet.csv"
    # Pfad zum Income Statement
    inc_path = fund_dir / f"{fund_base}_income_statement.csv"

    # Falls kein Balance Sheet oder Income Statement vorhanden ist -> Return
    if (not bal_path.exists()) or (not inc_path.exists()):
        # Falls eins fehlt -> kein Fundamentals-Join möglich
        return None

    # Auslesen der CSV
    bal = pd.read_csv(bal_path, parse_dates=["fiscalDateEnding"])
    inc = pd.read_csv(inc_path, parse_dates=["fiscalDateEnding"])

    # Alle Spalten, die Zahlen enthalten könnten, werden in float oder int umgewandelt (Fehler werden zu NaN)
    for df in [bal, inc]:
        for col in df.columns:
            if col not in ["fiscalDateEnding", "reportedCurrency", "period"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Kopieren der Datasets
    #bal_small = bal.copy()
    #inc_small = inc.copy()

    # Überprüfen, ob die Spalten exisiteren, sonst NaN
    def safe_col(df, col):
        return df[col] if col in df.columns else np.nan

    # Erstellen eines neuen Balance_Sheet-Dataset mit den wichtigsten Spalten/Informationen
    bal_feat = pd.DataFrame({
        "fiscalDateEnding": safe_col(bal, "fiscalDateEnding"),
        "totalLiabilities": safe_col(bal, "totalLiabilities"),
        "totalShareholderEquity": safe_col(bal, "totalShareholderEquity"),
        "totalAssets": safe_col(bal, "totalAssets"),
    })

    # Berechnung des Verschuldungsgrades (Verbindlichkeiten / Eigenkapital) (alle Nullstellen mit np.nan ersetzen, da durch 0 dividieren nicht geht)
    bal_feat["DebtEquity"] = (
        bal_feat["totalLiabilities"] /
        bal_feat["totalShareholderEquity"].replace({0: np.nan})
    )

    # Berechnung der Eigenkapitalquote (Gesamtvermögen / Eigenkapital) (alle Nullstellen mit np.nan ersetzen, da durch 0 dividieren nicht geht)
    bal_feat["Leverage"] = (
        bal_feat["totalAssets"] /
        bal_feat["totalShareholderEquity"].replace({0: np.nan})
    )

    # Erstellen eines neuen Income_Statement-Dataset mit den wichtigsten Spalten/Informationen
    inc_feat = pd.DataFrame({
        "fiscalDateEnding": safe_col(inc, "fiscalDateEnding"),
        "totalRevenue": safe_col(inc, "totalRevenue"),
        "netIncome": safe_col(inc, "netIncome"),
        "operatingIncome": safe_col(inc, "operatingIncome"),
    })

    # Berechnung des Nettogewinn am Umsatz (Nettogewinn / Gesamtumsatz) (alle Nullstellen mit np.nan ersetzen, da durch 0 dividieren nicht geht)
    inc_feat["ProfitMargin"] = (
        inc_feat["netIncome"] /
        inc_feat["totalRevenue"].replace({0: np.nan})
    )

    # Berechnung der Betriebsergebnis-Quote (EBIT / Gesamtumsatz) (alle Nullstellen mit np.nan ersetzen, da durch 0 dividieren nicht geht)
    inc_feat["OperatingMargin"] = (
        inc_feat["operatingIncome"] /
        inc_feat["totalRevenue"].replace({0: np.nan})
    )

    # Zusammenfügen der beiden Datasets zu einem 
    fundamentals = pd.merge(
        bal_feat,
        inc_feat,
        on="fiscalDateEnding",
        how="outer"
    ).sort_values("fiscalDateEnding")

    # Neu-Benennung der Reporting-Spalte
    fundamentals = fundamentals.rename(columns={"fiscalDateEnding": "ReportDate"})
    # Neusetzen des Index
    fundamentals = fundamentals.reset_index(drop=True)
    
    return fundamentals

def merge_price_and_fundamentals(price_df: pd.DataFrame,
                                 fund_df: pd.DataFrame) -> pd.DataFrame:

    # Überprüfen, ob beide Dataset vorhanden sind
    # Falls kein Fundamentals Dataset vorhanden ist -> Aktienkurs-Dataset bleibt bestehen und Fundamentals Spalten werden mit NaN aufgefüllt
    if fund_df is None or fund_df.empty:
        # Kopieren des Aktienkurs-Datasets
        merged = price_df.copy()
        # Iteration über Fundamentals Spalten
        for col in ["DebtEquity", "Leverage", "ProfitMargin", "OperatingMargin"]:
            # Falls die Spalte nicht im merged-Dataset vorhanden ist -> dazuhängen und mit NaN auffüllen
            if col not in merged.columns:
                merged[col] = np.nan
        return merged

    # Sortieren der Datasets nach Datum und Reset des Index
    price_sorted = price_df.sort_values("Date").reset_index(drop=True)
    fund_sorted = fund_df.sort_values("ReportDate").reset_index(drop=True)

    # Zusammenführen der beiden Dataset, wobei das Mapping Unternehmensbericht-Datum <= (älter) Aktienkurs-Datum ist - letzter verfügbare Bericht vor dem Aktienkursdatum
    merged = pd.merge_asof(
        price_sorted,
        fund_sorted,
        left_on="Date",
        right_on="ReportDate",
        direction="backward"
    )

    return merged

def time_based_train_test_split(df: pd.DataFrame, test_ratio: float):
    # Sortieren des Datasets nach Datum + Index wird neu gesetzt
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    # Berechnung Anzahl der Einträge im Dataset
    n = len(df_sorted)
    # Berechnung Anzahl der Zeilen für das Training-Dataset

    split_idx = int(np.floor((1 - test_ratio) * n))
    # Verwendung als Index für das Splitten des Datasets in ein Training- und ein Test-Datasets
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    return train_df, test_df

def evaluate_model(
    clf,
    X_train, y_train,
    X_test, y_test,
    df_test_full,
    proba_threshold=0.55,
    horizon_days=5
):

    # Testen des Modell mit Daten des Test-Datasets
    y_pred = clf.predict(X_test)
    # Berechnung der Wahrscheinlichkeit, ob die Aktien steigen (1) oder nicht (0) 
    # -> clf.predict_proba(X_test) gibt ein 2D-Array zurück mit 1.Spalte, Wahrscheinlichkeit, dass der Kurs fällt und 2. Spalte, Wahrscheinlichkeit, dass der Kurs steigt
    # [:, 1] -> nehme nur die zweite Spalte
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Ausgabe des Klassifikationsreports (wie gut passt das Modell zu den Test Daten (wie gut ist die Modell-Leistung))
    print("=== Klassifikationsreport (Test) ===")
    # 0 = nicht steigen; 1 = steigen
    # precision = Wie viele der positiven Vorhersagen waren tatsächlich richtig? Formel: TP / (TP + FP)
    # recall = Wie viele der tatsächlich positiven Fälle hat das Modell erkannt? Formel: TP / (TP + FN)
    # f1-score = Harmonic mean aus Precision & Recall → Balance zwischen „Genauigkeit der positiven Vorhersagen“ und „Empfindlichkeit“
    #support = Anzahl der tatsächlichen Beispiele dieser Klasse in y_test
    # accuracy = Gesamtanteil korrekt klassifizierter Samples. Formel: (TP + TN) / Gesamtzahl
    # macro avg = Durchschnitt von Precision, Recall und F1 über alle Klassen, ohne Gewichtung (jede Klasse zählt gleich stark)
    # weighted avg = Durchschnitt von Precision, Recall und F1, gewichtet nach support (Klassen mit mehr Beispielen zählen stärker)
    print(classification_report(y_test, y_pred, digits=3))
    # Print to txt
    with open(REPORT_FILE_NAME, "w", encoding="utf-8") as f:
        print("=== Klassifikationsreport (Test) ===", file=f)
        print(classification_report(y_test, y_pred, digits=3), file=f)

    # Versuch, ob auch ein ROC-AUC-Score erstellt werden kann
    # Wie gut unterscheidet Modell „wird steigen“ vs. „wird nicht steigen“
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC: {auc:.3f}")

        with open(REPORT_FILE_NAME, "a", encoding="utf-8") as f:
            print(f"ROC-AUC: {auc:.3f}", file=f)
    except ValueError:
        print("ROC-AUC nicht berechenbar (nur eine Klasse im Test-Set).")

    # Berechnugn des Accuracy Score - misst, wie viele Vorhersagen insgesamt richtig waren
    # acc = (TP + TN) / (TP + TN + FP + FN)
    acc = accuracy_score(y_test, y_pred)
    # Berechnung der Präzision - misst, wie viele der vorhergesagten positiven Fälle tatsächlich positiv waren.
    # prec = TP / (TP + FP)
    prec = precision_score(y_test, y_pred, zero_division=0)
    # Berechnung des Recalls - misst, wie viele der tatsächlich positiven Fälle erkannt wurden.
    # rec = TP / (TP + FN)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # Ausgabe der Metriken
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")

    with open(REPORT_FILE_NAME, "a", encoding="utf-8") as f:
        print(f"Accuracy : {acc:.3f}", file=f)
        print(f"Precision: {prec:.3f}", file=f)
        print(f"Recall   : {rec:.3f}", file=f)

    if VARIANTE == 1: #30 Tage halten und dann verkaufen
        # Mini-Backtest
        strat_df = df_test_full.copy()

        # Datum und Sortierung sicherstellen
        strat_df["Date"] = pd.to_datetime(strat_df["Date"])
        strat_df = strat_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # Wahrscheinlichkeit und Signal
        strat_df["proba_up"] = y_pred_proba
        strat_df["signal_long"] = (strat_df["proba_up"] > proba_threshold).astype(int)

        # Pro Ticker Positions-Flag über <horizon_days> Tage erzeugen-
        def build_position_per_ticker(g):
            sig = g["signal_long"].to_numpy()
            n = len(g)
            pos = np.zeros(n, dtype=int)

            remaining = 0  # wie viele Tage die aktuelle Position noch läuft

            for i in range(n):
                if remaining > 0:
                    # wir sind noch in einem laufenden Trade
                    pos[i] = 1
                    remaining -= 1
                elif sig[i] == 1:
                    # neues Einstiegssignal -> neue Position eröffnen
                    pos[i] = 1
                    remaining = horizon_days - 1  # heute inklusive, daher -1

            g["position"] = pos
            return g

        strat_df = strat_df.groupby("Ticker", group_keys=False).apply(build_position_per_ticker)

        # Tages-Strategie-Return mit Return_1d
        strat_df["strategy_return_1d"] = strat_df["position"] * strat_df["Return_1d"]

        # Gleichgewichtete tägliche Portfolio-Rendite
        def equal_weight_daily(d):
            active = d["position"].sum()
            if active == 0:
                return 0.0  # kein Trade aktiv -> 0% Tagesreturn
            # Durchschnitt der Returns der aktiven Positionen
            return d.loc[d["position"] == 1, "strategy_return_1d"].mean()

        daily_return = strat_df.groupby("Date").apply(equal_weight_daily)

        # Equity-Kurve und Gesamtrendite
        equity_curve_daily = (1 + daily_return).cumprod()
        total_return_daily = equity_curve_daily.iloc[-1] - 1 if len(equity_curve_daily) else np.nan

        print(f"\n=== Up/Down-Schwellwert für potenzielle Long-Strategie: {proba_threshold} ===")
        print(f"Variante: Mehrere Aktien, {horizon_days}-Tage-Hold, tägliche Gleichgewichtung")
        print(f"Mögliche Gesamtrendite bei Verwendung des Up/Down-Schwellwerts für den Testzeitraum {strat_df['Date'].dt.date.min()} bis {strat_df['Date'].dt.date.max()}: {total_return_daily:.2%}")

        # Plot Strategie-Kapitalverlauf über Testzeitraum
        plt.figure(figsize=(8,4))
        plt.plot(equity_curve_daily.index, equity_curve_daily.values, label="Strategie-Kapitalverlauf")
        plt.title("Backtest (Test-Periode, alle Ticker)")
        plt.xlabel("Test-Index (zeitlich sortiert)")
        plt.ylabel("Kumulierte Rendite")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    elif VARIANTE ==2: #30 Tage und dann halten bis DOWN
        # Mini Backtest Variante D
        strat_df = df_test_full.copy()

        # Datum und Sortierung
        strat_df["Date"] = pd.to_datetime(strat_df["Date"])
        strat_df = strat_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # Wahrscheinlichkeit und Signal
        strat_df["proba_up"] = y_pred_proba
        strat_df["signal_long"] = (strat_df["proba_up"] > proba_threshold).astype(int)

        # Variante 2: Position so lange halten, wie Signal Long bleibt, aber mindestens horizon_days

        def build_position_varD(g):
            sig = g["signal_long"].to_numpy()
            n = len(g)
            pos = np.zeros(n, dtype=int)

            in_pos = False
            hold_days_remaining = 0  # Resttage der Mindesthaltedauer

            for i in range(n):
                if in_pos:
                    # wir sind in einer offenen Position
                    pos[i] = 1
                    if hold_days_remaining > 0:
                        hold_days_remaining -= 1
                    else:
                        # Mindesthaltedauer ist vorbei, jetzt entscheidet nur noch das Signal
                        if sig[i] == 0:
                            in_pos = False
                            pos[i] = 0  # ab heute wieder flat
                else:
                    # aktuell flat, neues Signal kann Position öffnen
                    if sig[i] == 1:
                        in_pos = True
                        pos[i] = 1
                        hold_days_remaining = horizon_days - 1  # heute inklusive

            g["position_D"] = pos
            return g

        strat_df = strat_df.groupby("Ticker", group_keys=False).apply(build_position_varD)

        # Tages Strategie Return aus Return_1d
        strat_df["strategy_return_1d_D"] = strat_df["position_D"] * strat_df["Return_1d"]

        # Gleichgewichtete tägliche Portfolio Rendite
        def equal_weight_daily_D(d):
            active = d["position_D"].sum()
            if active == 0:
                return 0.0
            return d.loc[d["position_D"] == 1, "strategy_return_1d_D"].mean()

        daily_return_D = strat_df.groupby("Date").apply(equal_weight_daily_D)

        # Equity Kurve und Gesamtrendite
        equity_curve_D = (1 + daily_return_D).cumprod()
        total_return_D = equity_curve_D.iloc[-1] - 1 if len(equity_curve_D) else np.nan

        print(f"\n=== Variante D: min {horizon_days} Tage Hold, dann solange Signal Long bleibt ===")
        print(f"Schwellwert: {proba_threshold}")
        print(f"Mögliche Gesamtrendite bei Verwendung des Up/Down-Schwellwerts für den Testzeitraum {strat_df['Date'].dt.date.min()} bis {strat_df['Date'].dt.date.max()}: {total_return_D:.2%}")

        # Plot Strategie-Kapitalverlauf über Testzeitraum
        plt.figure(figsize=(8,4))
        plt.plot(equity_curve_D.index, equity_curve_D.values, label="Strategie-Kapitalverlauf")
        plt.title("Backtest (Test-Periode, alle Ticker)")
        plt.xlabel("Test-Index (zeitlich sortiert)")
        plt.ylabel("Kumulierte Rendite")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else: # Jeden Tag den 30 Tage ertrag rechnen -> FALSCH
        # Mini-Backtest
        # Kopieren des Test Datasets und Reset des Index
        strat_df = df_test_full.copy().reset_index(drop=True)

        # Speichern der Wahrscheinlichkeit, dass die Aktien steigen, in das Dataset
        strat_df["proba_up"] = y_pred_proba
        # Erstellen einer Empfehlung (für einen potenziellen Long), wenn die Wahrscheinlichkeit über einen gewissen Schwellwert ist
        strat_df["signal_long"] = (strat_df["proba_up"] > proba_threshold).astype(int)

        # Strategie-Return  - gibt den potenziellen Return für x Tage, wenn die Empfehlung für Long ist.
        strat_df["strategy_return"] = strat_df["signal_long"] * strat_df["Return_fwd"]

        # Pro Tag alle aktiven Long Signale gleichgewichtet mitteln:
        daily_return = strat_df.groupby("Date")["strategy_return"].mean().fillna(0.0)

        # Equity Kurve (tägliche kumulierte Performance)
        equity_curve_daily = (1 + daily_return).cumprod() - 1

        # Gesamtrendite der Strategie am Ende des Testzeitraums
        total_return_daily = equity_curve_daily.iloc[-1] if len(equity_curve_daily) else np.nan

        # Kapitalverlauf - kumulierte Renditen - Zeitreihe der kumulativen Performance - Equity-Kurve
        #equity_curve = (1 + strat_df["strategy_return"].fillna(0)).cumprod() - 1
        # Gesamtreturn am Ende des Testzeitraums
        # Greift den letzten Wert der "euqity-curve"-Liste ab, wenn Daten vorhanden sind
        #total_return_strategy = equity_curve.iloc[-1] if len(equity_curve) else np.nan

        print(f"\n=== Up/Down-Schwellwert für potenzielle Long-Strategie: {proba_threshold}) ===")
        # Achtung: ohne Gebühren, Slippage, etc.
        #print(f"Mögliche Gesamtrendite bei Verwendung des Up/Down-Schwellwerts für den Testzeitraum (alle Aktien): {total_return_strategy:.2%}")
        print(f"Mögliche Gesamtrendite bei Verwendung des Up/Down-Schwellwerts für den Testzeitraum {strat_df['Date'].dt.date.min()} bis {strat_df['Date'].dt.date.max()} (alle Aktien): {total_return_daily:.2%}")
    
        # Plot Strategie-Kapitalverlauf über Testzeitraum
        plt.figure(figsize=(8,4))
        plt.plot(equity_curve_daily.index, equity_curve_daily.values, label="Strategie-Kapitalverlauf")
        plt.title("Backtest (Test-Periode, alle Ticker)")
        plt.xlabel("Test-Index (zeitlich sortiert)")
        plt.ylabel("Kumulierte Rendite")
        plt.legend()
        plt.tight_layout()
        plt.show()


###############################################################################
# Main
###############################################################################
def main():
    # Alle verfügbaren Aktienkurs-Dateien finden (optional mit _xx oder _xx Endungen; z.b. _US.csv)
    price_files = glob(str(PRICE_DIR / "*_??.csv")) + glob(str(PRICE_DIR / "*_???.csv"))

    # Initialisierung der panel_rows Liste
    panel_rows = []

    # Iteration durch alle gefundenen Aktienkurs-Dateien
    for pf in price_files:
        pf_path = Path(pf)

        # Dateienname ohne Suffix z.B. "AAPL_US"
        basename = pf_path.stem

        # Aufteilung des Dateiennamen und entfernen der Länderkennung z.B. APPL_US -> APPL
        if "_" in basename:
            ticker_candidate = "_".join(basename.split("_")[:-1])
        else:
            ticker_candidate = basename

        # Check ob Dateienname im Mapping (Aktienkurs-Unternehmenskennzahlen exisitiert)
        if ticker_candidate not in TICKER_MAP:
            # Falls nicht -> Warnung in der Commandozeile ausgeben
            print(f"[WARN] Kein Mapping für {ticker_candidate}, überspringe.")
            continue

        # Aus dem Mapping den richtigen Dateinamen der Unternehmenskennzahlen-Dateinamen finden
        fund_base = TICKER_MAP[ticker_candidate]

        # Aktienkursdatei (.csv) einlesen
        price_raw = pd.read_csv(pf_path, parse_dates=["Date"])
        # Groß-/Kleinschreibung anpassen
        price_raw = price_raw.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        # Liste aller benötigten Spalten aus den Aktienkurs-CSVs
        needed_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        # Überprüfen, ob alle Spalten verfügbar sind -> Falls nicht: Warnung auf der Kommandozeile ausgeben
        missing_cols = [c for c in needed_cols if c not in price_raw.columns]
        if missing_cols:
            print(f"[WARN] {pf_path.name} fehlt Spalten {missing_cols}, überspringe.")
            continue

        # Aufrufen der build_price_features-Funktion - laden und aufbereitung der Aktienkurs-Daten
        price_feat = build_price_features(price_raw, TARGET_HORIZON_DAYS, VOLATILITY, MOMENTUM, VOLUME)
        # Hinzufügen einer Spalte um die Daten einer Aktie zuordnen zu können
        price_feat["Ticker"] = ticker_candidate

        # Aufrufen der load_fundamentals_for_ticker-Funktion - laden und aufbereitung der Fundamentals-Daten
        fund_df = load_fundamentals_for_ticker(fund_base, FUND_DIR)
        # Aufrufen der merge_price_and_fundamentals-Funktion - Zusammenführen der beiden Datasets
        merged = merge_price_and_fundamentals(price_feat, fund_df)

        # Hinzufügen des Datasets zur Liste fürs spätere trainieren 
        panel_rows.append(merged)

    # Überprüfen. ob die Liste nicht leer ist
    if len(panel_rows) == 0:
        raise RuntimeError("Kein Eintrag in der Trainingsliste gefunden! - Mapping prüfen!")

    # Zusammenführen aller Datasets zu einem (untereinander-merge) inkl. Neuindexierung
    full_panel = pd.concat(panel_rows, ignore_index=True)

    # Featureliste fürs Modell
    FEATURE_COLUMNS = [
        # Aktienkurs Features
        "SMA_ratio",
        "Volatility",
        "Momentum",
        "VolumeChange",

        # Fundamentale Features
        "DebtEquity",
        "Leverage",
        "ProfitMargin",
        "OperatingMargin",
    ]


    # Umwandeln der Akitennamen in Kateogiern für die Verarbeitung mit LightGBM
    full_panel["Ticker_cat"] = full_panel["Ticker"].astype("category").cat.codes
    # Erstellen einer Liste mit allen wichtigen Informationen für das Modell
    FEATURE_COLUMNS_WITH_TICKER = FEATURE_COLUMNS + ["Ticker_cat"]

    # Erstellen eines neuen Datasets mit nur den wichtigen Spalten + alle NaN werden bereinigt - 
    # in der TARGET_HORIZON_DAYS-Zeitspanne kann kein Return_fwd und Target berechnet werden - daher nicht brauchbar für Training und Test
    # aber später für die Vorhersage braucht man keines von beiden!!! -> daher am Schluss wieder full_panel verwenden
    model_df = full_panel.dropna(
        subset=FEATURE_COLUMNS_WITH_TICKER + ["Target", "Return_fwd", "Date"]
    ).copy()

    # Aufrufen der time_based_train_test_split-Funktion - Aufteilen des Datasets in ein Training- und ein Test-Datasets
    train_df, test_df = time_based_train_test_split(model_df, TEST_SPLIT_RATIO)

    # Selektieren der wichtigsten Daten fürs Trainieren ohne Datum, Return_fwd und Target Spalte
    X_train = train_df[FEATURE_COLUMNS_WITH_TICKER]
    # Verwenden der Target Spalte als Ergebnis
    y_train = train_df["Target"]

    # Selektieren der wichtigsten Daten fürs Testen ohne Datum, Return_fwd und Target Spalte
    X_test = test_df[FEATURE_COLUMNS_WITH_TICKER]
    # Verwenden der Target Spalte als Ergebnis
    y_test = test_df["Target"]

    # LightGBM Klassifier konfigurieren
    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    # Modell mit Trainingsdaten füttern
    clf.fit(X_train, y_train)

    # Aufruf der evaluate_model-Funktion - Modell-Metriken, Backtest
    evaluate_model(
        clf,
        X_train, y_train,
        X_test, y_test,
        df_test_full=test_df,
        proba_threshold=PROBA_THRESHOLD,
        horizon_days=TARGET_HORIZON_DAYS,
    )

    # Anzeigen, wie wichtig ein Feature/Datenspalte für das Modell war 
    # und wieviel es zur Verbesserung des Modells beigetragen hat
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS_WITH_TICKER)
    print("\n=== Feature Importances (globales Modell) ===")
    print(importances.sort_values(ascending=False))

    with open(REPORT_FILE_NAME, "a", encoding="utf-8") as f:
        print("\n=== Feature Importances (globales Modell) ===", file=f)
        print(importances.sort_values(ascending=False), file=f)

    # Plot, welches Feature wie wichtig für das Trainieren des Modells war
    plt.figure(figsize=(7,4))
    importances.sort_values().plot(kind="barh")
    plt.title("LightGBM - Feature Importance (alle Aktien)")
    plt.tight_layout()
    plt.show()

    # Erstellen einer Liste, wo alle Empfehlungen und Wahrscheinlichkeiten pro Aktie gespeichert werden 
    latest_signals_tst = []
    # Iteration durch alle Aktien einzeln
    for ticker_name, grp in full_panel.groupby("Ticker"):
        # Sortieren der vorhandenen Daten nach Datum + Abspeichern der letzten Zeile mit dem aktuellsten Datum
        last_row = grp.sort_values("Date").iloc[[-1]]
        # Berechnung der Wahrscheinlichkeit, ob die Aktie steigt (1) oder nicht (0) - 2D-Array hat nur einen Eintrag
        proba_up = clf.predict_proba(last_row[FEATURE_COLUMNS_WITH_TICKER])[:, 1][0]
        # Vorhersage ob es steigen wird (1) oder nicht (0) 
        #pred_up = clf.predict(last_row[FEATURE_COLUMNS_WITH_TICKER])[0]
        # Speichern, der Emfpehlungen in die Liste
        latest_signals_tst.append({
            "Ticker": ticker_name,
            "Date": last_row["Date"].values[0],
            "ProbUp": proba_up,
            #"Signal": "UP" if pred_up == 1 else "DOWN"
            "Signal": "UP" if proba_up > PROBA_THRESHOLD else "DOWN"
        })        
   
    # Speichern der gesamte List in einem Dataset + Sotierung nach Wahrscheinlichkeit
    latest_signals_tst_df = pd.DataFrame(latest_signals_tst).sort_values("ProbUp", ascending=False)
    # Ausgabe des Ergebnis
    print("\n=== Aktuelle Einschätzung TST(pro Ticker, letztes Datum) ===")
    print(latest_signals_tst_df.to_string(index=False))
    # Speichern als JSON
    latest_signals_tst_df.to_json("results_stock_prediction.json", orient="records", indent=4)
           
        
# Call der Main-Funktion zum Starten des Skripts
if __name__ == "__main__":
    main()
