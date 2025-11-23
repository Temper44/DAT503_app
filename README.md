# Share Analytic Dashboard App

> DAT_503 Group 40 ESA1

## Generelle Informationen

Dieses Repository enthält die Daten und den Code für unser ESA 1 Projekt in DAT_503. Das Endergebnis ist ein Prototyp eines Analyse-Dashboards zur Entscheidungsunterstützung.

Das Projekt besteht grob aus zwei Hauptteilen, das KI-Modell selbst und der Streamlit App.
Es kann sowohl lokal gestartet werden als auch [deployed](https://share-analytic.streamlit.app/) angesehen werden. Die deployte Version ist ein automatisierter Workflow, der immer den aktuell verfügbaren Datenbestand nutzt. Bei der lokalen Ausführung sind folgende Schritte beachten:

1. Das Fetch-Script `data_fetching_prices_stooq.py` ausführen, um die aktuellen Daten zu erhalten
2. Das Modell-Script `up_down_evalutation_all_v0.11.py` ausführen, um neueste Ergebnisse zu erhalten
3. (optional) Streamlit App ausführen

## Verzeichnisstruktur, Files und Autor

Um einen Überblick zu geben, werden folgend die wichtigsten Files erklärt. Zusätzlich wird der Autor angegeben.

- `.devcontainer/devcontainer.json`: VS Code dev container file - ME
- `.Github/workflows/fetch-evaluation.yml`: GitHub Actions workflow script - ME
- `data/fundamentals/*.csv`: datasets (fundamental data) consumed by scripts - LS
- `data/prices/*.csv`: datasets (historical price data) consumed by scripts - LS
- `reports/Klassifikationsreport_*.txt`: performance reports generated after model evaluation runs - LS
- `tabs/classification_tab.py`: Streamlit classification tab content - ME
- `tabs/probability_tab.py`: Streamlit probability tab content - ME
- `tabs/share_tab.py`: Streamlit share tab content - ME
- `utils/data_loaders.py`: Helper functions to load data for streamlit - ME
- `.gitignore`: Specifies files and folders excluded from version control - ME
- `app.py`: Main Streamlit application entry point - ME
- `data_fetching_prices_stooq.py`: script that fetches latest stock price data - LS
- `README_updown_v0.11.md`: detailed documentation for evaluation script - LS
- `README.md`: project information (the current file) - ME
- `requirements.txt`: package dependencies needed to run streamlit app - ME
- `results_stock_prediction.json`: output (predictions/probabilities) for currently processed data - LS
- `up_down_evalutation_all_v0.11.py`: script to train/evaluate the classification model & export results - LS

## Systemarchitektur & Automatisierungsprozess

Diese Erklärung beschreibt wie der deployte System automatisiert abläuft und welche Komponenten beteiligt sind.

Mit Hilfe eines GitHub Action Workflows, das einen Cron-Job jeden Tag um 01:00 Uhr und 13:00 MEZ ausführt, werden zwei Tasks durchgeführt:

1. Ausführen des Scripts `data_fetching_prices_stooq.py` um neue Daten in `data/prices` zu fetchen. Wenn eine Aktualiserung der Daten stattgefunden hat -> commit & push
2. Ausführen des Scripts `up_down_evalutation_all_v0.11.py` um neues `results_stock_prediction.json` & `Klassifikationsreport*\*.txt` zu erhalten. Wenn eine Aktualiserung der Daten stattgefunden hat -> commit & push

Da am GitHub Remote Repository die neuen Daten mittels commit & push am main branch aktualisert werden, bezieht auch Streamlit den neuen Stand. Der Prototyp ist mittels Streamlit Community Cloud, das mit dem Repository verknüpft ist, deployed.

## Lokale Ausführung

### Ausführung des Modells Scripts(`up_down_evalutation_all_v0.11.py`)

Bitte schauen Sie für die Ausführung des Scripts in [data_fetching_prices_stooq](README_updown_v0.11.md)

### Ausführung des Fetch Scripts (`data_fetching_prices_stooq.py`)

Bitte schauen Sie für die Ausführung des Scripts in [data_fetching_prices_stooq.py](data_fetching_prices_stooq.py)

### Ausführung der Streamlit App (`app.py`)

**Voraussetzungen**

- Python 3.8+ installiert
- Empfohlen: erzeuge ein virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Verantwortlichkeiten & AI Einsatz

**Autoren dieses Repositorys**: Lorenz Siedler (LS) und Mathias Ebner (ME)

### Lorenz Siedler

Für die Ideensuche, Hilfestellungen bei der Modellauswahl inkl. Mini-Backtest, Datenaufbereitung und finales Strukturieren und Überarbeiten des Codes, sowie für das Aufsetzen, Strukturieren und Ausformulieren des README habe ich ChatGPT - Modell GPT-5.1, November 2025 verwendet

### Mathias Ebner

**Recherche:** Für die Ideensuche, Recherche und Hilfestellung zur Umsetzung dieses Projektes wurde sowohl Perplexity (Student Abo) im Search Mode und Chat GPT-5.0 (free Version) beide November 2025 eingesetzt.

**Umsetzung:** Für die Erstellung und Refaktoring des Codes & Kommentare in den einzelnen Files wurde AI-Assisted Coding angewandt. Dazu wurde VS-Code GitHub CoPilot (Student Abo) mit den Modellen GPT-4.1 & GPT-5 beide November 2025 verwendet. Für die Durchführung wurden der Ask Mode für Umsetzungs- und Implmentierungsfragen und der Agent Mode für Codevorschläge, Kommentarteile und UI-Texte verwendet.

**Text:** Für das Optimieren und die leichte Textanpassung dieses README.md files wurde Chat GPT-5.0 (free Version) November 2025 eingesetzt.
