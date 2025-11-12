# Streamlit Deploy Test

This repository contains a minimal Streamlit app (`app.py`) used to verify your deployment pipeline.

Prerequisites

- Python 3.8+ installed
- Recommended: create a virtual environment

Windows PowerShell (recommended steps):

```powershell
python -m venv .venv;
.\.venv\Scripts\Activate.ps1;
pip install -r requirements.txt;
streamlit run app.py
```

If your deployment target uses a different start command, point it to `streamlit run app.py`.
