import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="Streamlit Deploy Test", page_icon="🚀")
st.title("Streamlit Deploy Test2")
st.write("A tiny app to verify your deployment pipeline. Choose a demo below.")

# Show timestamp created by the GitHub Actions test workflow (if present)
import os, json
OWNER = "MatEbner"   # your GitHub username
REPO  = "DAT503_app" # your repo name
RAW_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/results/results_stock_prediction.json"

# Load and display stock prediction results from GitHub (RAW_URL)
st.header("Latest Stock Model Results (from GitHub)")
df_results = None
try:
    r = requests.get(RAW_URL, timeout=5)
    if r.status_code == 200:
        # Try to load as list of dicts (array of results)
        payload = r.json()
        if isinstance(payload, list):
            import pandas as pd
            df_results = pd.DataFrame(payload)
        elif isinstance(payload, dict) and "results" in payload:
            import pandas as pd
            df_results = pd.DataFrame(payload["results"])
except Exception as e:
    st.info(f"Could not load results from GitHub: {e}")

if df_results is not None and not df_results.empty:
    # Convert epoch-ms to datetime for Date column
    if "Date" in df_results.columns:
        try:
            df_results["Date"] = pd.to_datetime(df_results["Date"], unit="ms")
        except Exception:
            pass
    st.dataframe(df_results)
    # Show a summary: top 5 by ProbUp
    if "ProbUp" in df_results.columns:
        st.subheader("Top 5 Stocks by ProbUp")
        st.table(df_results.sort_values("ProbUp", ascending=False).head(5)[["Ticker", "Date", "ProbUp", "Signal"]])
else:
    st.info("No remote stock prediction results found. Run the model and push results to GitHub.")

# time_path = os.path.join("results_stock_prediction.json")
# if os.path.exists(time_path):
#     try:
#         with open(time_path, "r", encoding="utf-8") as f:
#             payload = json.load(f)
#             ts = payload.get("time")
#             if ts:
#                 st.markdown(f"**Latest workflow run time:** {ts}")
#     except Exception:
#         # ignore parse errors
#         pass

option = st.selectbox("Choose demo", ["Line chart", "Dataframe", "Upload CSV", "Results (results.json)", "Price: V (Visa)"])

if option == "Line chart":
    st.header("Random line chart")
    df = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"]) 
    st.line_chart(df)

elif option == "Dataframe":
    st.header("Sample dataframe")
    df = pd.DataFrame({"x": range(20), "y": np.random.rand(20)})
    st.dataframe(df)
    st.bar_chart(df.set_index("x"))

elif option == "Results (results.json)":
    st.header("Model results (results.json)")
    import os
    path = os.path.join("results.json")
    if os.path.exists(path):
        try:
            df = pd.read_json(path)
        except Exception as e:
            st.error(f"Failed to read results.json: {e}")
        else:
            # Convert epoch-ms to datetime and format
            if "Date" in df.columns:
                try:
                    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
                except Exception:
                    # if Date is already readable, ignore
                    pass

            # Allow filtering
            st.write(f"Loaded {len(df)} rows from `results.json`.")
            cols = df.columns.tolist()
            with st.sidebar.expander("Results filters"):
                signals = df["Signal"].unique().tolist() if "Signal" in df.columns else []
                selected_signals = st.multiselect("Signal", options=signals, default=signals)
                min_prob = st.slider("Minimum ProbUp", 0.0, 1.0, 0.0, 0.01)
                top_n = st.number_input("Top N to chart", min_value=1, max_value=100, value=20)

            filtered = df.copy()
            if "Signal" in filtered.columns and selected_signals:
                filtered = filtered[filtered["Signal"].isin(selected_signals)]
            if "ProbUp" in filtered.columns:
                filtered = filtered[filtered["ProbUp"] >= min_prob]

            filtered = filtered.sort_values(by="ProbUp", ascending=False).reset_index(drop=True)
            st.dataframe(filtered)

            # Chart top N
            if not filtered.empty and "Ticker" in filtered.columns and "ProbUp" in filtered.columns:
                chart_df = filtered.head(top_n).set_index("Ticker")["ProbUp"]
                st.subheader("Top entries by ProbUp")
                st.bar_chart(chart_df)

            # Download
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("Download filtered results as CSV", data=csv, file_name="results_filtered.csv", mime="text/csv")
    else:
        st.info("No `results.json` file found in the project folder. Add it to the repo or use the Upload CSV option to display data.")

elif option == "Price: V (Visa)":
    st.header("Visa (V) — Price chart")
    import os
    csv_path = os.path.join("prices", "V_US.csv")
    if os.path.exists(csv_path):
        try:
            prices = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to read {csv_path}: {e}")
        else:
            # Expect columns: Date,Open,High,Low,Close,Volume
            if "Date" in prices.columns:
                try:
                    prices["Date"] = pd.to_datetime(prices["Date"])
                except Exception:
                    # maybe epoch ms
                    try:
                        prices["Date"] = pd.to_datetime(prices["Date"], unit="ms")
                    except Exception:
                        pass
            prices = prices.sort_values("Date")
            prices = prices.set_index("Date")

            st.write(f"Showing {len(prices)} rows from `{csv_path}`")

            # Date range selector
            min_date = prices.index.min().date()
            max_date = prices.index.max().date()
            start, end = st.slider("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            sel = prices.loc[str(start):str(end)]

            # Line chart of Close price
            if "Close" in sel.columns:
                st.subheader("Close price")
                st.line_chart(sel["Close"])

            # OHLC area (Open/High/Low/Close) if available
            if all(c in sel.columns for c in ["Open", "High", "Low", "Close"]):
                st.subheader("OHLC (Close, Open, High, Low)")
                st.area_chart(sel[["Open", "High", "Low", "Close"]])

            # Volume
            if "Volume" in sel.columns:
                st.subheader("Volume")
                st.bar_chart(sel["Volume"])

            # Allow CSV download of the selected range
            csv_bytes = sel.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("Download selected range as CSV", data=csv_bytes, file_name="V_prices_selected.csv", mime="text/csv")
    else:
        st.info(f"No price file found at `{csv_path}`. Make sure the `prices` folder contains V_US.csv")

elif option == "Upload CSV":
    st.header("Upload a CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"]) 
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(df)
            # show numeric columns as a chart if any
            num = df.select_dtypes(include=[np.number])
            if not num.empty:
                st.line_chart(num)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

st.markdown("---")
st.caption("This app is intentionally minimal — it's just to test deployment and routing.")

# create here down