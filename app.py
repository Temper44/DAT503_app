import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Streamlit Deploy Test", page_icon="🚀")
st.title("Streamlit Deploy Test")
st.write("A tiny app to verify your deployment pipeline. Choose a demo below.")

option = st.selectbox("Choose demo", ["Line chart", "Dataframe", "Upload CSV"])

if option == "Line chart":
    st.header("Random line chart")
    df = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"]) 
    st.line_chart(df)

elif option == "Dataframe":
    st.header("Sample dataframe")
    df = pd.DataFrame({"x": range(20), "y": np.random.rand(20)})
    st.dataframe(df)
    st.bar_chart(df.set_index("x"))

else:
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
