import streamlit as st
import requests
import pandas as pd
from io import BytesIO

st.title("Resume Analyzer Webapp")
st.write("Upload a CSV file to make predictions")

file = st.file_uploader("Upload your CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write(df.head())

    # Prepare file for POST
    files = {"file": file.getvalue()}

    # Prediction API
    api_url = "http://127.0.0.1:8000/predict/"
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        updated_csv = response.json().get("file")
        st.download_button("Download CSV with Predictions", updated_csv, file_name="predictions.csv")

        # Reload updated dataframe to use in plot
        updated_df = pd.read_csv(BytesIO(updated_csv.encode()))
        if 'actual' not in updated_df.columns:
            st.warning("No 'actual' column found for plotting. Please add actual values in CSV before uploading.")
        else:
            # Resend updated CSV with prediction & actual
            buffer = BytesIO()
            updated_df.to_csv(buffer, index=False)
            buffer.seek(0)
            plot_response = requests.post("http://127.0.0.1:8000/plot/", files={"file": buffer.getvalue()})
            
            if plot_response.status_code == 200:
                st.image(plot_response.content)
            else:
                st.error("Plotting failed.")
    else:
        st.error("Prediction failed. Check your FastAPI server.")
