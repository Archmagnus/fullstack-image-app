# app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Data Ingestion Dashboard", layout="wide")
st.title("📥 Upload Images or Data Files")

file = st.file_uploader("Upload an image, CSV, or Excel file", type=["jpg", "png", "csv", "xlsx"])

if file is not None:
    st.success(f"Uploading: {file.name}")
    files = {"file": (file.name, file, file.type)}
    response = requests.post("http://localhost:8000/upload-file/", files=files)

    if response.status_code == 200:
        st.success("✅ Upload successful!")
    else:
        st.error("❌ Upload failed.")

# Future: Display uploaded file records from DB
