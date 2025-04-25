import os
import pandas as pd
import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="📊 Data Ingestion Dashboard", layout="wide")
st.title("📥 Upload Files (Image / CSV / Excel)")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_details = {
        "filename": uploaded_file.name,
        "filetype": uploaded_file.type,
        "filesize": f"{round(len(uploaded_file.getvalue())/1024, 2)} KB"
    }
    st.write("📄 File details:", file_details)

    # For data files
    if uploaded_file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ File uploaded and parsed successfully.")
            st.dataframe(df.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not read the file: {e}")

    # For image files
    elif uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    # Upload to backend
    with st.spinner("📡 Uploading to backend..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post("http://localhost:8000/upload-file/", files=files)
            if response.status_code == 200:
                st.success("✅ File uploaded to backend successfully.")
            else:
                st.error("❌ Upload failed.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to backend. Is FastAPI running?")
