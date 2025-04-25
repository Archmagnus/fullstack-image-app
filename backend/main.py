import streamlit as st
import requests
import zipfile
import os
from io import BytesIO
import time

st.set_page_config(page_title="📊 Data Ingestion Dashboard", layout="wide")
st.title("📥 Upload Files (Image / CSV / Excel)")

uploaded_file = st.file_uploader("Upload a zip file containing images", type=["zip"])

if uploaded_file is not None:
    st.write("📄 File details:", uploaded_file.name)

    # Unzip and display the image files in the zip
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        image_names = zip_ref.namelist()
        st.write(f"Found {len(image_names)} images:")
        for image in image_names:
            st.write(f" - {image}")

    # Ask if user wants to run SegNet on the images
    run_dl = st.button("Run SegNet Model on Images")
    if run_dl:
        # Display the progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Send the zip file to backend and get the results
        with st.spinner("Running SegNet on images..."):
            response = requests.post("http://localhost:8000/upload-file/", files={"file": uploaded_file})
            if response.status_code == 200:
                results = response.json()
                status_text.success("✅ SegNet processing complete.")
                # Display the results (visualize segmented output if possible)
                st.write(results)
                progress_bar.progress(100)
            else:
                status_text.error("❌ Something went wrong with the backend.")
                progress_bar.progress(0)
