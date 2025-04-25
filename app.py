import streamlit as st
import requests
from PIL import Image
import io

st.title("Segmentation Results")

uploaded_file = st.file_uploader("Upload a zip file containing images", type=["zip"])

if uploaded_file is not None:
    st.write("📄 File details:", uploaded_file.name)

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
                for result in results['results']:
                    img_bytes = io.BytesIO(result['segmentation_result'])
                    seg_img = Image.open(img_bytes)
                    st.image(seg_img, caption=result['image_name'])
            else:
                status_text.error("❌ Something went wrong with the backend.")
                progress_bar.progress(0)
