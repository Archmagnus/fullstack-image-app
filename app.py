from backend.auth import router as auth_router
from backend.upload import router as upload_router
from backend.main import run_main_app  # if it's defined in main.py

import streamlit as st
from backend.main import run_main_app  # assuming you defined a run_main_app() in main.py

st.set_page_config(page_title="Image App", layout="wide")

run_main_app()
