# backend/upload.py
import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File
from backend.database import insert_file

router = APIRouter()

@router.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    file_type = file.filename.split(".")[-1].lower()

    # Save raw file
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # Store content if it's a text-based file
    parsed_text = ""
    if file_type in ["csv", "xlsx"]:
        df = pd.read_csv(file_path) if file_type == "csv" else pd.read_excel(file_path)
        parsed_text = df.to_csv(index=False)

    insert_file(file.filename, file_type, parsed_text)
    return {"message": "File uploaded successfully", "filename": file.filename}
