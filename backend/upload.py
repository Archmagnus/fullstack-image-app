from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from pathlib import Path
import shutil

router = APIRouter(prefix="/upload")

UPLOAD_FOLDER = Path("uploaded_files")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@router.post("/images")
def upload_images(file: UploadFile = File(...)):
    if not file.filename.endswith((".zip", ".rar")):
        raise HTTPException(status_code=400, detail="Only zip/rar files are allowed")

    destination = UPLOAD_FOLDER / file.filename
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"{file.filename} uploaded successfully"}
