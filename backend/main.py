# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.upload import router as upload_router
from backend.database import create_table

app = FastAPI()

origins = ["*"]  # Allow all for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

app.include_router(upload_router)

@app.on_event("startup")
def startup():
    create_table()

# Optional if needed to start FastAPI separately
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
