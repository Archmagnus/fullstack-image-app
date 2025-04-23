from fastapi import APIRouter, HTTPException, Form

router = APIRouter(prefix="/auth")

# Fake users (in real app, connect to DB)
users = {
    "admin": {"password": "adminpass", "role": "admin"},
    "user": {"password": "userpass", "role": "user"},
}

@router.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if username not in users or users[username]["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": "Login successful", "role": users[username]["role"]}
