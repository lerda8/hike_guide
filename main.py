from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select
from database import create_db_and_tables, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    create_db_and_tables()
    # Ensure upload directory exists
    import os
    os.makedirs(os.path.join("static", "uploads"), exist_ok=True)
    # Create dummy user for dev
    with Session(engine) as session:
        from models import User
        if not session.exec(select(User).where(User.username == "demo")).first():
            user = User(username="demo", email="demo@example.com", password_hash="dummy")
            session.add(user)
            session.commit()
    yield

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

from routers import hikes
app.include_router(hikes.router)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
