import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")
    # SQLite by default (zero setup). For MySQL set DATABASE_URL, e.g.
    #   mysql+pymysql://user:password@localhost/medvision
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///" + (BASE_DIR / "instance" / "medvision.db").as_posix())
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_DIR = os.environ.get("UPLOAD_DIR", str(BASE_DIR / "instance" / "uploads"))
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # --- OCR ---
    # Folder holding your fine-tuned model (the 'final_model' folder). Change with the MODEL_DIR env var.
    MODEL_DIR = os.environ.get("MODEL_DIR", str(BASE_DIR / "trocr-rxhandbd-checkpoints" / "final_model"))
    OCR_MODE = os.environ.get("OCR_MODE", "auto")   # auto = real model if it loads, else DEMO mode | real | mock
    NUM_BEAMS = 5
    SINGLE_REGION_MAX_SIDE = 700   # images this small (like RxHandBD word crops) are read as ONE region
