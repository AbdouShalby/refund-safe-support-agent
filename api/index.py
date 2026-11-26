"""Re-export the sandboxed FastAPI API as the Vercel function entry point."""

from app.web import app as app
