"""
EquiLens AI — MCP Server Application

FastAPI application for the local MCP server.
Handles tool registration, resource exposure, and lifecycle management.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from mcp_server.routes import router as mcp_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle: startup and shutdown hooks."""
    # --- Startup ---
    # TODO: Initialize fairness engine, load models, warm caches
    print(f"🚀 {settings.APP_NAME} MCP Server starting...")
    yield
    # --- Shutdown ---
    # TODO: Cleanup resources
    print(f"🛑 {settings.APP_NAME} MCP Server shutting down...")


app = FastAPI(
    title=f"{settings.APP_NAME} — MCP Server",
    description="Model Context Protocol server for AI-powered fairness analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---
app.include_router(mcp_router, prefix="/mcp", tags=["MCP"])


@app.get("/", tags=["Root"])
async def root():
    """Server identity endpoint."""
    return {
        "service": settings.APP_NAME,
        "type": "mcp-server",
        "version": "0.1.0",
        "status": "running",
    }
