"""
EquiLens AI — Application Entrypoint

Launches the backend FastAPI server (local dev) with comprehensive logging
and environment configuration.

Usage (Local Development):
    # Default: Start backend on localhost:8080
    python main.py

    # With environment variables
    APP_ENV=local BACKEND_PORT=8000 python main.py

    # See all options
    python main.py --help

Environment Variables:
    - APP_ENV: development | local | staging | production (default: development)
    - BACKEND_HOST: Server bind address (default: 0.0.0.0)
    - BACKEND_PORT: Server port (default: 8080)
    - LOG_LEVEL: DEBUG | INFO | WARNING | ERROR (default: INFO)
    - GEMINI_API_KEY: API key for Gemini explanations (optional for local testing)
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from config.settings import settings


# =====================================================================
# Logging Configuration
# =====================================================================

def setup_logging() -> logging.Logger:
    """Configure logging for the application."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    return logger


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    """Start the FastAPI backend server."""
    parser = argparse.ArgumentParser(
        description="EquiLens AI — Fairness & Bias Auditing Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start backend (default)
  python main.py

  # Start with custom port
  BACKEND_PORT=9000 python main.py

  # Debug mode with verbose logging
  DEBUG=true LOG_LEVEL=DEBUG python main.py

  # Production-like setup
  APP_ENV=production DEBUG=false python main.py
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["backend", "mcp"],
        default="backend",
        help="Server mode: 'backend' for FastAPI (default), 'mcp' for MCP server",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload on file changes (for production)",
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    # Log startup info
    logger.info("=" * 80)
    logger.info(f"🚀 Starting {settings.APP_NAME}")
    logger.info("=" * 80)
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    logger.info(f"Mode: {args.mode}")

    # Validate critical settings
    if args.mode == "backend" and not settings.GEMINI_API_KEY:
        logger.warning(
            "⚠️  GEMINI_API_KEY not set. LLM explanations will be unavailable."
        )

    if args.mode == "backend":
        logger.info(f"Server will run at http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}")
        logger.info("📖 API Documentation available at http://localhost:8080/docs")
        logger.info("=" * 80)
        logger.info("")

        reload = settings.DEBUG and not args.no_reload

        uvicorn.run(
            "backend.app:app",
            host=settings.BACKEND_HOST,
            port=settings.BACKEND_PORT,
            reload=reload,
            log_level=settings.LOG_LEVEL.lower(),
        )
    elif args.mode == "mcp":
        logger.info(f"MCP Server will run at http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}")
        logger.info("=" * 80)
        logger.info("")

        reload = settings.DEBUG and not args.no_reload

        uvicorn.run(
            "mcp_server.server:app",
            host=settings.MCP_SERVER_HOST,
            port=settings.MCP_SERVER_PORT,
            reload=reload,
            log_level=settings.LOG_LEVEL.lower(),
        )


if __name__ == "__main__":
    main()
