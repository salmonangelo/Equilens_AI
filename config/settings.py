"""
EquiLens AI — Application Settings

Type-safe configuration loaded from environment variables / .env file.
All settings are validated at startup via Pydantic.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "EquiLens AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # --- Server ---
    MCP_SERVER_HOST: str = "127.0.0.1"
    MCP_SERVER_PORT: int = 8000
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8080

    # --- Google Cloud ---
    GCP_PROJECT_ID: str = ""
    GCP_REGION: str = "us-central1"

    # --- Gemini / Vertex AI ---
    GEMINI_API_KEY: str = ""
    VERTEX_AI_LOCATION: str = "us-central1"
    MODEL_NAME: str = "gemini-2.0-flash"

    # --- CORS ---
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5000,http://localhost:5173"

    # --- Feature Flags ---
    ENABLE_STREAMING: bool = True
    ENABLE_CACHING: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


# Singleton instance — import this throughout the application
settings = Settings()
