from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Add any specific settings required by the application here
    # For example:
    # spotify_client_id: str | None = None
    # spotify_client_secret: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
