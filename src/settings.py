from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Attributes
    ----------
    openai_api_key : str
        API key for OpenAI services.
    openai_model : str
        OpenAI model.
    openai_temperature : float
        Sampling temperature for the model.
    """

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0, env="OPENAI_TEMPERATURE")

    class Config:
        """
        Pydantic configuration.

        Attributes
        ----------
        env_file : str
            Path to the environment variables file.
        """

        env_file = ".env"


settings = Settings()
