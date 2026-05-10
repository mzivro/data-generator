from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    openai_temperature: float = 0.0

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    @field_validator("openai_api_key")
    def validate_openai_api_key(cls, v):
        """
        Validate OpenAI API key.

        Parameters
        ----------
        v : str
            API key string.

        Returns
        -------
        str
            Stripped API key.

        Raises
        ------
        ValueError
            If API key is empty.
        """
        v = v.strip()
        if not v:
            raise ValueError("No OpenAI API key provided")
        return v

    @field_validator("openai_temperature")
    def validate_openai_temperature(cls, v):
        """
        Validate model temperature.

        Parameters
        ----------
        v : float
            Model temperature.

        Returns
        -------
        float
            Validated model temperature.

        Raises
        ------
        ValueError
            If model temperature is not in range.
        """
        if 0.0 <= v <= 2.0:
            return v
        raise ValueError("Model temperature must be in <0; 2> range")


try:
    settings = Settings()
except ValidationError as e:
    raise RuntimeError(f"Invalid configuration:\n{e}") from e
