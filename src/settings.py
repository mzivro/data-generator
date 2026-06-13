from pydantic import BaseModel, field_validator


class Settings(BaseModel):
    """
    Application configuration loaded from environment variables.

    Attributes
    ----------
    api_key : str
        API key for OpenAI services.
    model : str
        OpenAI model.
    temperature : float
        Sampling temperature for the model.
    """

    api_key: str
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0

    @field_validator("api_key")
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

    @field_validator("model")
    def validate_model(cls, v):
        """
        Validate and normalize the model name.

        Parameters
        ----------
        v : str
            Model name provided by the user.

        Returns
        -------
        str
            Trimmed model name.

        Raises
        ------
        ValueError
            If the model name is empty after removing whitespace.
        """
        v = v.strip()

        if not v:
            raise ValueError("Model cannot be empty")

        return v
