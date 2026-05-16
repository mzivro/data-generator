import pytest
from pydantic import ValidationError

from settings import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings(openai_api_key="sk-test")
        assert settings.openai_model == "gpt-4.1-mini"
        assert settings.openai_temperature == 0.0

    def test_custom_values(self):
        settings = Settings(
            openai_api_key="sk-test",
            openai_model="gpt-4o",
            openai_temperature=1.5,
        )
        assert settings.openai_model == "gpt-4o"
        assert settings.openai_temperature == 1.5

    def test_api_key_is_stripped(self):
        settings = Settings(openai_api_key="  sk-test  ")
        assert settings.openai_api_key == "sk-test"

    @pytest.mark.parametrize("api_key", ["", "   "])
    def test_empty_api_key_raises(self, api_key):
        with pytest.raises(ValidationError, match="No OpenAI API key provided"):
            Settings(openai_api_key=api_key)

    @pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
    def test_valid_temperature(self, temperature):
        settings = Settings(openai_api_key="sk-test", openai_temperature=temperature)
        assert settings.openai_temperature == temperature

    @pytest.mark.parametrize("temperature", [-0.1, 2.1])
    def test_invalid_temperature_raises(self, temperature):
        with pytest.raises(ValidationError, match="Model temperature must be in"):
            Settings(openai_api_key="sk-test", openai_temperature=temperature)
